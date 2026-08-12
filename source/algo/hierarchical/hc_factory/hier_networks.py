"""Neural nets for hierarchical RL: structured StateEncoder + Q MLP."""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Masked mean over ``dim``. ``mask`` broadcastable to ``x``, values in {0,1}."""
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=x.dtype)
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(1e-6)
    return num / den


class QNetwork(nn.Module):
    """MLP Q-network for masked discrete action selection."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class StateEncoder(nn.Module):
    """Encode full ``preprocess_for_buffer`` dict → fixed vector ``z`` (default 256-D).

    Branches: progress / ongoing(+subtask Transformer) / human / robot /
    machine / material / storage, then trunk MLP.
    """

    def __init__(
        self,
        out_dim: int = 256,
        emb_dim: int = 32,
        hidden: int = 128,
        # vocab sizes (pad/unk headroom; state_id uses offset encoding up to ~2015)
        num_task: int = 32,
        num_subtask: int = 32,
        num_task_type: int = 8,
        num_product: int = 8,
        num_machine: int = 32,
        num_submat: int = 16,
        num_mat_state: int = 16,
        num_entity_state: int = 2048,
        max_ongoing: int = 5,
        max_subtasks: int = 16,
        max_material: int = 16,
        num_agents: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.max_ongoing = max_ongoing
        self.max_subtasks = max_subtasks
        self.max_material = max_material
        self.num_agents = num_agents

        self.task_emb = nn.Embedding(num_task, emb_dim, padding_idx=0)
        self.subtask_emb = nn.Embedding(num_subtask, emb_dim, padding_idx=0)
        self.task_type_emb = nn.Embedding(num_task_type, emb_dim, padding_idx=0)
        self.product_emb = nn.Embedding(num_product, emb_dim, padding_idx=0)
        self.machine_emb = nn.Embedding(num_machine, emb_dim, padding_idx=0)
        self.submat_emb = nn.Embedding(num_submat, emb_dim, padding_idx=0)
        self.mat_state_emb = nn.Embedding(num_mat_state, emb_dim, padding_idx=0)
        self.state_emb = nn.Embedding(num_entity_state, emb_dim, padding_idx=0)

        # progress: scalars(7) + product slots
        self.progress_mlp = nn.Sequential(
            nn.Linear(7 + 4 * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
        )

        # ongoing slot features (ids + scalars) before / after subtask pool
        slot_in = emb_dim * 8 + 8 + 3 + 3 + emb_dim  # ids + cont + areas + sub_pool
        self.slot_mlp = nn.Sequential(nn.Linear(slot_in, hidden), nn.ReLU(), nn.Linear(hidden, 64))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=transformer_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.subtask_transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.ongoing_out = nn.Sequential(nn.Linear(64, 128), nn.ReLU())

        # movers: state + areas(2) + route_start/end(12) + progress/len/has/detour/yield(5)
        mover_in = emb_dim + 2 + 12 + 5
        self.human_mlp = nn.Sequential(nn.Linear(mover_in + 1, 64), nn.ReLU())  # +subtask_time
        self.robot_mlp = nn.Sequential(nn.Linear(mover_in, 64), nn.ReLU())
        self.human_out = nn.Linear(64, 64)
        self.robot_out = nn.Linear(64, 64)

        # machine workstation
        ws_in = emb_dim + 1 + 1  # state emb + proc_time + task_rec norm
        self.ws_mlp = nn.Sequential(nn.Linear(ws_in, 32), nn.ReLU())
        self.machine_mlp = nn.Sequential(nn.Linear(emb_dim + 32, 64), nn.ReLU())

        self.material_mlp = nn.Sequential(nn.Linear(emb_dim * 2 + 1, 32), nn.ReLU())
        self.storage_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1 + max_material, 32),
            nn.ReLU(),
        )

        # trunk: 64+128+64+64+64+32+32 = 448
        trunk_in = 64 + 128 + 64 + 64 + 64 + 32 + 32
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, out_dim),
            nn.ReLU(),
        )

    def _clamp_id(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return x.long().clamp(0, n - 1)

    def _encode_progress(self, prog: dict) -> torch.Tensor:
        # (B, 7) + product type slots qty/not_started/finished
        scalars = prog["scalars"]
        if scalars.ndim == 1:
            scalars = scalars.unsqueeze(0)
        b = scalars.shape[0]
        extras = torch.stack(
            [
                prog["product_order_qty"],
                prog["not_started_n"],
                prog["finished_n"],
            ],
            dim=-1,
        )  # (B, P, 3)
        if extras.ndim == 2:
            extras = extras.unsqueeze(0)
        flat = extras.reshape(b, -1)
        return self.progress_mlp(torch.cat([scalars, flat], dim=-1))

    def _encode_ongoing(self, ot: dict) -> torch.Tensor:
        mask = ot["mask"]
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        b, m = mask.shape

        def _e(name: str, emb: nn.Embedding) -> torch.Tensor:
            t = ot[name]
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return emb(self._clamp_id(t, emb.num_embeddings))

        task = _e("task_id", self.task_emb)
        ttype = _e("task_type_id", self.task_type_emb)
        prod = _e("product_id", self.product_emb)
        mach = _e("machine_id", self.machine_emb)
        log_m = _e("logistic_machine_id", self.machine_emb)
        submat = _e("logistic_submat_id", self.submat_emb)
        next_t = _e("next_task_id", self.task_emb)
        next_l = _e("next_logistic_id", self.task_emb)

        def _f(name: str) -> torch.Tensor:
            t = ot[name].float()
            return t.unsqueeze(0) if t.ndim == 1 else t

        cont = torch.stack(
            [
                _f("task_done"),
                _f("is_final"),
                _f("age_norm"),
                _f("ongoing_index"),
                _f("num_subtasks_n"),
                _f("human_slot").clamp_min(-1).float() / 10.0,
                _f("robot_slot").clamp_min(-1).float() / 10.0,
                _f("workstation_i").clamp_min(-1).float() / 4.0,
            ],
            dim=-1,
        )  # (B, M, 8)
        start_a = ot["start_area_ids"].float()
        goal_a = ot["goal_area_ids"].float()
        if start_a.ndim == 2:
            start_a = start_a.unsqueeze(0)
            goal_a = goal_a.unsqueeze(0)
        start_a = start_a / 300.0
        goal_a = goal_a / 300.0

        # subtask transformer: (B, M, S, 4) → (B, M, emb)
        seq = ot["subtask_seq"]
        smask = ot["subtask_mask"]
        if seq.ndim == 3:
            seq = seq.unsqueeze(0)
            smask = smask.unsqueeze(0)
        # embed 4 agent columns and sum
        s_emb = 0
        for j in range(self.num_agents):
            s_emb = s_emb + self.subtask_emb(self._clamp_id(seq[..., j], self.subtask_emb.num_embeddings))
        mat_seq = ot["mat_state_seq"]
        if mat_seq.ndim == 2:
            mat_seq = mat_seq.unsqueeze(0)
        s_emb = s_emb + self.mat_state_emb(self._clamp_id(mat_seq, self.mat_state_emb.num_embeddings))

        bm = b * m
        s_flat = s_emb.reshape(bm, self.max_subtasks, self.emb_dim)
        pad = smask.reshape(bm, self.max_subtasks) < 0.5  # True = ignore
        # all-pad rows break transformer; force first token valid
        all_pad = pad.all(dim=-1)
        pad = pad.clone()
        pad[all_pad, 0] = False
        tok = self.subtask_transformer(s_flat, src_key_padding_mask=pad)
        tok_mask = (~pad).float().unsqueeze(-1)
        sub_pool = (tok * tok_mask).sum(dim=1) / tok_mask.sum(dim=1).clamp_min(1e-6)
        sub_pool = sub_pool.reshape(b, m, self.emb_dim)

        slot = torch.cat(
            [task, ttype, prod, mach, log_m, submat, next_t, next_l, cont, start_a, goal_a, sub_pool],
            dim=-1,
        )
        slot_h = self.slot_mlp(slot)
        pooled = masked_mean(slot_h, mask, dim=1)
        return self.ongoing_out(pooled)

    def _encode_movers(self, group: dict, is_human: bool) -> torch.Tensor:
        mask = group["mask"]
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        state = group["state_id"]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        st = self.state_emb(self._clamp_id(state, self.state_emb.num_embeddings))

        def _get(name: str, nd: int) -> torch.Tensor:
            t = group[name].float()
            if t.ndim == nd - 1:
                t = t.unsqueeze(0)
            return t

        cur = _get("current_area_id", 2) / 300.0
        tgt = _get("target_area_id", 2) / 300.0
        rs = _get("route_start", 3)
        re = _get("route_end", 3)
        rp = _get("route_progress", 2)
        rl = _get("route_length", 2) / 4000.0
        hr = _get("has_route", 2)
        det = _get("detour_active", 2)
        yld = _get("yield_active", 2)
        feats = [st, cur.unsqueeze(-1), tgt.unsqueeze(-1), rs, re, rp.unsqueeze(-1), rl.unsqueeze(-1), hr.unsqueeze(-1), det.unsqueeze(-1), yld.unsqueeze(-1)]
        if is_human:
            stc = _get("subtask_time_counter", 2) / 100.0
            feats.append(stc.unsqueeze(-1))
            h = self.human_mlp(torch.cat(feats, dim=-1))
            h = self.human_out(h)
        else:
            h = self.robot_mlp(torch.cat(feats, dim=-1))
            h = self.robot_out(h)
        return masked_mean(h, mask, dim=1)

    def _encode_machines(self, mach: dict) -> torch.Tensor:
        mask = mach["mask"]
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mid = mach["machine_id"]
        if mid.ndim == 1:
            mid = mid.unsqueeze(0)
        m_emb = self.machine_emb(self._clamp_id(mid, self.machine_emb.num_embeddings))

        ws_mask = mach["workstation_mask"]
        state = mach["state_id"]
        proc = mach["processing_time_step"]
        trec = mach["ongoing_task_record_index"]
        if ws_mask.ndim == 2:
            ws_mask = ws_mask.unsqueeze(0)
            state = state.unsqueeze(0)
            proc = proc.unsqueeze(0)
            trec = trec.unsqueeze(0)
        st = self.state_emb(self._clamp_id(state, self.state_emb.num_embeddings))
        ws_feat = self.ws_mlp(
            torch.cat(
                [st, (proc.float() / 100.0).unsqueeze(-1), (trec.float().clamp_min(-1) / 10.0).unsqueeze(-1)],
                dim=-1,
            )
        )
        ws_pool = masked_mean(ws_feat, ws_mask, dim=2)
        h = self.machine_mlp(torch.cat([m_emb, ws_pool], dim=-1))
        return masked_mean(h, mask, dim=1)

    def _encode_materials(self, mat: dict) -> torch.Tensor:
        mask = mat["mask"]
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        pid = mat["product_id"]
        fid = mat["finished_task_id"]
        trec = mat["ongoing_task_record_index"]
        if pid.ndim == 1:
            pid = pid.unsqueeze(0)
            fid = fid.unsqueeze(0)
            trec = trec.unsqueeze(0)
        h = self.material_mlp(
            torch.cat(
                [
                    self.product_emb(self._clamp_id(pid, self.product_emb.num_embeddings)),
                    self.task_emb(self._clamp_id(fid, self.task_emb.num_embeddings)),
                    (trec.float().clamp_min(-1) / 10.0).unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        return masked_mean(h, mask, dim=1)

    def _encode_storage(self, stor: dict) -> torch.Tensor:
        mask = stor["mask"]
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        sid = stor["state_id"]
        tid = stor["material_type_id"]
        nmat = stor["num_material"]
        idx = stor["material_idx_list"]
        if sid.ndim == 1:
            sid = sid.unsqueeze(0)
            tid = tid.unsqueeze(0)
            nmat = nmat.unsqueeze(0)
            idx = idx.unsqueeze(0)
        h = self.storage_mlp(
            torch.cat(
                [
                    self.state_emb(self._clamp_id(sid, self.state_emb.num_embeddings)),
                    self.submat_emb(self._clamp_id(tid, self.submat_emb.num_embeddings)),
                    (nmat.float() / 10.0).unsqueeze(-1),
                    idx.float().clamp_min(-1) / 10.0,
                ],
                dim=-1,
            )
        )
        return masked_mean(h, mask, dim=1)

    def forward(self, pre: dict) -> torch.Tensor:
        """``pre``: single-sample or batched preprocess dict → ``(B, out_dim)``."""
        # detect batch from progress.scalars
        scalars = pre["progress"]["scalars"]
        squeeze = scalars.ndim == 1

        h = torch.cat(
            [
                self._encode_progress(pre["progress"]),
                self._encode_ongoing(pre["progress"]["ongoing_tasks"]),
                self._encode_movers(pre["human"], is_human=True),
                self._encode_movers(pre["robot"], is_human=False),
                self._encode_machines(pre["machine"]),
                self._encode_materials(pre["material"]),
                self._encode_storage(pre["storage"]),
            ],
            dim=-1,
        )
        z = self.trunk(h)
        if squeeze:
            z = z.squeeze(0)
        return z
