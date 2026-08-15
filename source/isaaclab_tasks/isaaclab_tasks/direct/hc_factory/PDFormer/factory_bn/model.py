"""BNPDFormer: PDFormer dense forecast + STGNPP event prediction.

Borrowing map
-------------
* Jiang et al. PDFormer (AAAI'23)
    - STSelfAttention: geo / semantic / temporal heads
    - Delay-aware pattern keys injected into geo Key
    - Dense multi-step bottleneck_score regression

* Jin et al. STGNPP (AAAI'23)
    - SpatioTemporalInquirer over historical bottleneck events
    - Continuous GRU (flow between events + discrete at events)
    - Periodic-gated cumulative intensity + NLL for next onset
    - Duration head (MAE)
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from factory_bn.backbone import DataEmbedding, STEncoderBlock, TokenEmbedding
from factory_bn.stgnpp import ContinuousGRU, PeriodicGatedIntensity, SpatioTemporalInquirer


class BNPDFormer(nn.Module):
    def __init__(self, config: dict[str, Any], data_feature: dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_feature = data_feature

        self.num_nodes = int(data_feature["num_nodes"])
        self.feature_dim = int(data_feature["feature_dim"])
        self.output_dim = int(config.get("output_dim", 1))
        self.input_window = int(config.get("input_window", 12))
        self.output_window = int(config.get("output_window", 1))
        self.embed_dim = int(config.get("embed_dim", 64))
        self.skip_dim = int(config.get("skip_dim", 128))
        self.s_attn_size = int(config.get("s_attn_size", 3))
        self.t_attn_size = int(config.get("t_attn_size", 1))
        self.far_mask_delta = int(config.get("far_mask_delta", 4))
        self.dtw_delta = int(config.get("dtw_delta", 5))
        self.lape_dim = int(config.get("lape_dim", 8))
        self.device = config.get("device", torch.device("cpu"))

        self.w_score = float(config.get("w_score", 1.0))
        self.w_will = float(config.get("w_will", 0.5))
        self.w_mark = float(config.get("w_mark", 0.3))
        self.w_event = float(config.get("w_event", 1.0))
        self.w_tts = float(config.get("w_tts", 0.1))
        self.w_cause = float(config.get("w_cause", 0.4))
        self.pos_weight = float(config.get("will_pos_weight", 20.0))
        self.use_stgnpp = bool(config.get("use_stgnpp", True))

        self.n_cause_classes = int(
            data_feature.get("n_cause_classes") or len(data_feature.get("cause_classes") or [])
        )
        if self.n_cause_classes <= 0:
            self.n_cause_classes = 10
        cause_w = data_feature.get("cause_class_weight")
        if cause_w is None:
            cause_w = np.ones(self.n_cause_classes, dtype=np.float32)
        self.register_buffer(
            "cause_class_weight",
            torch.as_tensor(np.asarray(cause_w, dtype=np.float32)),
        )

        adj_mx = data_feature["adj_mx"]
        sh_mx = data_feature["sh_mx"]
        sem_mx = data_feature["sem_mx"]
        pattern_keys = data_feature.get("pattern_keys")
        if pattern_keys is None:
            pattern_keys = np.zeros((16, self.s_attn_size, self.output_dim), dtype=np.float32)

        geo_mask = torch.zeros(self.num_nodes, self.num_nodes)
        geo_mask[sh_mx >= self.far_mask_delta] = 1
        self.register_buffer("geo_mask", geo_mask.bool())

        sem_mask = torch.ones(self.num_nodes, self.num_nodes)
        nn_idx = np.argsort(sem_mx, axis=1)[:, : self.dtw_delta]
        for i in range(self.num_nodes):
            sem_mask[i, nn_idx[i]] = 0
        self.register_buffer("sem_mask", sem_mask.bool())

        self.register_buffer(
            "pattern_keys",
            torch.from_numpy(np.asarray(pattern_keys, dtype=np.float32)),
        )
        self.register_buffer("lap_mx", self._laplacian_pe(adj_mx, self.lape_dim))

        drop = float(config.get("drop", 0.0))
        attn_drop = float(config.get("attn_drop", 0.0))
        drop_path = float(config.get("drop_path", 0.1))
        enc_depth = int(config.get("enc_depth", 4))
        geo_num_heads = int(config.get("geo_num_heads", 2))
        sem_num_heads = int(config.get("sem_num_heads", 2))
        t_num_heads = int(config.get("t_num_heads", 4))
        n_heads = geo_num_heads + sem_num_heads + t_num_heads
        if self.embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim={self.embed_dim} must be divisible by {n_heads}")

        self.pattern_embeddings = nn.ModuleList(
            [TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)]
        )
        self.enc_embed_layer = DataEmbedding(
            self.feature_dim,
            self.embed_dim,
            self.lape_dim,
            adj_mx,
            drop=drop,
            add_time_in_day=False,
            add_day_in_week=False,
            device=self.device,
        )
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList(
            [
                STEncoderBlock(
                    dim=self.embed_dim,
                    s_attn_size=self.s_attn_size,
                    t_attn_size=self.t_attn_size,
                    geo_num_heads=geo_num_heads,
                    sem_num_heads=sem_num_heads,
                    t_num_heads=t_num_heads,
                    mlp_ratio=float(config.get("mlp_ratio", 4)),
                    qkv_bias=True,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=enc_dpr[i],
                    act_layer=nn.GELU,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    device=self.device,
                    type_ln=config.get("type_ln", "pre"),
                    output_dim=self.output_dim,
                )
                for i in range(enc_depth)
            ]
        )
        self.skip_convs = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, self.skip_dim, kernel_size=1) for _ in range(enc_depth)]
        )
        self.end_conv1 = nn.Conv2d(self.input_window, self.output_window, kernel_size=1)
        self.end_conv2 = nn.Conv2d(self.skip_dim, self.output_dim, kernel_size=1)

        # ---- Auxiliary window heads (sparse-data friendly) ----
        hidden = int(config.get("event_hidden", 64))
        self.aux_pool = nn.Sequential(nn.Linear(self.embed_dim, hidden), nn.ReLU())
        self.will_head = nn.Linear(hidden + 1, 1)
        self.cause_head = nn.Linear(hidden + 1, self.n_cause_classes)
        self.node_score_gate = nn.Linear(self.embed_dim, 1)
        self.tts_aux = nn.Sequential(nn.Linear(hidden + 1, 1), nn.Softplus())

        # ---- STGNPP event path ----
        self.inquirer = SpatioTemporalInquirer(self.embed_dim)
        self.cont_gru = ContinuousGRU(
            self.embed_dim, n_flow_layers=int(config.get("n_flow_layers", 2))
        )
        self.intensity = PeriodicGatedIntensity(self.embed_dim, hidden=hidden)
        # seed state when a node has no historical events: use last encoder state
        self.empty_event_proj = nn.Linear(self.embed_dim, self.embed_dim)

    @staticmethod
    def _laplacian_pe(adj: np.ndarray, lape_dim: int) -> torch.Tensor:
        import scipy.sparse as sp

        adj_sp = sp.coo_matrix(adj)
        d = np.asarray(adj_sp.sum(1)).flatten()
        isolated = int(np.sum(d == 0))
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)
        lap = sp.eye(adj.shape[0]) - d_mat @ adj_sp @ d_mat
        eigval, eigvec = np.linalg.eigh(lap.toarray())
        idx = np.argsort(eigval)
        eigvec = np.real(eigvec[:, idx])
        start = isolated + 1
        end = start + lape_dim
        if end > eigvec.shape[1]:
            pe = np.zeros((adj.shape[0], lape_dim), dtype=np.float32)
            avail = eigvec[:, start:]
            pe[:, : avail.shape[1]] = avail
        else:
            pe = eigvec[:, start:end].astype(np.float32)
        return torch.from_numpy(pe)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns score_pred (B,Tout,N,1), enc (B,T,N,D), h_last (B,N,D)."""
        B, T, N, _ = x.shape
        x_pattern_list = []
        for i in range(self.s_attn_size):
            x_pattern = F.pad(
                x[:, : T + i + 1 - self.s_attn_size, :, : self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0),
                "constant",
                0,
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)

        pat_embs, key_embs = [], []
        for i in range(self.output_dim):
            pat_embs.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1))
            key_embs.append(self.pattern_embeddings[i](self.pattern_keys[..., i]).unsqueeze(-1))
        x_patterns = torch.cat(pat_embs, dim=-1)
        pattern_keys = torch.cat(key_embs, dim=-1)

        enc = self.enc_embed_layer(x, self.lap_mx)
        skip = 0
        for i, block in enumerate(self.encoder_blocks):
            enc = block(enc, x_patterns, pattern_keys, self.geo_mask, self.sem_mask)
            skip = skip + self.skip_convs[i](enc.permute(0, 3, 2, 1))

        h_last = enc[:, -1, :, :]
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        score_pred = skip.permute(0, 3, 2, 1)
        return score_pred, enc, h_last

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        score_pred, enc, h_last = self.encode(batch["X"])

        # auxiliary graph-level will / mark from last hidden
        node_energy = self.node_score_gate(h_last).squeeze(-1)  # (B, N)
        pooled = self.aux_pool(h_last).mean(dim=1)
        max_e = node_energy.max(dim=1).values.unsqueeze(-1)
        feat = torch.cat([pooled, max_e], dim=-1)
        will_logit = self.will_head(feat).squeeze(-1)
        tts_aux = self.tts_aux(feat).squeeze(-1)
        cause_logits = self.cause_head(feat)

        out: dict[str, torch.Tensor] = {
            "score_pred": score_pred,
            "will_logit": will_logit,
            "mark_logits": node_energy,
            "cause_logits": cause_logits,
            "tts_aux": tts_aux,
            "h_last": h_last,
        }

        if self.use_stgnpp and "event_idx" in batch:
            H_e = self.inquirer(
                enc, batch["event_idx"], batch["event_dur"], batch["event_mask"]
            )
            h_evt = self.cont_gru(H_e, batch["event_mask"], batch["inter_tau"])
            # nodes without history: fall back to last encoder state
            has_hist = (batch["event_mask"].sum(dim=-1) > 0).unsqueeze(-1).float()
            h_seed = self.empty_event_proj(h_last)
            h_evt = has_hist * h_evt + (1.0 - has_hist) * h_seed
            out["h_event"] = h_evt
            # intensity predictions for next tau (for inference)
            phase = batch["phase"]
            if phase.dim() == 2:
                phase_exp = phase.unsqueeze(1).expand(-1, self.num_nodes, -1)
            else:
                phase_exp = phase
            # evaluate Λ at predicted / given next_tau for monitoring
            tau_q = batch["next_tau"].clamp_min(1e-3)
            with torch.enable_grad():
                # no graph needed here for forward-only Lam
                Lam = self.intensity.cumulative(h_evt, tau_q.detach(), phase_exp)
            out["Lam"] = Lam
            out["dur_event"] = self.intensity.duration(h_evt)

        return out

    def calculate_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        out = self.forward(batch)
        y_score = batch["y_score"]
        will = batch["will"]
        mark = batch["mark"]

        loss_score = F.smooth_l1_loss(out["score_pred"], y_score)
        pos_weight = torch.tensor([self.pos_weight], device=will.device)
        loss_will = F.binary_cross_entropy_with_logits(
            out["will_logit"], will, pos_weight=pos_weight
        )

        pos_mask = (will > 0.5) & (mark >= 0)
        if pos_mask.any():
            loss_mark = F.cross_entropy(out["mark_logits"][pos_mask], mark[pos_mask])
            tts_n = batch["tts"][pos_mask] / 60.0
            loss_tts = F.smooth_l1_loss(out["tts_aux"][pos_mask] / 60.0, tts_n)
        else:
            zero = out["score_pred"].sum() * 0.0
            loss_mark = zero
            loss_tts = zero

        cause = batch.get("cause")
        if cause is not None:
            valid_cause = cause >= 0
            if valid_cause.any():
                w = self.cause_class_weight
                if w.numel() != out["cause_logits"].shape[-1]:
                    w = None
                loss_cause = F.cross_entropy(
                    out["cause_logits"][valid_cause],
                    cause[valid_cause],
                    weight=w,
                )
            else:
                loss_cause = out["score_pred"].sum() * 0.0
        else:
            loss_cause = out["score_pred"].sum() * 0.0

        loss_event = out["score_pred"].sum() * 0.0
        nll_v = loss_event
        dur_v = loss_event
        if self.use_stgnpp and "next_mask" in batch and batch["next_mask"].sum() > 0:
            h_evt = out["h_event"]
            phase = batch["phase"]
            if phase.dim() == 2:
                phase_exp = phase.unsqueeze(1).expand(-1, self.num_nodes, -1)
            else:
                phase_exp = phase
            # flatten valid next-event nodes
            m = batch["next_mask"] > 0.5
            loss_event, stats_e = self.intensity.nll_and_duration(
                h_evt[m],
                batch["next_tau"][m].clamp_min(1e-3),
                batch["next_dur"][m],
                torch.ones(m.sum(), device=m.device),
                phase_exp[m],
                dur_weight=1.0,
            )
            nll_v = stats_e["nll"]
            dur_v = stats_e["dur_mae"]

        total = (
            self.w_score * loss_score
            + self.w_will * loss_will
            + self.w_mark * loss_mark
            + self.w_tts * loss_tts
            + self.w_event * loss_event
            + self.w_cause * loss_cause
        )
        stats = {
            "loss": float(total.detach().cpu()),
            "loss_score": float(loss_score.detach().cpu()),
            "loss_will": float(loss_will.detach().cpu()),
            "loss_mark": float(loss_mark.detach().cpu()),
            "loss_tts": float(loss_tts.detach().cpu()),
            "loss_event": float(loss_event.detach().cpu()) if torch.is_tensor(loss_event) else float(loss_event),
            "loss_cause": float(loss_cause.detach().cpu()) if torch.is_tensor(loss_cause) else float(loss_cause),
            "nll": float(nll_v.detach().cpu()) if torch.is_tensor(nll_v) else float(nll_v),
            "dur_mae": float(dur_v.detach().cpu()) if torch.is_tensor(dur_v) else float(dur_v),
        }
        return total, stats

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.eval()
        # intensity NLL path needs grad for ∂Λ/∂τ only in training; inference uses duration + Lam
        out = self.forward(batch)
        out["will_prob"] = torch.sigmoid(out["will_logit"])
        out["mark_prob"] = torch.softmax(out["mark_logits"], dim=-1)
        out["cause_prob"] = torch.softmax(out["cause_logits"], dim=-1)
        out["cause_pred"] = out["cause_prob"].argmax(dim=-1)
        return out
