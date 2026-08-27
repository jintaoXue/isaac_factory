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

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from factory_bn.backbone import DataEmbedding, STEncoderBlock, TokenEmbedding
from factory_bn.stgnpp import ContinuousGRU, PeriodicGatedIntensity, SpatioTemporalInquirer


OCC_TYPE_NAMES = ("machine", "gantry", "agv")
OCC_TYPE_ALIASES = {
    "machine": ("machine",),
    "gantry": ("gantry",),
    "agv": ("transport_robot", "agv"),
}


def apply_hot_type_affine(
    logits: torch.Tensor,
    type_masks: dict[str, torch.Tensor],
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Per-type scale/bias on occupancy logits. scale=1, bias=0 is identity."""
    out = logits
    for i, name in enumerate(OCC_TYPE_NAMES):
        mask = type_masks.get(name)
        if mask is None:
            continue
        mask = mask.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        if mask.numel() != logits.shape[-1] or float(mask.sum()) <= 0:
            continue
        m = mask.view(1, 1, -1)
        out = out * (1.0 + (scale[i] - 1.0) * m) + bias[i] * m
    return out


def occupancy_bce_cell_weight(
    y_hot: torch.Tensor,
    type_masks: dict[str, torch.Tensor],
    *,
    default_pos_weight: float,
    pos_weight_by_type: dict[str, float] | None = None,
    fp_weight_by_type: dict[str, float] | None = None,
) -> torch.Tensor:
    """Per-cell BCE multiplier: positives × pos_weight[type], negatives × fp_weight[type].

    Gantry occupancy is ~19% of cells; a global pos_weight=4 makes FN four times
    costlier than FP and is the main reason gantry over-predicts.
    """
    pos_map = {str(k): float(v) for k, v in (pos_weight_by_type or {}).items()}
    fp_map = {str(k): float(v) for k, v in (fp_weight_by_type or {}).items()}
    default_w = 1.0 + (float(default_pos_weight) - 1.0) * y_hot
    w = default_w
    tagged = torch.zeros_like(y_hot)
    for name in OCC_TYPE_NAMES:
        node = type_masks.get(name)
        if node is None:
            continue
        node = node.to(device=y_hot.device, dtype=y_hot.dtype).reshape(-1)
        if node.numel() != y_hot.shape[-1] or float(node.sum()) <= 0:
            continue
        m = node.view(1, 1, -1)
        pw = float(pos_map.get(name, default_pos_weight))
        fw = float(fp_map.get(name, 1.0))
        type_w = y_hot * pw + (1.0 - y_hot) * fw
        w = torch.where(m > 0.5, type_w, w)
        tagged = torch.maximum(tagged, m.expand_as(y_hot))
    return torch.where(tagged > 0.5, w, default_w)


def occupancy_cell_weight(
    step_w: torch.Tensor,
    occ_node_mask: torch.Tensor | None,
) -> torch.Tensor:
    """(B, K, N) = near-horizon step weight × labeled occupancy columns.

    Human / buffer stay 0. Machine, gantry, and AGV are supervised.
    """
    weight = step_w.unsqueeze(-1)
    if occ_node_mask is None:
        return weight
    node = occ_node_mask.float()
    if node.dim() == 2:
        return weight * node.unsqueeze(1)
    return weight * node


def occupancy_type_node_masks(
    resource_types: list[str] | None,
    num_nodes: int,
) -> dict[str, torch.Tensor]:
    """Per-type (N,) 0/1 masks. AGV aliases transport_robot."""
    types = [str(t).strip().lower() for t in (resource_types or [])]
    out: dict[str, torch.Tensor] = {}
    for name, aliases in OCC_TYPE_ALIASES.items():
        mask = torch.zeros(int(num_nodes), dtype=torch.float32)
        for i, t in enumerate(types[: int(num_nodes)]):
            if t in aliases:
                mask[i] = 1.0
        out[name] = mask
    return out


def type_balanced_occupancy_losses(
    logits: torch.Tensor,
    y_hot: torch.Tensor,
    hot_m: torch.Tensor,
    type_masks: dict[str, torch.Tensor],
    *,
    hot_pos_weight: float,
    w_dice: float,
    w_iou: float,
    pos_weight_by_type: dict[str, float] | None = None,
    fp_weight_by_type: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean BCE / Dice / IoU over types that have any weight this batch.

    Gantry cells (~4× more positives) cannot dominate the occupancy loss.
    Types with empty weight are skipped, not treated as zero loss.
    """
    zero = logits.sum() * 0.0
    pw = occupancy_bce_cell_weight(
        y_hot,
        type_masks,
        default_pos_weight=hot_pos_weight,
        pos_weight_by_type=pos_weight_by_type,
        fp_weight_by_type=fp_weight_by_type,
    )
    bce = F.binary_cross_entropy_with_logits(logits, y_hot, reduction="none")
    bce_parts: list[torch.Tensor] = []
    dice_parts: list[torch.Tensor] = []
    iou_parts: list[torch.Tensor] = []
    for name in OCC_TYPE_NAMES:
        node = type_masks.get(name)
        if node is None:
            continue
        node = node.to(device=hot_m.device, dtype=hot_m.dtype).reshape(-1)
        if node.numel() != hot_m.shape[-1] or float(node.sum()) <= 0:
            continue
        tm = hot_m * node.view(1, 1, -1)
        if float(tm.sum()) < 1e-6:
            continue
        denom = tm.sum().clamp_min(1.0)
        bce_parts.append((bce * pw * tm).sum() / denom)
        if w_dice > 0:
            dice_parts.append(soft_dice_loss(logits, y_hot, tm))
        if w_iou > 0:
            iou_parts.append(soft_iou_loss(logits, y_hot, tm))
    if not bce_parts:
        denom = hot_m.sum().clamp_min(1.0)
        loss_hot = (bce * pw * hot_m).sum() / denom
        loss_dice = soft_dice_loss(logits, y_hot, hot_m) if w_dice > 0 else zero
        loss_iou = soft_iou_loss(logits, y_hot, hot_m) if w_iou > 0 else zero
        return loss_hot, loss_dice, loss_iou
    loss_hot = torch.stack(bce_parts).mean()
    loss_dice = torch.stack(dice_parts).mean() if dice_parts else zero
    loss_iou = torch.stack(iou_parts).mean() if iou_parts else zero
    return loss_hot, loss_dice, loss_iou


def occupied_type_id(
    y_hot: torch.Tensor,
    type_masks: dict[str, torch.Tensor],
    occ_node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dominant occupied type in the horizon: 1=machine, 2=gantry, 3=AGV, 0=none."""
    y = y_hot.float()
    if occ_node_mask is not None:
        node = occ_node_mask.float()
        if node.dim() == 1:
            node = node.unsqueeze(0)
        if node.dim() == 2:
            y = y * node.unsqueeze(1)
    batch = y.shape[0]
    counts: list[torch.Tensor] = []
    for name in OCC_TYPE_NAMES:
        mask = type_masks.get(name)
        if mask is None:
            counts.append(y.new_zeros(batch))
            continue
        mask = mask.to(device=y.device, dtype=y.dtype).reshape(-1)
        if mask.numel() != y.shape[-1]:
            counts.append(y.new_zeros(batch))
            continue
        counts.append((y * mask.view(1, 1, -1)).reshape(batch, -1).sum(dim=-1))
    stacked = torch.stack(counts, dim=1)
    max_c, arg = stacked.max(dim=1)
    return torch.where(max_c > 0.5, arg + 1, torch.zeros_like(arg))


def contrastive_class_ids(
    y_block: torch.Tensor,
    dim_id: torch.Tensor,
    type_id: torch.Tensor,
) -> torch.Tensor:
    """Integer class (block, disturbance dim, resource type). Unknown dim → 5."""
    dim_id = dim_id.reshape(-1).to(dtype=torch.long)
    dim_id = torch.where(dim_id < 0, torch.full_like(dim_id, 5), dim_id.clamp(min=0, max=5))
    type_id = type_id.reshape(-1).to(dtype=torch.long).clamp(min=0, max=3)
    y_block = y_block.reshape(-1).to(dtype=torch.long)
    return y_block * 32 + dim_id * 4 + type_id


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1.0,
) -> torch.Tensor:
    """1 − Dice on sigmoid(logits); empty mask → 0 (eps keeps the ratio defined)."""
    pred = torch.sigmoid(logits)
    w = weight.float()
    inter = (pred * target * w).sum()
    denom = (pred * w).sum() + (target * w).sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """SupCon: same integer label is a positive. Skip if batch < 2 or no pairs."""
    if z.dim() != 2 or z.shape[0] < 2:
        return z.new_zeros(())
    z = F.normalize(z, dim=-1)
    labels = labels.reshape(-1)
    b = z.shape[0]
    sim = (z @ z.T) / max(float(temperature), 1e-6)
    self_mask = torch.eye(b, dtype=torch.bool, device=z.device)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    if not pos.any():
        return z.new_zeros(())
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(sim) * (~self_mask).float()
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True).clamp_min(1e-8))
    pos_n = pos.float().sum(dim=1).clamp_min(1.0)
    loss_i = -(log_prob * pos.float()).sum(dim=1) / pos_n
    return loss_i[pos.any(dim=1)].mean()


def soft_iou_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1.0,
) -> torch.Tensor:
    """1 − soft IoU on sigmoid(logits); empty mask → 0."""
    pred = torch.sigmoid(logits)
    w = weight.float()
    inter = (pred * target * w).sum()
    union = (pred * w).sum() + (target * w).sum() - inter
    return 1.0 - (inter + eps) / (union + eps)


def _horizon_block_flag(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """1 if any supervised node is occupied in the forecast horizon."""
    y_hot = batch.get("y_hot")
    if y_hot is None:
        wh = batch.get("window_hot")
        if wh is None:
            x = batch["X"]
            return torch.zeros(x.shape[0], device=x.device)
        return (wh.reshape(-1) > 0.5).to(dtype=torch.long)
    occ = batch.get("occ_node_mask")
    if occ is not None:
        if occ.dim() == 1:
            occ = occ.unsqueeze(0)
        if occ.dim() == 2:
            y_hot = y_hot * occ.unsqueeze(1)
    return (y_hot.reshape(y_hot.shape[0], -1).sum(dim=-1) > 0.5).to(dtype=torch.long)


class BNPDFormer(nn.Module):
    def __init__(self, config: dict[str, Any], data_feature: dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_feature = data_feature

        self.num_nodes = int(data_feature["num_nodes"])
        self.feature_dim = int(data_feature["feature_dim"])
        self.output_dim = int(config.get("output_dim", 1))
        self.input_window = int(config.get("input_window", 30))
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
        self.remain_to_jobs_done = bool(config.get("remain_to_jobs_done", False))
        self.max_remain_windows = int(config.get("max_remain_windows", 15))
        self.w_hot = float(config.get("w_hot", 1.0))
        self.w_dice = float(config.get("w_dice", 1.0))
        self.w_iou = float(config.get("w_iou", 1.0))
        self.w_remain_len = float(config.get("w_remain_len", 0.5))
        self.w_contrast = float(config.get("w_contrast", 0.1))
        self.contrast_temp = float(config.get("contrast_temp", 0.2))
        self.type_balanced_occupancy = bool(config.get("type_balanced_occupancy", True))
        self.use_grouped_embed = bool(config.get("use_grouped_embed", True))
        self.hot_pos_weight = float(config.get("hot_pos_weight", 8.0))
        self.hot_pos_weight_by_type = {
            str(k): float(v)
            for k, v in (config.get("hot_pos_weight_by_type") or {}).items()
            if str(k) in OCC_TYPE_NAMES
        }
        self.hot_fp_weight_by_type = {
            str(k): float(v)
            for k, v in (config.get("hot_fp_weight_by_type") or {}).items()
            if str(k) in OCC_TYPE_NAMES
        }
        self.near_remain_windows = int(config.get("near_remain_windows", 15))
        self.remain_loss_tau = float(config.get("remain_loss_tau", 40.0))

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
        for name, mask in occupancy_type_node_masks(
            [str(t) for t in data_feature.get("resource_types") or []],
            self.num_nodes,
        ).items():
            self.register_buffer(f"occ_type_{name}", mask)

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
            use_grouped_embed=self.use_grouped_embed,
        )
        self.contrast_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
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
        hidden = int(config.get("event_hidden", 64))
        if self.remain_to_jobs_done:
            self.remain_fuse = nn.Linear(self.embed_dim + self.skip_dim, self.embed_dim)
            self.jobs_mlp = nn.Sequential(nn.Linear(2, self.embed_dim), nn.GELU())
            self.remain_len_head = nn.Sequential(
                nn.Linear(hidden + self.embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
                nn.Softplus(),
            )
            self.remain_score_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, 1),
            )
            self.remain_hot_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, 1),
            )
            self.hot_type_scale = nn.Parameter(torch.ones(len(OCC_TYPE_NAMES)))
            bias_init = config.get("hot_type_bias_init") or [0.0, 0.0, 0.0]
            if not isinstance(bias_init, (list, tuple)) or len(bias_init) != len(OCC_TYPE_NAMES):
                bias_init = [0.0, 0.0, 0.0]
            self.hot_type_bias = nn.Parameter(
                torch.tensor([float(x) for x in bias_init], dtype=torch.float32)
            )
            self.end_conv1 = None
            self.end_conv2 = None
        else:
            self.end_conv1 = nn.Conv2d(self.input_window, self.output_window, kernel_size=1)
            self.end_conv2 = nn.Conv2d(self.skip_dim, self.output_dim, kernel_size=1)

        # ---- Auxiliary window heads (sparse-data friendly) ----
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
        self.intensity = PeriodicGatedIntensity(
            self.embed_dim,
            hidden=hidden,
            gate_floor=float(config.get("intensity_gate_floor", 0.1)),
        )
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
        """Returns skip (B, skip_dim, N, T), enc (B,T,N,D), h_last (B,N,D)."""
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
        return skip, enc, h_last

    @staticmethod
    def _sin_time_pe(k_max: int, dim: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(k_max, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(dim, 1))
        )
        pe = torch.zeros(k_max, dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        n_cos = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(pos * div[:n_cos])
        return pe

    def _remain_decode(
        self,
        skip: torch.Tensor,
        h_last: torch.Tensor,
        jobs_remaining: torch.Tensor,
        jobs_total: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Occupancy until remaining jobs done: scores (B,K,N,1), hot logits (B,K,N), R (B,)."""
        skip_pool = skip.mean(dim=-1).permute(0, 2, 1)
        h = self.remain_fuse(torch.cat([h_last, skip_pool], dim=-1))
        jobs = jobs_remaining.float().clamp_min(0.0)
        total = jobs_total.float().clamp_min(1.0)
        cond = torch.stack([jobs / total, total.clamp_min(1.0) / 32.0], dim=-1)
        cond_e = self.jobs_mlp(cond)
        h = h + cond_e.unsqueeze(1)
        pooled = self.aux_pool(h).mean(dim=1)
        remain_len = self.remain_len_head(torch.cat([pooled, cond_e], dim=-1)).squeeze(-1)
        k_max = self.max_remain_windows
        pe = self._sin_time_pe(k_max, self.embed_dim, h.device)
        h_k = h.unsqueeze(1) + pe.unsqueeze(0).unsqueeze(2)
        score_pred = self.remain_score_mlp(h_k)
        hot_logit = self.remain_hot_mlp(h_k).squeeze(-1)
        hot_logit = apply_hot_type_affine(
            hot_logit,
            self._occ_type_masks(),
            self.hot_type_scale,
            self.hot_type_bias,
        )
        return score_pred, hot_logit, remain_len

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        skip, enc, h_last = self.encode(batch["X"])
        if self.remain_to_jobs_done:
            jobs_r = batch.get("jobs_remaining")
            jobs_t = batch.get("jobs_total")
            if jobs_r is None:
                jobs_r = torch.ones(h_last.shape[0], device=h_last.device)
            if jobs_t is None:
                jobs_t = torch.ones(h_last.shape[0], device=h_last.device)
            score_pred, hot_logit, remain_len_pred = self._remain_decode(
                skip, h_last, jobs_r, jobs_t
            )
        else:
            skip_t = F.relu(skip.permute(0, 3, 2, 1))
            skip_t = self.end_conv1(skip_t)
            skip_t = self.end_conv2(F.relu(skip_t.permute(0, 3, 2, 1)))
            score_pred = skip_t.permute(0, 3, 2, 1)
            hot_logit = None
            remain_len_pred = None

        # auxiliary graph-level will / mark from last hidden
        node_energy = self.node_score_gate(h_last).squeeze(-1)  # (B, N)
        pooled = self.aux_pool(h_last).mean(dim=1)
        max_e = node_energy.max(dim=1).values.unsqueeze(-1)
        feat = torch.cat([pooled, max_e], dim=-1)
        will_logit = self.will_head(feat).squeeze(-1)
        tts_aux = self.tts_aux(feat).squeeze(-1)
        cause_logits = self.cause_head(feat)

        z = F.normalize(self.contrast_proj(h_last.mean(dim=1)), dim=-1)
        out: dict[str, torch.Tensor] = {
            "score_pred": score_pred,
            "will_logit": will_logit,
            "mark_logits": node_energy,
            "cause_logits": cause_logits,
            "tts_aux": tts_aux,
            "h_last": h_last,
            "z": z,
        }
        if hot_logit is not None:
            out["hot_logit"] = hot_logit
        if remain_len_pred is not None:
            out["remain_len_pred"] = remain_len_pred

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

    def _remain_step_weight(self, remain_mask: torch.Tensor | None) -> torch.Tensor | None:
        """Down-weight far remaining windows so occupancy loss is not all-zero.

        Supervise at most ``near_remain_windows`` (A.1 H, default 15 min) with
        exponential decay ``exp(-k / tau)``. ``remain_len`` is still full-horizon.
        """
        if remain_mask is None:
            return None
        k_max = remain_mask.shape[-1]
        steps = torch.arange(k_max, device=remain_mask.device, dtype=torch.float32)
        near = max(int(self.near_remain_windows), 1)
        tau = max(float(self.remain_loss_tau), 1.0)
        decay = torch.exp(-steps / tau) * (steps < float(near)).float()
        return remain_mask.float() * decay.unsqueeze(0)

    def _occ_type_masks(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in OCC_TYPE_NAMES:
            buf = getattr(self, f"occ_type_{name}", None)
            if buf is not None:
                out[name] = buf
        return out

    def calculate_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        out = self.forward(batch)
        y_score = batch["y_score"]
        will = batch["will"]
        mark = batch["mark"]

        remain_mask = batch.get("remain_mask")
        step_w = self._remain_step_weight(remain_mask) if remain_mask is not None else None
        if step_w is not None and step_w.dim() == 2:
            m = step_w.unsqueeze(-1).unsqueeze(-1)
            denom = m.sum().clamp_min(1.0)
            loss_score = (F.smooth_l1_loss(out["score_pred"], y_score, reduction="none") * m).sum() / denom
        else:
            loss_score = F.smooth_l1_loss(out["score_pred"], y_score)

        zero = out["score_pred"].sum() * 0.0
        loss_hot = zero
        loss_dice = zero
        loss_iou = zero
        loss_remain_len = zero
        if "hot_logit" in out and "y_hot" in batch and step_w is not None:
            logits = out["hot_logit"]
            y_hot = batch["y_hot"]
            hot_m = occupancy_cell_weight(step_w, batch.get("occ_node_mask"))
            type_masks = self._occ_type_masks()
            if self.type_balanced_occupancy and any(
                float(m.sum()) > 0 for m in type_masks.values()
            ):
                loss_hot, loss_dice, loss_iou = type_balanced_occupancy_losses(
                    logits,
                    y_hot,
                    hot_m,
                    type_masks,
                    hot_pos_weight=self.hot_pos_weight,
                    w_dice=self.w_dice,
                    w_iou=self.w_iou,
                    pos_weight_by_type=self.hot_pos_weight_by_type,
                    fp_weight_by_type=self.hot_fp_weight_by_type,
                )
            else:
                bce = F.binary_cross_entropy_with_logits(logits, y_hot, reduction="none")
                pw = occupancy_bce_cell_weight(
                    y_hot,
                    type_masks,
                    default_pos_weight=self.hot_pos_weight,
                    pos_weight_by_type=self.hot_pos_weight_by_type,
                    fp_weight_by_type=self.hot_fp_weight_by_type,
                )
                loss_hot = (bce * pw * hot_m).sum() / hot_m.sum().clamp_min(1.0)
                if self.w_dice > 0:
                    loss_dice = soft_dice_loss(logits, y_hot, hot_m)
                if self.w_iou > 0:
                    loss_iou = soft_iou_loss(logits, y_hot, hot_m)
        if "remain_len_pred" in out and "remain_len" in batch:
            loss_remain_len = F.smooth_l1_loss(
                torch.log1p(out["remain_len_pred"]),
                torch.log1p(batch["remain_len"].clamp_min(0.0)),
            )
        if self.w_will > 0:
            pos_weight = torch.tensor([self.pos_weight], device=will.device)
            loss_will = F.binary_cross_entropy_with_logits(
                out["will_logit"], will, pos_weight=pos_weight
            )
        else:
            loss_will = zero

        if self.w_mark > 0 or self.w_tts > 0:
            pos_mask = (will > 0.5) & (mark >= 0)
            if pos_mask.any():
                loss_mark = F.cross_entropy(out["mark_logits"][pos_mask], mark[pos_mask])
                tts_n = batch["tts"][pos_mask] / 60.0
                loss_tts = F.smooth_l1_loss(out["tts_aux"][pos_mask] / 60.0, tts_n)
            else:
                loss_mark = zero
                loss_tts = zero
        else:
            loss_mark = zero
            loss_tts = zero

        cause = batch.get("cause")
        if cause is not None and self.w_cause > 0:
            valid_cause = cause >= 0
            window_hot = batch.get("window_hot")
            if window_hot is not None:
                valid_cause = valid_cause & (window_hot.reshape(-1) > 0.5)
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
        nll_arr = loss_event
        nll_surv = loss_event
        if self.use_stgnpp and "h_event" in out:
            h_evt = out["h_event"]
            phase = batch["phase"]
            if phase.dim() == 2:
                phase_exp = phase.unsqueeze(1).expand(-1, self.num_nodes, -1)
            else:
                phase_exp = phase
            next_mask = batch["next_mask"] > 0.5
            if "surv_mask" in batch:
                surv_mask = batch["surv_mask"] > 0.5
            else:
                surv_mask = ~next_mask
            n_pos = int(next_mask.sum().item())
            n_surv = int(surv_mask.sum().item())
            if n_pos > 0:
                _loss_arr, stats_e = self.intensity.nll_and_duration(
                    h_evt[next_mask],
                    batch["next_tau"][next_mask].clamp_min(1e-3),
                    batch["next_dur"][next_mask],
                    torch.ones(n_pos, device=h_evt.device),
                    phase_exp[next_mask],
                    dur_weight=1.0,
                )
                nll_arr = stats_e["nll"]
                dur_v = stats_e["dur_mae"]
            if n_surv > 0:
                lam_h = self.intensity.cumulative(
                    h_evt[surv_mask],
                    batch["next_tau"][surv_mask].clamp_min(1e-3),
                    phase_exp[surv_mask],
                )
                nll_surv = lam_h.mean()
            # Equal-weight the two means so ~33 censored nodes do not drown
            # the handful of in-horizon arrivals.
            if n_pos > 0 and n_surv > 0:
                nll_v = 0.5 * nll_arr + 0.5 * nll_surv
            elif n_pos > 0:
                nll_v = nll_arr
            elif n_surv > 0:
                nll_v = nll_surv
            if n_pos + n_surv > 0:
                loss_event = nll_v + dur_v

        loss_contrast = zero
        if self.w_contrast > 0 and "z" in out:
            y_block = _horizon_block_flag(batch)
            dim_id = batch.get("run_dim_id")
            if dim_id is None:
                dim_id = torch.zeros(out["z"].shape[0], dtype=torch.long, device=out["z"].device)
            y_hot_c = batch.get("y_hot")
            if y_hot_c is None:
                y_hot_c = y_block.new_zeros(out["z"].shape[0], 1, 1)
            type_id = occupied_type_id(y_hot_c, self._occ_type_masks(), batch.get("occ_node_mask"))
            labels = contrastive_class_ids(y_block, dim_id, type_id)
            loss_contrast = supervised_contrastive_loss(
                out["z"], labels, temperature=self.contrast_temp
            )

        total = (
            self.w_score * loss_score
            + self.w_will * loss_will
            + self.w_mark * loss_mark
            + self.w_tts * loss_tts
            + self.w_event * loss_event
            + self.w_cause * loss_cause
            + self.w_hot * loss_hot
            + self.w_dice * loss_dice
            + self.w_iou * loss_iou
            + self.w_remain_len * loss_remain_len
            + self.w_contrast * loss_contrast
        )
        stats = {
            "loss": float(total.detach().cpu()),
            "loss_score": float(loss_score.detach().cpu()),
            "loss_will": float(loss_will.detach().cpu()),
            "loss_mark": float(loss_mark.detach().cpu()),
            "loss_tts": float(loss_tts.detach().cpu()),
            "loss_event": float(loss_event.detach().cpu()) if torch.is_tensor(loss_event) else float(loss_event),
            "loss_cause": float(loss_cause.detach().cpu()) if torch.is_tensor(loss_cause) else float(loss_cause),
            "loss_hot": float(loss_hot.detach().cpu()) if torch.is_tensor(loss_hot) else float(loss_hot),
            "loss_dice": float(loss_dice.detach().cpu()) if torch.is_tensor(loss_dice) else float(loss_dice),
            "loss_iou": float(loss_iou.detach().cpu()) if torch.is_tensor(loss_iou) else float(loss_iou),
            "loss_remain_len": float(loss_remain_len.detach().cpu()) if torch.is_tensor(loss_remain_len) else float(loss_remain_len),
            "loss_contrast": float(loss_contrast.detach().cpu()) if torch.is_tensor(loss_contrast) else float(loss_contrast),
            "nll": float(nll_v.detach().cpu()) if torch.is_tensor(nll_v) else float(nll_v),
            "nll_arrival": float(nll_arr.detach().cpu()) if torch.is_tensor(nll_arr) else float(nll_arr),
            "nll_surv": float(nll_surv.detach().cpu()) if torch.is_tensor(nll_surv) else float(nll_surv),
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
        if "hot_logit" in out:
            out["hot_prob"] = torch.sigmoid(out["hot_logit"])
        return out
