"""STGNPP building blocks adapted for factory bottleneck events.

Faithful to Jin et al., AAAI 2023 (STGNPP):
  Spatio-temporal Inquirer → Continuous GRU (flow + discrete)
  → periodic-gated cumulative intensity → NLL + duration head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalInquirer(nn.Module):
    """Gather encoder states at historical event time indices and aggregate.

    For each node, events falling inside the input window are indexed by
    ``event_idx ∈ [0, T)``; invalid slots use ``-1``.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim + 1, embed_dim)  # + duration

    def forward(
        self,
        enc: torch.Tensor,
        event_idx: torch.Tensor,
        event_dur: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            enc: (B, T, N, D)
            event_idx: (B, N, L) int, -1 = pad
            event_dur: (B, N, L) float, duration (seconds), pad 0
            event_mask: (B, N, L) float {0,1}
        Returns:
            H_e: (B, N, L, D) event embeddings (zeros where masked)
        """
        B, T, N, D = enc.shape
        L = event_idx.shape[-1]
        # clamp idx for gather safety
        idx = event_idx.clamp(min=0, max=T - 1)  # (B, N, L)
        # enc_n: (B, N, T, D)
        enc_n = enc.permute(0, 2, 1, 3)
        idx_exp = idx.unsqueeze(-1).expand(B, N, L, D)
        gathered = torch.gather(enc_n, 2, idx_exp)  # (B, N, L, D)
        dur = event_dur.unsqueeze(-1)
        h = self.proj(torch.cat([gathered, dur], dim=-1))
        return h * event_mask.unsqueeze(-1)


class GRUFlowCell(nn.Module):
    """Continuous residual flow between events (neural-flow style GRU)."""

    def __init__(self, dim: int):
        super().__init__()
        self.wz = nn.Linear(dim + 1, dim)
        self.wh = nn.Linear(dim + 1, dim)
        self.w_tau = nn.Linear(1, dim, bias=False)

    def forward(self, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        h: (..., D), tau: (...,) inter-event time (normalized)
        """
        tau = tau.unsqueeze(-1)
        inp = torch.cat([h, tau], dim=-1)
        z = torch.sigmoid(self.wz(inp))
        g = torch.tanh(self.wh(inp))
        # φ(t)=tanh(Wτ τ), |φ|<1, φ(0)=0
        phi = torch.tanh(self.w_tau(tau))
        return h + phi * (1.0 - z) * (g - h)


class ContinuousGRU(nn.Module):
    """Alternate GRU-flow (between events) and discrete GRU (at events)."""

    def __init__(self, dim: int, n_flow_layers: int = 2):
        super().__init__()
        self.flow_layers = nn.ModuleList([GRUFlowCell(dim) for _ in range(n_flow_layers)])
        self.disc_gru = nn.GRUCell(dim, dim)

    def forward(
        self,
        H_e: torch.Tensor,
        event_mask: torch.Tensor,
        inter_tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            H_e: (B, N, L, D) event embeddings
            event_mask: (B, N, L)
            inter_tau: (B, N, L) time since previous event (pad 0); index 0 = 0
        Returns:
            h_last: (B, N, D) hidden state after last valid event (0 if none)
        """
        B, N, L, D = H_e.shape
        h = torch.zeros(B, N, D, device=H_e.device, dtype=H_e.dtype)
        has_any = torch.zeros(B, N, device=H_e.device, dtype=H_e.dtype)

        for i in range(L):
            m = event_mask[:, :, i]  # (B, N)
            if m.sum() == 0:
                continue
            tau = inter_tau[:, :, i]
            # continuous evolution from previous state
            h_flow = h
            for cell in self.flow_layers:
                h_flow = cell(h_flow, tau)
            # instantaneous update at event
            x = H_e[:, :, i, :]
            h_new = self.disc_gru(
                x.reshape(B * N, D),
                h_flow.reshape(B * N, D),
            ).reshape(B, N, D)
            m_e = m.unsqueeze(-1)
            h = torch.where(m_e > 0, h_new, h)
            has_any = torch.maximum(has_any, m)

        return h * has_any.unsqueeze(-1)


class PeriodicGatedIntensity(nn.Module):
    """Cumulative intensity Λ(τ|h) with periodic gate (STGNPP eq.19).

    Λ is produced by MLP; intensity λ = ∂Λ/∂τ via autograd for NLL.
    Gate uses episode-phase proxies for (time-of-day, day-of-week) when
    calendar time is unavailable in simulation.

    ``gate_floor`` keeps λ away from the 1e-6 NLL clamp when the sigmoid
    gate would otherwise zero the cumulative intensity.
    """

    def __init__(self, dim: int, hidden: int = 64, gate_floor: float = 0.1):
        super().__init__()
        self.gate_floor = float(gate_floor)
        self.f_l = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )
        last_linear = self.f_l[-2]
        assert isinstance(last_linear, nn.Linear)
        nn.init.constant_(last_linear.bias, 1.0)
        self.f_p = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.dur_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def cumulative(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (..., D), tau: (...,) > 0, phase: (..., 2) in [0,1]
        returns Λ: (...,)
        """
        base = self.f_l(torch.cat([h, tau.unsqueeze(-1)], dim=-1)).squeeze(-1)
        raw_gate = torch.sigmoid(self.f_p(phase)).squeeze(-1)
        floor = self.gate_floor
        gate = floor + (1.0 - floor) * raw_gate
        return base * gate

    def duration(self, h: torch.Tensor) -> torch.Tensor:
        return self.dur_head(h).squeeze(-1)

    def nll_and_duration(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        dur_true: torch.Tensor,
        mask: torch.Tensor,
        phase: torch.Tensor,
        dur_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Negative log-likelihood of inter-event time + duration MAE.

        NLL = Λ(τ) - log(∂Λ/∂τ)   (Omi / STGNPP cumulative form)
        """
        tau_req = tau.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            Lam = self.cumulative(h, tau_req, phase)
            ones = torch.ones_like(Lam)
            (dLam,) = torch.autograd.grad(
                Lam, tau_req, grad_outputs=ones, create_graph=self.training, retain_graph=True
            )
        intensity = dLam.clamp_min(1e-6)
        nll = Lam - torch.log(intensity)
        nll = (nll * mask).sum() / mask.sum().clamp_min(1.0)

        dur_pred = self.duration(h)
        # duration in same units as tau normalization caller uses
        dur_loss = F.l1_loss(dur_pred * mask, dur_true * mask, reduction="sum")
        dur_loss = dur_loss / mask.sum().clamp_min(1.0)

        total = nll + dur_weight * dur_loss
        return total, {
            "nll": nll.detach(),
            "dur_mae": dur_loss.detach(),
            "dur_pred": dur_pred.detach(),
            "Lam": Lam.detach(),
            "intensity": intensity.detach(),
        }
