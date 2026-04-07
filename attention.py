"""
HydraFluxAttention — Main Module
=================================

Assembles FluxHeads + ContextRouter + output projection into a drop-in
replacement for nn.MultiheadAttention.

Architecture at a glance
-------------------------

Input x ──┬──────────────────────────────────────────────────────────┐
          │  Head-0  (stride=1,  SpectralGate, contra-flow)          │
          │  Head-1  (stride=1,  SpectralGate, contra-flow)          │
          │  Head-2  (stride=2,  SpectralGate, contra-flow)          │
          │  ...                                                       │
          │  Head-H  (stride=S,  SpectralGate, contra-flow)          │
          └──────────────────────── ContextRouter ────────────────────┘
                                         │
                                  output projection
                                         │
                                       out

"""

import torch
import torch.nn as nn
from typing import Optional

from .config import HydraFluxConfig
from .heads import FluxHead
from .router import ContextRouter


class HydraFluxAttention(nn.Module):
    """
    HydraFlux Attention (HFA).

    Drop-in attention module that combines:
      - Multi-resolution head-specific K/V pooling
      - Per-head spectral frequency gating on attention logits
      - Contra-flow (reverse-sequence) residual within each head
      - Learned per-token dynamic head routing via ContextRouter

    Parameters
    ----------
    config : HydraFluxConfig
        Full configuration.  A sensible default is used if None.

    Example
    -------
    >>> from hydraflux import HydraFluxAttention, HydraFluxConfig
    >>> cfg = HydraFluxConfig(d_model=256, num_heads=4,
    ...                       resolution_schedule=[1, 1, 2, 4])
    >>> hfa = HydraFluxAttention(cfg)
    >>> x = torch.randn(2, 64, 256)          # [B, T, D]
    >>> out = hfa(x)                          # [B, T, D]
    """

    def __init__(self, config: Optional[HydraFluxConfig] = None):
        super().__init__()
        if config is None:
            config = HydraFluxConfig()
        self.config = config

        # One FluxHead per entry in resolution_schedule
        self.heads = nn.ModuleList([
            FluxHead(config, stride=s)
            for s in config.resolution_schedule
        ])

        self.router = ContextRouter(
            d_model=config.d_model,
            num_heads=config.num_heads,
            hidden_size=config.router_hidden,
            dropout=config.dropout,
        )

        # Output projection: head_dim * num_heads == d_model
        self.out_proj = nn.Linear(config.head_dim, config.d_model, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor [B, T, D]
            Input sequence.
        mask : optional Tensor [B, T, T_k] bool
            True = position is masked (ignored).  For variable-length
            padding masks, expand to [B, 1, 1, T_k] before passing.

        Returns
        -------
        out  : Tensor [B, T, D]
        """
        B, T, D = x.shape

        # Run each head independently
        head_outs = []
        for head in self.heads:
            h_out = head(x, mask=mask)        # [B, T, head_dim]
            head_outs.append(h_out)

        # Stack → [B, T, num_heads, head_dim]
        stacked = torch.stack(head_outs, dim=2)

        # Context-aware per-token routing
        mixed = self.router(x, stacked)       # [B, T, head_dim]

        # Output projection
        out = self.out_proj(mixed)            # [B, T, D]
        return out

    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"d_model={c.d_model}, num_heads={c.num_heads}, "
            f"resolution_schedule={c.resolution_schedule}, "
            f"spectral_bands={c.spectral_bands}, "
            f"causal={c.causal}, params={self.num_parameters:,}"
        )
