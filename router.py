"""
ContextRouter
=============

A lightweight gating network that dynamically weights the contribution of
each FluxHead's output *per token*.  Unlike static mixture-of-experts (MoE)
or simply summing heads, the router produces a [B, T, num_heads] soft-weight
tensor conditioned on the input, then uses it to combine head outputs.

This is related to—but distinct from—standard MoE:
  - Standard MoE routes to DIFFERENT feed-forward experts (discrete or soft).
  - ContextRouter routes to ATTENTION HEADS that already differ in resolution
    and spectral characteristics; the router adds a learned per-token blending
    on top of that structural diversity.
  - It is non-parametrically cheaper than a separate gating network per layer
    in MoE because the routing signal is derived from a shared projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextRouter(nn.Module):
    """
    Per-token dynamic head weighting.

    Parameters
    ----------
    d_model : int
    num_heads : int
    hidden_size : int
        Hidden size of the two-layer routing MLP.
    dropout : float
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden_size, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_heads, bias=True),
        )

        # Temperature for sharpness control (learned)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        head_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : [B, T, D]  — original input (pre-projection)
        head_outputs : [B, T, num_heads, head_dim]

        Returns
        -------
        mixed : [B, T, D]  — weighted sum of head outputs
        """
        B, T, _ = x.shape

        # Gate weights: [B, T, num_heads]
        temp = torch.exp(self.log_temp).clamp(min=0.1, max=10.0)
        weights = F.softmax(self.gate(x) / temp, dim=-1)

        # Weighted sum: [B, T, num_heads, H] * [B, T, num_heads, 1] → [B, T, H]
        weights = weights.unsqueeze(-1)           # [B, T, num_heads, 1]
        mixed = (head_outputs * weights).sum(dim=2)  # [B, T, head_dim]

        return mixed
