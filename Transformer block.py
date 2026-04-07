"""
experiments/transformer_block.py
=================================

Shows how to plug HydraFluxAttention into a standard Transformer block.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hydraflux import HydraFluxAttention, HydraFluxConfig


class HydraFluxTransformerBlock(nn.Module):
    """
    A single Transformer encoder block using HydraFlux Attention.

    Replace standard MHA with HFA for drop-in improved attention.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()

        config = HydraFluxConfig(
            d_model=d_model,
            num_heads=num_heads,
            resolution_schedule=[1, 1, 2, 2, 4, 4, 8, 8][:num_heads],
            spectral_bands=16,
            dropout=dropout,
        )

        self.attn = HydraFluxAttention(config)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN (more stable training)
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ff(self.norm2(x))
        return x


class HydraFluxEncoder(nn.Module):
    """Stack of HydraFlux Transformer blocks."""

    def __init__(self, vocab_size: int = 30000, d_model: int = 512,
                 num_layers: int = 6, num_heads: int = 8,
                 max_seq_len: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            HydraFluxTransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


if __name__ == "__main__":
    print("HydraFlux Transformer Encoder demo")
    enc = HydraFluxEncoder(vocab_size=10000, d_model=256, num_layers=4,
                           num_heads=4)
    tokens = torch.randint(0, 10000, (2, 64))
    out = enc(tokens)
    print(f"  Input  shape: {tokens.shape}")
    print(f"  Output shape: {out.shape}")
    params = sum(p.numel() for p in enc.parameters())
    print(f"  Total params: {params:,}")
