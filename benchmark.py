"""
tests/test_hydraflux.py
=======================

Unit tests for HydraFlux Attention components.

Run with:
    pytest tests/ -v
"""

import math
import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hydraflux import HydraFluxAttention, HydraFluxConfig
from hydraflux.heads import FluxHead, SpectralGate
from hydraflux.router import ContextRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    return HydraFluxConfig(
        d_model=64,
        num_heads=4,
        resolution_schedule=[1, 1, 2, 4],
        spectral_bands=8,
        router_hidden=32,
        dropout=0.0,
        use_rotary=True,
        flux_alpha=0.5,
        max_seq_len=128,
        causal=False,
    )


@pytest.fixture
def sample_input():
    torch.manual_seed(42)
    return torch.randn(2, 32, 64)  # [B=2, T=32, D=64]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestHydraFluxConfig:
    def test_valid_config(self, small_config):
        assert small_config.head_dim == 16

    def test_bad_num_heads(self):
        with pytest.raises(AssertionError):
            HydraFluxConfig(d_model=64, num_heads=5,
                            resolution_schedule=[1, 1, 2, 4, 8])

    def test_schedule_mismatch(self):
        with pytest.raises(AssertionError):
            HydraFluxConfig(d_model=64, num_heads=4,
                            resolution_schedule=[1, 2, 4])  # length 3 ≠ 4

    def test_flux_alpha_clamp(self):
        with pytest.raises(AssertionError):
            HydraFluxConfig(d_model=64, num_heads=4,
                            resolution_schedule=[1, 1, 2, 4],
                            flux_alpha=1.5)


# ---------------------------------------------------------------------------
# SpectralGate tests
# ---------------------------------------------------------------------------

class TestSpectralGate:
    def test_output_shape(self):
        gate = SpectralGate(num_bands=8, seq_len_hint=64)
        logits = torch.randn(2, 32, 40)  # [B, Tq, Tk]
        out = gate(logits)
        assert out.shape == logits.shape, "Shape mismatch"

    def test_gradient_flows(self):
        gate = SpectralGate(num_bands=8)
        logits = torch.randn(2, 16, 16, requires_grad=True)
        out = gate(logits)
        out.sum().backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# FluxHead tests
# ---------------------------------------------------------------------------

class TestFluxHead:
    def test_forward_shape(self, small_config, sample_input):
        head = FluxHead(small_config, stride=1)
        out = head(sample_input)
        assert out.shape == (2, 32, small_config.head_dim)

    def test_stride_2(self, small_config, sample_input):
        head = FluxHead(small_config, stride=2)
        out = head(sample_input)
        assert out.shape == (2, 32, small_config.head_dim)

    def test_stride_4(self, small_config, sample_input):
        head = FluxHead(small_config, stride=4)
        out = head(sample_input)
        assert out.shape == (2, 32, small_config.head_dim)

    def test_no_rotary(self, small_config, sample_input):
        cfg = HydraFluxConfig(
            d_model=64, num_heads=4,
            resolution_schedule=[1, 1, 2, 4],
            use_rotary=False, dropout=0.0
        )
        head = FluxHead(cfg, stride=1)
        out = head(sample_input)
        assert out.shape == (2, 32, cfg.head_dim)

    def test_gradient_flows(self, small_config, sample_input):
        head = FluxHead(small_config, stride=1)
        x = sample_input.clone().requires_grad_(True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# ContextRouter tests
# ---------------------------------------------------------------------------

class TestContextRouter:
    def test_output_shape(self, small_config, sample_input):
        router = ContextRouter(
            d_model=64, num_heads=4, hidden_size=32
        )
        head_outs = torch.randn(2, 32, 4, 16)   # [B, T, H, head_dim]
        out = router(sample_input, head_outs)
        assert out.shape == (2, 32, 16)

    def test_weights_sum_to_one(self, small_config, sample_input):
        router = ContextRouter(d_model=64, num_heads=4, hidden_size=32)
        # Access weights directly
        import torch.nn.functional as F
        logits = router.gate(sample_input)
        temp = torch.exp(router.log_temp)
        weights = F.softmax(logits / temp, dim=-1)
        sums = weights.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Full HydraFluxAttention tests
# ---------------------------------------------------------------------------

class TestHydraFluxAttention:
    def test_output_shape(self, small_config, sample_input):
        hfa = HydraFluxAttention(small_config)
        out = hfa(sample_input)
        assert out.shape == sample_input.shape

    def test_default_config(self):
        hfa = HydraFluxAttention()       # uses HydraFluxConfig defaults
        x = torch.randn(1, 16, 512)
        out = hfa(x)
        assert out.shape == x.shape

    def test_batch_independence(self, small_config):
        """Outputs for sample i must not depend on sample j."""
        hfa = HydraFluxAttention(small_config)
        hfa.eval()
        torch.manual_seed(0)
        x = torch.randn(4, 20, 64)

        with torch.no_grad():
            out_batch = hfa(x)
            out_single = hfa(x[2:3])

        assert torch.allclose(out_batch[2:3], out_single, atol=1e-5), (
            "Batch independence violated"
        )

    def test_gradient_flows(self, small_config, sample_input):
        hfa = HydraFluxAttention(small_config)
        x = sample_input.clone().requires_grad_(True)
        out = hfa(x)
        out.sum().backward()
        assert x.grad is not None

    def test_deterministic_eval(self, small_config, sample_input):
        hfa = HydraFluxAttention(small_config).eval()
        with torch.no_grad():
            out1 = hfa(sample_input)
            out2 = hfa(sample_input)
        assert torch.allclose(out1, out2)

    def test_mask_respected(self, small_config):
        """Masking all-but-first key should change the output."""
        hfa = HydraFluxAttention(small_config).eval()
        x = torch.randn(1, 16, 64)
        mask_none = None
        mask_all  = torch.ones(1, 16, 16, dtype=torch.bool)
        mask_all[:, :, 0] = False   # only allow attending to position 0

        with torch.no_grad():
            out_none = hfa(x, mask=mask_none)
            out_mask = hfa(x, mask=mask_all)

        assert not torch.allclose(out_none, out_mask), (
            "Mask had no effect on output"
        )

    def test_param_count_reasonable(self, small_config):
        hfa = HydraFluxAttention(small_config)
        vmha = nn.MultiheadAttention(64, 4, batch_first=True)
        hfa_p  = sum(p.numel() for p in hfa.parameters())
        vmha_p = sum(p.numel() for p in vmha.parameters())
        # HFA should be less than 10× standard MHA
        assert hfa_p < vmha_p * 10, (
            f"HFA params ({hfa_p}) > 10× VanillaMHA ({vmha_p})"
        )

    def test_odd_sequence_length(self, small_config):
        """Stride pooling must handle non-power-of-2 sequence lengths."""
        hfa = HydraFluxAttention(small_config).eval()
        for T in [7, 13, 33, 100]:
            x = torch.randn(1, T, 64)
            out = hfa(x)
            assert out.shape == (1, T, 64), f"Failed at T={T}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
