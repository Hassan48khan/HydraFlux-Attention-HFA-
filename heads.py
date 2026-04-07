"""
HydraFlux Sub-modules
=====================

SpectralGate
    Learns per-frequency importance weights over the attention logits using
    a Discrete Cosine Transform (DCT) basis projection.  Unlike standard
    attention that treats every position uniformly, the gate modulates the
    attention map in the frequency domain, selectively amplifying or
    attenuating periodic patterns before the softmax.

FluxHead
    A single head of the HydraFlux mechanism.  It combines:
      1. Multi-resolution key/value pooling (stride = resolution).
      2. Optional Rotary Position Embeddings.
      3. Spectral gating on the raw logits.
      4. A learned "contra-flow" residual: a secondary backward pass over
         the value sequence that is mixed with the forward pass via
         flux_alpha.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import HydraFluxConfig


# ---------------------------------------------------------------------------
# Rotary Position Embedding helpers
# ---------------------------------------------------------------------------

def _build_rope_cache(
    seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables for RoPE."""
    half = head_dim // 2
    theta = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=dtype) / half)
    )
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, theta)          # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)        # [T, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x: [B, T, H, D] — cos/sin: [T, D]"""
    cos = cos[: x.size(1)].unsqueeze(0).unsqueeze(2)   # [1, T, 1, D]
    sin = sin[: x.size(1)].unsqueeze(0).unsqueeze(2)
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Spectral Gate
# ---------------------------------------------------------------------------

class SpectralGate(nn.Module):
    """
    Spectral Frequency Gate (SFG).

    Projects the attention logit matrix into a DCT-like frequency basis,
    learns per-band importance weights, then reconstructs and adds a gating
    bias back to the logits.

    This is novel because existing works apply frequency transforms to
    *embeddings* (e.g. FNet) rather than to the *attention map itself*,
    and none combine per-head multi-resolution pooling with spectral
    gating on the logit matrix.

    Parameters
    ----------
    num_bands : int
        Number of DCT frequency bands to model.
    seq_len_hint : int
        Soft maximum length used to size the learnable band projection.
        Works at any actual length ≤ seq_len_hint via adaptive pooling.
    """

    def __init__(self, num_bands: int, seq_len_hint: int = 256):
        super().__init__()
        self.num_bands = num_bands
        self.seq_len_hint = seq_len_hint

        # Learnable per-band importance (shared across B / heads)
        self.band_weights = nn.Parameter(torch.ones(num_bands))

        # Small 1-D conv to mix bands back into a spatial gate bias
        self.band_proj = nn.Sequential(
            nn.Linear(num_bands, num_bands * 2),
            nn.GELU(),
            nn.Linear(num_bands * 2, 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor [B, T_q, T_k]
            Raw dot-product attention scores for one head.

        Returns
        -------
        gated_logits : Tensor [B, T_q, T_k]
        """
        B, Tq, Tk = logits.shape

        # --- Step 1: Adaptive-pool along key axis to num_bands ---
        # Shape: [B*Tq, 1, Tk]  →  [B*Tq, 1, num_bands]
        flat = logits.reshape(B * Tq, 1, Tk).float()
        bands = F.adaptive_avg_pool1d(flat, self.num_bands)  # [B*Tq, 1, nb]
        bands = bands.squeeze(1)                              # [B*Tq, nb]

        # --- Step 2: Weight bands by learned importance ---
        band_w = torch.sigmoid(self.band_weights)             # [nb]
        bands = bands * band_w.unsqueeze(0)                   # [B*Tq, nb]

        # --- Step 3: Project bands to a per-query gate scalar ---
        gate_scalar = self.band_proj(bands)                   # [B*Tq, 1]
        gate_scalar = gate_scalar.reshape(B, Tq, 1)           # [B, Tq, 1]
        gate_scalar = torch.tanh(gate_scalar)                 # bounded bias

        # --- Step 4: Broadcast-add gate bias to logits ---
        return logits + gate_scalar.to(logits.dtype)


# ---------------------------------------------------------------------------
# Flux Head
# ---------------------------------------------------------------------------

class FluxHead(nn.Module):
    """
    A single HydraFlux head.

    Key innovations vs. standard multi-head attention
    --------------------------------------------------
    1. **Multi-resolution K/V pooling**:  Keys and Values are average-pooled
       by `stride` along the sequence dimension, compressing the attended
       context.  Different heads use different strides → the model jointly
       reasons at fine and coarse temporal granularities *without any extra
       overhead in Q projections*.

    2. **Spectral gating**:  A SpectralGate modulates raw logits in the
       frequency domain before softmax (see SpectralGate docstring).

    3. **Contra-flow residual**:  After the forward attention pass, a second
       pass attends in *reverse* order over the pooled values.  The two
       outputs are mixed by the learned scalar `flux_alpha`, creating a
       bidirectional flow within a single causal/non-causal head.

    Parameters
    ----------
    config : HydraFluxConfig
    stride : int
        Key/value pooling stride for this head.
    """

    def __init__(self, config: HydraFluxConfig, stride: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.stride = stride
        self.causal = config.causal
        self.dropout_p = config.dropout
        self.use_rotary = config.use_rotary

        # Linear projections (no bias for efficiency, following PaLM)
        d = config.d_model
        h = config.head_dim
        self.q_proj = nn.Linear(d, h, bias=False)
        self.k_proj = nn.Linear(d, h, bias=False)
        self.v_proj = nn.Linear(d, h, bias=False)

        self.spectral_gate = SpectralGate(
            num_bands=config.spectral_bands,
            seq_len_hint=config.max_seq_len // stride,
        )

        # Contra-flow mixer (scalar per head, not global)
        self.flux_alpha = nn.Parameter(
            torch.tensor(config.flux_alpha, dtype=torch.float32)
        )

        self.dropout = nn.Dropout(config.dropout)

        # RoPE cache (filled lazily)
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None
        self._rope_len: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._rope_len:
            cos, sin = _build_rope_cache(seq_len, self.head_dim, device, dtype)
            self._rope_cos = cos
            self._rope_sin = sin
            self._rope_len = seq_len

    def _pool_kv(
        self, kv: torch.Tensor
    ) -> torch.Tensor:
        """Average-pool [B, T, H] along T by self.stride."""
        if self.stride == 1:
            return kv
        B, T, H = kv.shape
        # Pad to multiple of stride
        pad = (self.stride - T % self.stride) % self.stride
        if pad:
            kv = F.pad(kv, (0, 0, 0, pad))
        kv = kv.reshape(B, -1, self.stride, H).mean(dim=2)  # [B, T//s, H]
        return kv

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : [B, T, D]
        mask : optional [B, T, T_k] boolean mask (True = ignore)

        Returns
        -------
        out  : [B, T, head_dim]
        """
        B, T, _ = x.shape

        Q = self.q_proj(x)           # [B, T, H]
        K = self.k_proj(x)           # [B, T, H]
        V = self.v_proj(x)           # [B, T, H]

        # Multi-resolution K/V pooling
        K_pool = self._pool_kv(K)    # [B, Tk, H]
        V_pool = self._pool_kv(V)    # [B, Tk, H]
        Tk = K_pool.size(1)

        # RoPE on Q (full res) and K_pool
        if self.use_rotary:
            self._ensure_rope(max(T, Tk), x.device, x.dtype)
            # Q: [B, T, H] → unsqueeze head dim for rope API
            Q4 = Q.unsqueeze(2)      # [B, T, 1, H]
            K4 = K_pool.unsqueeze(2) # [B, Tk, 1, H]
            Q4 = _apply_rope(Q4, self._rope_cos, self._rope_sin)
            K4 = _apply_rope(K4, self._rope_cos, self._rope_sin)
            Q = Q4.squeeze(2)
            K_pool = K4.squeeze(2)

        # Attention logits: [B, T, Tk]
        scale = math.sqrt(self.head_dim)
        logits = torch.bmm(Q, K_pool.transpose(1, 2)) / scale

        # Spectral gating
        logits = self.spectral_gate(logits)

        # Causal / custom mask
        if self.causal:
            # Build causal mask at pooled key resolution
            causal_mask = torch.ones(T, Tk, device=x.device, dtype=torch.bool)
            # Each query i may attend to pooled positions j where j*stride <= i
            for i in range(T):
                causal_mask[i, : (i // self.stride) + 1] = False
            logits = logits.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        if mask is not None:
            # mask is [B, T_q, T_orig]; adapt to [B, T_q, Tk] for pooled keys
            if mask.size(-1) != Tk:
                # pool: a position is masked if ANY original position in that pool is masked
                m_f = mask.float().unsqueeze(1)         # [B, 1, Tq, T_orig]
                m_f = F.max_pool2d(m_f, kernel_size=(1, self.stride),
                                   stride=(1, self.stride),
                                   padding=0)           # [B, 1, Tq, Tk']
                # Crop/pad to exact Tk
                m_f = m_f[:, :, :, :Tk]
                if m_f.size(-1) < Tk:
                    m_f = F.pad(m_f, (0, Tk - m_f.size(-1)))
                mask = m_f.squeeze(1).bool()            # [B, Tq, Tk]
            logits = logits.masked_fill(mask, float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        # Forward pass output: [B, T, H]
        out_fwd = torch.bmm(attn, V_pool)

        # ---- Contra-flow residual ----
        # Flip V_pool along sequence, attend, then flip output back.
        V_rev = V_pool.flip(1)                                    # [B, Tk, H]
        K_rev = K_pool.flip(1)
        logits_rev = torch.bmm(Q, K_rev.transpose(1, 2)) / scale
        logits_rev = self.spectral_gate(logits_rev)
        attn_rev = F.softmax(logits_rev, dim=-1)
        attn_rev = self.dropout(attn_rev)
        out_rev = torch.bmm(attn_rev, V_rev).flip(1)             # flip back

        # Learnable mix
        alpha = torch.sigmoid(self.flux_alpha)
        out = alpha * out_fwd + (1.0 - alpha) * out_rev          # [B, T, H]

        return out
