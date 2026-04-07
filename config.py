"""
HydraFlux configuration.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HydraFluxConfig:
    """
    Configuration for the HydraFlux Attention mechanism.

    Attributes
    ----------
    d_model : int
        Total model dimension (must be divisible by num_heads).
    num_heads : int
        Number of flux heads. Each head operates at a different
        temporal/spatial resolution defined by `resolution_schedule`.
    resolution_schedule : List[int]
        Per-head stride for key/value pooling.
        E.g. [1, 2, 4, 8] → head-0 attends at full resolution,
        head-1 at half, etc. Length must equal num_heads.
    spectral_bands : int
        Number of frequency bands used by the SpectralGate.
    router_hidden : int
        Hidden size of the lightweight ContextRouter MLP.
    dropout : float
        Attention dropout probability.
    use_rotary : bool
        Apply Rotary Position Embeddings (RoPE) inside each head.
    flux_alpha : float
        Interpolation weight for the bidirectional residual flux.
        0 → pure standard residual; 1 → pure contra-flow residual.
    max_seq_len : int
        Maximum sequence length (used for RoPE cache pre-computation).
    causal : bool
        Whether to apply a causal (auto-regressive) mask.
    """

    d_model: int = 512
    num_heads: int = 8
    resolution_schedule: List[int] = field(
        default_factory=lambda: [1, 1, 2, 2, 4, 4, 8, 8]
    )
    spectral_bands: int = 16
    router_hidden: int = 128
    dropout: float = 0.1
    use_rotary: bool = True
    flux_alpha: float = 0.5
    max_seq_len: int = 2048
    causal: bool = False

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        assert len(self.resolution_schedule) == self.num_heads, (
            f"resolution_schedule length ({len(self.resolution_schedule)}) "
            f"must match num_heads ({self.num_heads})"
        )
        assert 0.0 <= self.flux_alpha <= 1.0, "flux_alpha must be in [0, 1]"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads
