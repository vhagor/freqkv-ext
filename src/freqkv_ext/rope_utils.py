"""RoPE-related utilities used by the DFT RoPE-aware compressor.

LLaMA-style RoPE pairs adjacent hidden dimensions ``(d_{2i}, d_{2i+1})`` and rotates the pair
by angle ``theta_i * t`` at sequence position ``t``. For the LLaMA family, the base is

    theta_i = base ** (-2 i / d), i = 0 .. d/2 - 1

with ``base=10000`` (LLaMA-2) or ``base=500000`` (LLaMA-3 long-context).

Key relations used downstream:

1. Pairing two real channels into a complex sequence makes RoPE a pointwise complex
   multiplication ``c_i(t) <- c_i(t) * exp(j theta_i t)``.

2. By the DFT modulation theorem, this is a circular shift by ``theta_i`` of the
   sequence-axis spectrum: ``C_i_post[w] = C_i_pre[w - theta_i]`` (continuous form).
   In a length-N DFT, ``theta_i`` corresponds to bin offset ``round(theta_i * N / (2*pi))``.

3. Therefore the "post-RoPE energy band" of the i-th pair is centered at bin
   ``n_i = (theta_i * N) / (2*pi)``, NOT at zero. A RoPE-matched bandpass keeps L bins
   around ``n_i`` per pair.
"""

from __future__ import annotations

import math

import torch


def default_rope_thetas(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """Standard LLaMA-family RoPE angles, shape ``[head_dim // 2]``."""
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    i = torch.arange(head_dim // 2, dtype=torch.float64)
    return (base ** (-2.0 * i / head_dim)).float()


def thetas_to_bin_offsets(thetas: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Convert RoPE angles to DFT bin offsets for a length-``seq_len`` transform.

    Returns an integer tensor of bin shifts ``n_i = round(theta_i * seq_len / (2*pi)) mod seq_len``
    of shape ``[head_dim // 2]``.
    """
    offsets = (thetas * seq_len / (2.0 * math.pi)).round().long()
    return offsets % seq_len


def real_pair_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Reshape ``[..., head_dim]`` to ``[..., head_dim // 2]`` complex tensor.

    Pairing convention: ``c_i = x[..., 2i] + 1j * x[..., 2i+1]`` (LLaMA's RoPE pairing).
    """
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0
    x_pair = x.reshape(*x.shape[:-1], head_dim // 2, 2)
    return torch.view_as_complex(x_pair.contiguous().to(torch.float32))


def complex_to_real_pair(c: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Inverse of `real_pair_to_complex`."""
    x_pair = torch.view_as_real(c)  # [..., head_dim // 2, 2]
    out = x_pair.reshape(*c.shape[:-1], c.shape[-1] * 2)
    return out.to(out_dtype)
