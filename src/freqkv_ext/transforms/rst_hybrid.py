"""RST hybrid compressor (FreqKV fixed-length interface).

RST = RoPE-Spectral Transform coding with sparse residual:

    x_hat = bandpass_bulk(x)  +  sparse_residual(x - bandpass_bulk(x))

The bulk captures the deterministic RoPE frequency comb with a per-pair
bandpass; the residual captures localized events (needles, code symbols) that
the smooth bulk misses.

This file provides the *interface-conformant* version: like the wavelet
compressor, it reconstructs to full length N then truncates to length L with
the FreqKV ``sqrt(L/N)`` amplitude convention, so it drops cleanly into
FreqKV's fixed-length cache for training (experiment E4).

The principled form keeps the sparse coefficient set in the cache directly
(no truncation); that is a larger cache-structure refactor and is validated
offline first via ``scripts/rate_distortion.py`` (experiment E2).
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def rst_compress(
    x: torch.Tensor,
    compress_len: int,
    seq_dim: int = 2,
    kv_type: str = "key",
    alpha: float = 0.7,
    rope_base: float = 10000.0,
    rope_thetas: Optional[torch.Tensor] = None,
    residual_domain: str = "time",
    wavelet: str = "db4",
    **_unused,
) -> torch.Tensor:
    """Hybrid bulk+residual compression, returning a length-``compress_len`` tensor.

    Args:
        x: ``[B, H, N, head_dim]``. For ``kv_type == "key"`` pass POST-RoPE K
            (the patch wrapper handles the rotation).
        compress_len: target sequence length L.
        alpha: fraction of the budget given to the spectral bulk (rest to
            the sparse residual).
    """
    assert seq_dim == 2, "Only seq_dim=2 supported."
    if compress_len == 0:
        return x[:, :, 0:0]
    if compress_len >= x.shape[seq_dim]:
        return x

    # Lazy import to avoid any package import cycle at module load time.
    from ..rdcodecs import rst_keep_reconstruct

    B, H, N, D = x.shape
    L = compress_len
    gamma = L / N
    x_hat = rst_keep_reconstruct(
        x,
        gamma=gamma,
        alpha=alpha,
        rope_base=rope_base,
        rope_thetas=rope_thetas,
        is_key=(kv_type == "key"),
        residual_domain=residual_domain,
        wavelet=wavelet,
    )  # [B, H, N, D] float32
    x_short = x_hat[:, :, :L, :] * math.sqrt(L / N)
    return x_short.to(x.dtype)
