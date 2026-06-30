"""Compression operators.

All operators consume and produce tensors of shape `[bsz, num_heads, seq_len, head_dim]`.
The intended drop-in slot is FreqKV's `dct_compress(x, compress_len, seq_dim=2, kv_type="key")`.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import torch

from .dct_baseline import dct_compress_baseline
from .dft_lowpass import dft_lowpass_compress
from .dft_rope_aware import dft_rope_aware_compress
from .rst_hybrid import rst_compress
from .wavelet import wavelet_adaptive_compress

Compressor = Callable[..., torch.Tensor]


METHODS = {
    "dct": dct_compress_baseline,
    "dft_lowpass": dft_lowpass_compress,
    "dft_rope": dft_rope_aware_compress,
    "wavelet": wavelet_adaptive_compress,
    "rst": rst_compress,
}


def get_compressor(name: str) -> Compressor:
    """Look up a compressor by name. Raises KeyError if unknown."""
    if name not in METHODS:
        raise KeyError(
            f"Unknown compressor {name!r}. Available: {sorted(METHODS)}"
        )
    return METHODS[name]


__all__ = [
    "Compressor",
    "METHODS",
    "get_compressor",
    "dct_compress_baseline",
    "dft_lowpass_compress",
    "dft_rope_aware_compress",
    "wavelet_adaptive_compress",
    "rst_compress",
]
