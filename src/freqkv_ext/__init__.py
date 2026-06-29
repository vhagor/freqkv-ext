"""freqkv_ext: DSP-extended KV cache compressors.

This package provides drop-in replacements for FreqKV's `dct_compress` function:
    - DCT (baseline, faithful re-implementation of FreqKV).
    - DFT low-pass (DFT analog of FreqKV; sanity baseline).
    - DFT RoPE-aware bandpass (per hidden-dim pair, centered at theta_i).
    - Wavelet hard / adaptive thresholding.

All compressors implement the same signature:

    compress(x, target_len, seq_dim=2, kv_type="key", **kwargs) -> Tensor

with shape `[bsz, num_heads, seq_len, head_dim]` in and `[bsz, num_heads, target_len, head_dim]` out.

Use `freqkv_ext.patch.install(method=...)` to monkey-patch FreqKV's
`llama_attn_replace_dct_mempe.dct_compress` symbol with the chosen variant.
"""

from .transforms import (
    Compressor,
    dct_compress_baseline,
    dft_lowpass_compress,
    dft_rope_aware_compress,
    wavelet_adaptive_compress,
    get_compressor,
)

__all__ = [
    "Compressor",
    "dct_compress_baseline",
    "dft_lowpass_compress",
    "dft_rope_aware_compress",
    "wavelet_adaptive_compress",
    "get_compressor",
]

__version__ = "0.1.0"
