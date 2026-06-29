"""Wavelet-based KV compressor with adaptive thresholding.

Motivation: DCT/DFT are global transforms; a single localized event (a "needle"
token, a code symbol, a number) spreads energy across many high-frequency
coefficients and is the first thing to be discarded under any low-pass scheme.

Discrete wavelet transform gives a time-frequency localized basis: the same
localized event maps to a small number of large coefficients at the appropriate
scale, and aggressive thresholding can keep it cheaply.

Implementation notes
--------------------
- We use PyWavelets for the actual DWT (CPU). For GPU paths we keep tensors on
  CPU during the transform and ship results back to the original device.
  GPU-native wavelet kernels (e.g. via pywavelets-cuda or hand-written conv) are
  a separate engineering effort; the current code prioritizes correctness.
- The compressor expects a target output length ``L < N`` and returns a length-L
  tensor in the time domain to keep the FreqKV interface invariant. Internally,
  we threshold to keep approximately ``L * head_dim`` largest-magnitude wavelet
  coefficients across (seq, hidden) per head; then inverse-transform and
  resample (truncate-or-pad) to length L.

This is a deliberate compromise: the principled wavelet form would change the
cache interface to "store sparse coefficient sets". That's a larger
refactor reserved for a later experiment. The current form lets us A/B against
DCT/DFT under identical FreqKV scaffolding.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

try:
    import pywt
except ImportError as e:  # pragma: no cover
    pywt = None
    _pywt_error: Optional[Exception] = e
else:
    _pywt_error = None


def _dwt_compress_1d(
    signal: np.ndarray, wavelet: str, level: int, keep_ratio: float
) -> np.ndarray:
    """Compress a 1D signal via wavelet hard-thresholding and reconstruct.

    Args:
        signal: 1D float numpy array.
        wavelet: pywt wavelet name (e.g. ``"db4"``, ``"sym8"``, ``"bior4.4"``).
        level: number of decomposition levels.
        keep_ratio: fraction of total coefficients to keep (sorted by magnitude).

    Returns:
        Reconstructed 1D array (same length as input).
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level, mode="symmetric")
    flat, slices = pywt.coeffs_to_array(coeffs)
    n_keep = max(1, int(flat.size * keep_ratio))
    if n_keep >= flat.size:
        rec = pywt.waverec(coeffs, wavelet=wavelet, mode="symmetric")
        return rec[: signal.shape[0]]
    threshold = np.partition(np.abs(flat).ravel(), -n_keep)[-n_keep]
    flat_thr = np.where(np.abs(flat) >= threshold, flat, 0.0)
    coeffs_thr = pywt.array_to_coeffs(flat_thr, slices, output_format="wavedec")
    rec = pywt.waverec(coeffs_thr, wavelet=wavelet, mode="symmetric")
    return rec[: signal.shape[0]]


def wavelet_adaptive_compress(
    x: torch.Tensor,
    compress_len: int,
    seq_dim: int = 2,
    kv_type: str = "key",
    wavelet: str = "db4",
    level: Optional[int] = None,
    keep_ratio: Optional[float] = None,
    **_unused,
) -> torch.Tensor:
    """Wavelet compression with hard thresholding then truncate-to-length-L.

    ``keep_ratio`` defaults to ``compress_len / q_len`` (= the FreqKV compression
    ratio gamma). ``level`` defaults to ``floor(log2(q_len))``.

    Returns a tensor of shape ``[B, H, compress_len, head_dim]``.
    """
    if pywt is None:  # pragma: no cover
        raise RuntimeError(
            "PyWavelets not installed. Install with `pip install pywavelets`."
        ) from _pywt_error

    assert seq_dim == 2, "Only seq_dim=2 supported."
    if compress_len == 0:
        return x[:, :, 0:0]
    if compress_len >= x.shape[seq_dim]:
        return x

    bsz, num_heads, q_len, head_dim = x.shape
    L = compress_len
    if keep_ratio is None:
        keep_ratio = L / q_len
    if level is None:
        level = max(1, int(math.floor(math.log2(q_len))))

    orig_device = x.device
    orig_dtype = x.dtype
    # Move to CPU float32 for pywt.
    x_np = x.detach().to("cpu", torch.float32).numpy()

    # Apply DWT compression along the sequence axis, independently for each
    # (batch, head, hidden) channel. Vectorize via reshape.
    x_reshaped = x_np.transpose(0, 1, 3, 2).reshape(-1, q_len)  # [B*H*d, N]
    out_reshaped = np.empty_like(x_reshaped)
    for k in range(x_reshaped.shape[0]):
        out_reshaped[k] = _dwt_compress_1d(
            x_reshaped[k], wavelet=wavelet, level=level, keep_ratio=keep_ratio
        )

    rec = out_reshaped.reshape(bsz, num_heads, head_dim, q_len).transpose(0, 1, 3, 2)
    # Resample to length L by truncating-then-rescaling (matches FreqKV's
    # sqrt(L/N) amplitude convention).
    rec_short = rec[:, :, :L, :] * math.sqrt(L / q_len)

    return torch.from_numpy(rec_short).to(device=orig_device, dtype=orig_dtype)
