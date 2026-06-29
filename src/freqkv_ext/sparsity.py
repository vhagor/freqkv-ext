"""Sparsity analysis: how many coefficients (DCT / DFT / wavelet) does each
real-valued trace of length ``N`` need to capture ``target`` (default 95%) of
its $L^2$ energy.

This is the **wavelet-vs-DCT GO/NO-GO experiment** that complements the
DFT-RoPE modulation check in :mod:`freqkv_ext.spectrum`:

- If the median DCT 95%-energy fraction $\geq$ median wavelet 95%-energy
  fraction across layers, the wavelet basis offers no structural advantage
  over FreqKV's DCT for compressing KV; wavelet path is unlikely to pay off.
- If wavelet's median is materially lower than DCT's (e.g. 0.1 vs 0.35), a
  principled sparse-coefficient wavelet cache can in principle compress the
  same K to a fraction of the budget without losing energy.

Functions take real-valued ``[M, N]`` numpy arrays (a sample of per-channel
sequence-axis traces) and return ``[M]`` per-trace sparsity fractions in
$[0, 1]$.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:  # pywt is an optional dep at unit-test time
    import pywt as _pywt
    _HAS_PYWT = True
except ImportError:  # pragma: no cover
    _pywt = None
    _HAS_PYWT = False


def _sparsity_at(coeffs: np.ndarray, target: float) -> float:
    """Min fraction of coefficients (sorted by squared magnitude) needed to
    accumulate ``target`` of the total energy.

    Args:
        coeffs: 1D coefficient array (real or magnitude-of-complex).
        target: energy fraction in (0, 1].

    Returns:
        Float in [0, 1].
    """
    if coeffs.size == 0:
        return 0.0
    e = coeffs.astype(np.float64) ** 2
    total = float(e.sum())
    if total == 0.0:
        return 1.0
    e_sorted = np.sort(e)[::-1]
    cum = np.cumsum(e_sorted) / total
    k = int(np.searchsorted(cum, target) + 1)
    return min(k / e.size, 1.0)


def dct_sparsity_per_trace(traces: np.ndarray, target: float = 0.95) -> np.ndarray:
    """DCT-II sparsity. Uses scipy if available, else a manual DCT via FFT."""
    try:
        from scipy.fftpack import dct as _scipy_dct
        def _dct(x: np.ndarray) -> np.ndarray:
            return _scipy_dct(x, type=2, norm="ortho")
    except ImportError:  # pragma: no cover
        def _dct(x: np.ndarray) -> np.ndarray:
            # Fall back: DCT-II via 2N FFT of mirrored signal.
            N = x.shape[-1]
            y = np.concatenate([x, x[::-1]], axis=-1)
            Y = np.fft.fft(y)[:N].real
            k = np.arange(N)
            w = np.exp(-1j * np.pi * k / (2 * N)).real
            out = Y * w
            out[0] *= 1.0 / np.sqrt(N)
            out[1:] *= np.sqrt(2.0 / N)
            return out

    out = np.empty(traces.shape[0], dtype=np.float32)
    for m in range(traces.shape[0]):
        out[m] = _sparsity_at(_dct(traces[m]), target)
    return out


def dft_sparsity_per_trace(traces: np.ndarray, target: float = 0.95) -> np.ndarray:
    """DFT sparsity, treating each complex rfft bin's magnitude as one coefficient.

    Real-input rfft has ``N // 2 + 1`` bins; we use them as the sparsity universe.
    """
    out = np.empty(traces.shape[0], dtype=np.float32)
    for m in range(traces.shape[0]):
        C = np.fft.rfft(traces[m])
        out[m] = _sparsity_at(np.abs(C), target)
    return out


def wavelet_sparsity_per_trace(
    traces: np.ndarray,
    target: float = 0.95,
    wavelet: str = "db4",
    mode: str = "symmetric",
    level: Optional[int] = None,
) -> np.ndarray:
    """DWT sparsity over the concatenated coefficient pyramid."""
    if not _HAS_PYWT:
        raise RuntimeError("PyWavelets is required for wavelet_sparsity_per_trace. "
                           "Install via `uv pip install pywavelets`.")
    out = np.empty(traces.shape[0], dtype=np.float32)
    N = traces.shape[1]
    if level is None:
        level = max(1, min(_pywt.dwt_max_level(N, wavelet), int(np.floor(np.log2(N)))))
    for m in range(traces.shape[0]):
        coeffs = _pywt.wavedec(traces[m], wavelet, level=level, mode=mode)
        flat = np.concatenate([np.asarray(c).ravel() for c in coeffs])
        out[m] = _sparsity_at(flat, target)
    return out


def sample_traces(
    keys: torch.Tensor,
    num_traces: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample ``num_traces`` 1D sequence-axis slices from a K tensor.

    Args:
        keys: ``[B, H, N, D]`` real tensor.
        num_traces: number of (batch, head, hidden) traces to retain.
        seed: RNG seed for the sampling.

    Returns:
        ``[K, N]`` float32 numpy array where ``K = min(num_traces, B*H*D)``.
    """
    B, H, N, D = keys.shape
    flat = keys.to(torch.float32).cpu().permute(0, 1, 3, 2).reshape(-1, N)
    flat_np = flat.numpy()
    M = flat_np.shape[0]
    k = min(num_traces, M)
    rng = np.random.default_rng(seed)
    idx = rng.choice(M, size=k, replace=False)
    return flat_np[idx].copy()
