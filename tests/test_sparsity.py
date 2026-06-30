"""Unit tests for freqkv_ext.sparsity.

We verify:
1. The sparsity fraction is well-defined for trivial signals (constant -> 1
   coeff in DCT; impulse -> ~1 coeff in wavelet).
2. AR(1) smooth signals are sparser in DCT than in wavelets (well-known).
3. Spike-on-flat signals are sparser in wavelets than in DCT (the headline
   case for the wavelet KV claim).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from freqkv_ext.sparsity import (
    _sparsity_at,
    dct_sparsity_per_trace,
    dft_sparsity_per_trace,
    wavelet_band_energy,
    wavelet_sparsity_per_trace,
)


def test_wavelet_band_energy_normalizes():
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((16, 128)).astype(np.float32)
    approx, details = wavelet_band_energy(traces, wavelet="db4")
    assert 0.0 <= approx <= 1.0
    assert details.ndim == 1 and details.size >= 1
    assert abs(approx + float(details.sum()) - 1.0) < 1e-5  # bands partition energy


def test_wavelet_band_energy_smooth_is_approx_heavy():
    t = np.linspace(0, 1, 256)
    smooth = np.stack([np.sin(2 * np.pi * (1 + i % 2) * t) for i in range(16)]).astype(np.float32)
    approx, _ = wavelet_band_energy(smooth, wavelet="db4")
    assert approx > 0.5  # smooth signal => most energy in the coarse approx band


# -------- sanity --------


def test_sparsity_at_boundary_values():
    """An impulse should need 1 coefficient out of N to capture all energy."""
    impulse = np.zeros(64, dtype=np.float64)
    impulse[10] = 1.0
    assert _sparsity_at(impulse, 0.95) == pytest.approx(1 / 64, abs=1e-9)

    # All-zero signal: degenerate, return 1.0 by convention.
    assert _sparsity_at(np.zeros(8), 0.95) == 1.0

    # Uniform energy: must take all coeffs for any target > 0.
    flat = np.ones(16, dtype=np.float64)
    # ceil(0.95 * 16) = 16
    assert _sparsity_at(flat, 0.95) == pytest.approx(16 / 16)


def test_dct_constant_signal_is_one_coefficient():
    """A DC signal has all energy in DCT bin 0."""
    traces = np.ones((4, 128), dtype=np.float32)
    s = dct_sparsity_per_trace(traces, target=0.95)
    assert np.all(s == pytest.approx(1 / 128))


def test_dft_real_input_uses_rfft_universe():
    """DFT sparsity universe is N//2 + 1 (rfft bins). For a sinusoid at a
    rational frequency this should be one bin."""
    N = 256
    t = np.arange(N)
    traces = np.cos(2 * np.pi * 8 * t / N)[None, :].astype(np.float32)
    s = dft_sparsity_per_trace(traces, target=0.95)
    # rfft gives N//2 + 1 = 129 bins. 1 bin captures all energy.
    assert s[0] == pytest.approx(1 / (N // 2 + 1), abs=1e-3)


# -------- the headline comparisons --------


def _ar1(N: int, num: int, rho: float = 0.95, seed: int = 0) -> np.ndarray:
    """Generate ``num`` AR(1) traces of length N."""
    rng = np.random.default_rng(seed)
    out = np.zeros((num, N), dtype=np.float32)
    out[:, 0] = rng.standard_normal(num)
    for t in range(1, N):
        out[:, t] = rho * out[:, t - 1] + math.sqrt(1 - rho * rho) * rng.standard_normal(num)
    return out


def test_dct_beats_wavelet_on_ar1_smooth():
    """For pure AR(1) (smooth, no transients) DCT is the KLT limit; wavelets
    are typically less sparse. We expect median(DCT) <= median(wavelet)."""
    traces = _ar1(N=512, num=32, rho=0.95)
    dct_s = dct_sparsity_per_trace(traces, target=0.95)
    wav_s = wavelet_sparsity_per_trace(traces, target=0.95, wavelet="db4")
    # Both should be << 1 (smooth signal compresses well in both bases).
    assert np.median(dct_s) < 0.5
    assert np.median(wav_s) < 0.7
    # DCT should be at least as good as wavelet on AR(1).
    assert np.median(dct_s) <= np.median(wav_s) + 0.05


def test_wavelet_beats_dct_on_spike_on_flat():
    """A spike on a flat baseline (a 'needle') is the wavelet basis's home
    turf. Median wavelet sparsity should be << median DCT sparsity."""
    N = 512
    num = 32
    rng = np.random.default_rng(0)
    traces = np.zeros((num, N), dtype=np.float32)
    # Each trace: small Gaussian noise + one large spike at a random position.
    traces += 0.01 * rng.standard_normal((num, N)).astype(np.float32)
    for m in range(num):
        pos = rng.integers(N // 4, 3 * N // 4)
        traces[m, pos] = 1.0
    dct_s = dct_sparsity_per_trace(traces, target=0.95)
    wav_s = wavelet_sparsity_per_trace(traces, target=0.95, wavelet="db4")
    # For 1-spike-in-flat at 95% target, wavelet should need a tiny fraction
    # while DCT smears the spike across many bins.
    assert np.median(wav_s) < 0.1, f"wavelet={np.median(wav_s):.3f}"
    assert np.median(dct_s) > 0.5, f"dct={np.median(dct_s):.3f}"
    # Strict ordering with margin.
    assert np.median(wav_s) < np.median(dct_s) - 0.3
