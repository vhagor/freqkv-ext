"""Unit tests for compression operators. CPU-only, runnable locally."""

from __future__ import annotations

import math
import pytest
import torch

from freqkv_ext.transforms import (
    dct_compress_baseline,
    dft_lowpass_compress,
    dft_rope_aware_compress,
    wavelet_adaptive_compress,
    get_compressor,
)
from freqkv_ext.rope_utils import (
    default_rope_thetas,
    thetas_to_bin_offsets,
    real_pair_to_complex,
    complex_to_real_pair,
)


# ---- Pair/complex roundtrip ----

def test_real_complex_roundtrip():
    x = torch.randn(2, 4, 16, 8)
    c = real_pair_to_complex(x)
    assert c.shape == (2, 4, 16, 4)
    assert c.is_complex()
    y = complex_to_real_pair(c, out_dtype=x.dtype)
    torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


def test_theta_bin_offsets_match_modulation():
    """Empirically check: applying RoPE to a unit-impulse-at-bin-0 pair-complex
    sequence shifts its DFT peak to bin theta*N/(2*pi)."""
    head_dim = 8
    N = 64
    thetas = default_rope_thetas(head_dim, base=10000.0)
    expected_bins = thetas_to_bin_offsets(thetas, N)
    # Build a baseband signal: constant 1 per pair -> DFT delta at bin 0.
    x = torch.ones(1, 1, N, head_dim)
    c = real_pair_to_complex(x)  # [1,1,N,d_pair]
    # Apply RoPE: multiply per pair by exp(j theta * t).
    t = torch.arange(N, dtype=torch.float32)
    phase = torch.exp(1j * t.unsqueeze(1) * thetas.unsqueeze(0))
    c_rot = c * phase.unsqueeze(0).unsqueeze(0)
    C = torch.fft.fft(c_rot, dim=2).abs()  # [1,1,N,d_pair]
    # For each pair, the peak should be at expected_bins.
    for pair_idx, want in enumerate(expected_bins.tolist()):
        peak = int(C[0, 0, :, pair_idx].argmax())
        # Allow off-by-one due to rounding when theta*N/(2*pi) is non-integer.
        assert abs(peak - want) <= 1, (
            f"pair {pair_idx}: peak {peak} != expected {want}"
        )


# ---- Shape / identity ----

@pytest.mark.parametrize("method", [
    dct_compress_baseline,
    dft_lowpass_compress,
    dft_rope_aware_compress,
])
def test_shape_invariants(method):
    B, H, N, D = 2, 4, 32, 16
    x = torch.randn(B, H, N, D)
    L = 16
    y = method(x, L, seq_dim=2, kv_type="key")
    assert y.shape == (B, H, L, D)
    assert y.dtype == x.dtype


@pytest.mark.parametrize("method", [
    dct_compress_baseline,
    dft_lowpass_compress,
    dft_rope_aware_compress,
])
def test_zero_compression_returns_empty(method):
    x = torch.randn(1, 1, 8, 8)
    y = method(x, 0, seq_dim=2, kv_type="key")
    assert y.shape == (1, 1, 0, 8)


@pytest.mark.parametrize("method", [
    dct_compress_baseline,
    dft_lowpass_compress,
    dft_rope_aware_compress,
])
def test_full_length_is_identity(method):
    x = torch.randn(1, 2, 16, 8)
    y = method(x, 16, seq_dim=2, kv_type="key")
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


# ---- Energy / amplitude rescaling ----

def test_registry_contains_all_methods():
    for name in ("dct", "dft_lowpass", "dft_rope", "wavelet"):
        fn = get_compressor(name)
        assert callable(fn)


def test_wavelet_shape_invariant():
    B, H, N, D = 1, 2, 32, 8
    x = torch.randn(B, H, N, D)
    L = 16
    y = wavelet_adaptive_compress(x, L, seq_dim=2, kv_type="key", wavelet="db4", level=3)
    assert y.shape == (B, H, L, D)
    assert y.dtype == x.dtype


def test_wavelet_value_path():
    # V branch should also work and produce sensible shape.
    B, H, N, D = 1, 2, 32, 8
    x = torch.randn(B, H, N, D)
    y = wavelet_adaptive_compress(x, 8, seq_dim=2, kv_type="value", wavelet="db4", level=3)
    assert y.shape == (B, H, 8, D)


@pytest.mark.parametrize("method", [
    dct_compress_baseline,
    dft_lowpass_compress,
])
def test_energy_roughly_preserved_smooth(method):
    """For a smooth (low-frequency) signal, low-pass compression preserves
    most of the energy. We don't check exact equality (rescaling is sqrt(L/N))
    but check that the compressed signal's per-sample energy is within
    a reasonable factor of the original."""
    torch.manual_seed(0)
    B, H, N, D = 1, 2, 256, 32
    # Smooth signal: cumulative sum of small noise (random walk = low-pass-ish).
    x = torch.cumsum(torch.randn(B, H, N, D) * 0.05, dim=2)
    L = 128
    y = method(x, L, seq_dim=2, kv_type="key")
    # FreqKV's rescale is sqrt(L/N), so per-sample amplitude scales by sqrt(L/N).
    # Per-sample energy of y vs x should be within ~order of magnitude.
    e_x = (x**2).sum() / x.numel()
    e_y = (y**2).sum() / y.numel()
    ratio = (e_y / e_x).item()
    # Loose bound; tightens later when we lock in the convention.
    assert 0.05 < ratio < 5.0, f"energy ratio out of range: {ratio}"
