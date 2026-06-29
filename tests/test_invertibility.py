"""Invertibility & sanity tests, CPU-only."""

from __future__ import annotations

import math
import pytest
import torch

from freqkv_ext.transforms.dct_baseline import _dct, _idct


def test_dct_idct_roundtrip():
    """Composing DCT and IDCT on the same length must return the input."""
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    y = _idct(_dct(x))
    torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)


def test_dct_dc_amplitude():
    """For a constant input, DCT-II concentrates all energy in bin 0."""
    N = 32
    x = torch.ones(1, N)
    Y = _dct(x)
    # Other bins should be tiny.
    other_max = Y[0, 1:].abs().max().item()
    assert other_max < 1e-5
    assert Y[0, 0].abs() > 0


def test_dft_lowpass_invariant_on_low_freq():
    """A low-frequency signal should pass through DFT-low-pass nearly unchanged
    (up to amplitude rescale)."""
    from freqkv_ext.transforms.dft_lowpass import dft_lowpass_compress

    torch.manual_seed(0)
    B, H, N, D = 1, 1, 64, 8
    # Generate signal with energy only in low DFT bins.
    pair_idx = D // 2
    # Construct directly in spectrum then IDFT.
    spec = torch.zeros(B, H, pair_idx, N, dtype=torch.complex64)
    spec[..., :4] = torch.randn(B, H, pair_idx, 4, dtype=torch.complex64)
    sig_c = torch.fft.ifft(spec, dim=-1)  # complex
    sig_pair = torch.view_as_real(sig_c)  # [B, H, d_pair, N, 2]
    sig = sig_pair.permute(0, 1, 3, 2, 4).reshape(B, H, N, D).float()

    L = 32
    y = dft_lowpass_compress(sig, L, seq_dim=2, kv_type="key")
    # The retained content should still capture most of the original energy.
    # FreqKV's sqrt(L/N) rescale convention preserves *total* energy (not per-sample),
    # so we check sum |y|^2 ~= sum |x|^2 for a band-limited signal that fits in L.
    e_orig = (sig**2).sum()
    e_kept = (y**2).sum()
    ratio = (e_kept / e_orig).item()
    assert 0.7 < ratio < 1.3, (
        f"low-freq band-limited signal lost too much total energy: "
        f"orig={e_orig.item():.4f}, kept={e_kept.item():.4f}, ratio={ratio:.3f}"
    )
