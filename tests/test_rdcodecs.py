"""Tests for the offline rate-distortion codecs (E2 plumbing)."""

from __future__ import annotations

import numpy as np
import torch

from freqkv_ext.rdcodecs import (
    anchor_holdout_reconstruct,
    causal_attention_output,
    dct_keep_reconstruct,
    dft_rope_keep_reconstruct,
    energy_fraction_in_tokens,
    error_localization,
    excess_kurtosis_along_seq,
    pair_energy_curves,
    relative_frobenius_error,
    retained_energy,
    rst_keep_reconstruct,
    top_energy_tokens,
    token_energy_profile,
    water_fill_allocation,
    wavelet_keep_reconstruct,
)
from freqkv_ext.spectrum import apply_llama_rope_to_key

torch.manual_seed(0)


def _smooth_lowfreq(B, H, N, D):
    """Low-frequency (smooth) pre-RoPE-like signal."""
    t = torch.linspace(0, 1, N)
    base = torch.stack([torch.sin(2 * np.pi * (1 + j % 3) * t) for j in range(D)], dim=-1)
    return base[None, None].expand(B, H, N, D).contiguous()


def test_codecs_preserve_shape_and_float32():
    x = torch.randn(2, 4, 64, 16)
    for fn in (dct_keep_reconstruct, dft_rope_keep_reconstruct, wavelet_keep_reconstruct):
        out = fn(x, 0.5)
        assert out.shape == x.shape
        assert out.dtype == torch.float32
    out = rst_keep_reconstruct(x, 0.5, alpha=0.7)
    assert out.shape == x.shape and out.dtype == torch.float32


def test_full_budget_near_lossless():
    x = torch.randn(1, 2, 64, 16)
    for fn in (dct_keep_reconstruct, dft_rope_keep_reconstruct, wavelet_keep_reconstruct):
        out = fn(x, 1.0)
        assert relative_frobenius_error(x, out) < 1e-4, fn.__name__


def test_dft_rope_beats_dct_on_post_rope_key():
    # Smooth pre-RoPE K -> post-RoPE K is a high-frequency comb. The RoPE-matched
    # bandpass should reconstruct it far better than a low-pass DCT at low budget.
    x_pre = _smooth_lowfreq(1, 2, 128, 16)
    x_post = apply_llama_rope_to_key(x_pre, rope_base=10000.0)
    gamma = 0.1
    err_dft = relative_frobenius_error(x_post, dft_rope_keep_reconstruct(x_post, gamma, is_key=True))
    err_dct = relative_frobenius_error(x_post, dct_keep_reconstruct(x_post, gamma))
    assert err_dft < err_dct


def test_rst_alpha_one_equals_bulk():
    x = torch.randn(1, 2, 64, 16)
    a = dft_rope_keep_reconstruct(x, 0.5, is_key=True)
    b = rst_keep_reconstruct(x, 0.5, alpha=1.0, is_key=True)
    assert relative_frobenius_error(a, b) < 1e-5


def test_rst_residual_helps_on_spiky_signal():
    # Smooth bulk + a few large spikes: RST (bulk+residual) should beat pure
    # bandpass (alpha=1) at the same total budget.
    x_pre = _smooth_lowfreq(1, 2, 128, 16).clone()
    x_pre[0, 0, 30, 5] += 12.0
    x_pre[0, 1, 77, 9] -= 10.0
    x_post = apply_llama_rope_to_key(x_pre, rope_base=10000.0)
    gamma = 0.2
    err_bulk = relative_frobenius_error(x_post, rst_keep_reconstruct(x_post, gamma, alpha=1.0, is_key=True))
    err_rst = relative_frobenius_error(x_post, rst_keep_reconstruct(x_post, gamma, alpha=0.7, is_key=True))
    assert err_rst < err_bulk


def test_attention_identity_zero_error():
    q = torch.randn(1, 4, 32, 16)
    k = torch.randn(1, 4, 32, 16)
    v = torch.randn(1, 4, 32, 16)
    a1 = causal_attention_output(q, k, v)
    a2 = causal_attention_output(q, k, v)
    assert relative_frobenius_error(a1, a2) == 0.0


def test_attention_gqa_repeat_runs():
    q = torch.randn(1, 8, 16, 8)
    k = torch.randn(1, 2, 16, 8)  # GQA: 8 q heads, 2 kv heads
    v = torch.randn(1, 2, 16, 8)
    out = causal_attention_output(q, k, v)
    assert out.shape == (1, 8, 16, 8)


def test_causal_attention_matches_manual():
    q = torch.randn(1, 1, 5, 4)
    k = torch.randn(1, 1, 5, 4)
    v = torch.randn(1, 1, 5, 4)
    out = causal_attention_output(q, k, v)[0, 0]
    scale = 1.0 / 2.0
    scores = (q[0, 0] @ k[0, 0].T) * scale
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    ref = torch.softmax(scores, dim=-1) @ v[0, 0]
    assert torch.allclose(out, ref, atol=1e-5)


def test_water_filling_not_worse_than_uniform():
    d_pair, N = 4, 16
    # Steep pairs (fast energy capture) and shallow pairs.
    curves = np.zeros((d_pair, N))
    for i in range(d_pair):
        if i < 2:
            curves[i] = np.minimum(1.0, 0.9 + 0.1 * np.arange(N) / N)  # near-instant
        else:
            curves[i] = np.linspace(1.0 / N, 1.0, N)  # slow
    total = 2 * d_pair  # uniform = 2 bins each
    uni = np.full(d_pair, 2, dtype=np.int64)
    wf = water_fill_allocation(curves, total)
    assert wf.sum() == total
    assert retained_energy(curves, wf) >= retained_energy(curves, uni) - 1e-9


def test_pair_energy_curves_monotone():
    x = torch.randn(1, 2, 64, 8)
    curves = pair_energy_curves(x)
    assert curves.shape == (4, 64)
    assert np.all(np.diff(curves, axis=1) >= -1e-9)  # cumulative => non-decreasing
    assert np.allclose(curves[:, -1], 1.0, atol=1e-6)


def _key_with_sink(N=64, sink_pos=0, sink_mag=20.0):
    K = torch.randn(1, 2, N, 8) * 0.1
    K[:, :, sink_pos, :] += sink_mag
    return K


def test_token_energy_profile_finds_sink():
    K = _key_with_sink()
    prof = token_energy_profile(K)
    assert prof.shape == (64,)
    assert int(prof.argmax()) == 0
    assert int(top_energy_tokens(K, 1).item()) == 0


def test_energy_fraction_in_tokens_high_for_sink():
    K = _key_with_sink()
    idx = top_energy_tokens(K, 1)
    assert energy_fraction_in_tokens(K, idx) > 0.5  # one token dominates


def test_excess_kurtosis_heavy_tail_positive():
    K = _key_with_sink(sink_mag=30.0)
    assert excess_kurtosis_along_seq(K) > 1.0  # spike => leptokurtic
    smooth = torch.randn(1, 2, 256, 8)
    assert excess_kurtosis_along_seq(smooth) < 1.0  # gaussian ~ 0


def test_anchor_holdout_makes_anchor_rows_exact():
    K = _key_with_sink()
    idx = top_energy_tokens(K, 1)
    rec = anchor_holdout_reconstruct(K, dct_keep_reconstruct, 0.25, idx)
    assert torch.allclose(rec[:, :, idx, :], K[:, :, idx, :].float(), atol=1e-5)


def test_anchor_holdout_reduces_error_on_sinky_signal():
    K = _key_with_sink(sink_mag=20.0)
    g = 0.25
    e_plain = relative_frobenius_error(K, dct_keep_reconstruct(K, g))
    idx = top_energy_tokens(K, 1)
    e_anchor = relative_frobenius_error(K, anchor_holdout_reconstruct(K, dct_keep_reconstruct, g, idx))
    assert e_anchor < e_plain  # exact sink removes the smeared global error


def test_error_localization_in_unit_range():
    K = _key_with_sink()
    rec = dct_keep_reconstruct(K, 0.25)
    idx = top_energy_tokens(K, 4)
    frac = error_localization(K, rec, idx)
    assert 0.0 <= frac <= 1.0
