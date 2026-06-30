"""Rate-distortion codecs for the OFFLINE (training-free) E2 experiment.

These differ from ``freqkv_ext.transforms`` in one important way:

    * ``transforms`` codecs follow FreqKV's interface: compress a length-N
      sequence to a SHORTER length-L cache. Used in training / patching.

    * ``rdcodecs`` here are "keep-and-reconstruct-to-N": keep a budget of
      coefficients, zero the rest, and inverse-transform back to the SAME
      length N. This is the textbook transform-coding setup for measuring
      rate-distortion: the reconstruction error directly equals the energy
      thrown away by the basis at a given budget.

All codecs take a real tensor ``x`` of shape ``[B, H, N, D]`` and a budget
fraction ``gamma in (0, 1]``, and return a real reconstruction of the same
shape (float32). "Budget" is measured in retained real scalars per channel,
so the four codecs are directly comparable at equal ``gamma``.

The RST hybrid splits the budget between a RoPE-matched bandpass "bulk"
(fraction ``alpha``) and a time-domain sparse "residual" (fraction
``1 - alpha``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .rope_utils import (
    complex_to_real_pair,
    default_rope_thetas,
    real_pair_to_complex,
    thetas_to_bin_offsets,
)
from .transforms.dct_baseline import _dct, _idct

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _budget_len(gamma: float, n: int) -> int:
    """Number of coefficients to keep for budget fraction gamma (>=1)."""
    return max(1, min(n, int(round(gamma * n))))


def _topk_keep_lastdim(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the top-``k`` entries by absolute value along the last dim; zero rest."""
    n = x.shape[-1]
    if k >= n:
        return x
    if k <= 0:
        return torch.zeros_like(x)
    absx = x.abs()
    kth = torch.topk(absx, k, dim=-1).values[..., -1:]  # [..., 1]
    mask = absx >= kth
    return x * mask


# --------------------------------------------------------------------------
# codecs: [B, H, N, D] real -> [B, H, N, D] real (float32), keep gamma*N coeffs
# --------------------------------------------------------------------------


def dct_keep_reconstruct(
    x: torch.Tensor, gamma: float, select: str = "lowpass", **_unused
) -> torch.Tensor:
    """Sequence-axis DCT-II compression keeping ``gamma*N`` coefficients.

    ``select``:
        - ``"lowpass"`` (FreqKV): keep the lowest ``L`` frequency coefficients.
        - ``"topk"``: keep the ``L`` largest-magnitude coefficients (adaptive).
          Used to separate the *basis* effect from the *selection-rule* effect
          when comparing DCT against magnitude-thresholded wavelets.
    """
    x = x.to(torch.float32)
    B, H, N, D = x.shape
    L = _budget_len(gamma, N)
    x_seqlast = x.permute(0, 1, 3, 2)  # [B, H, D, N]
    X = _dct(x_seqlast, norm="ortho")
    if select == "topk":
        X = _topk_keep_lastdim(X, L)
    else:
        X[..., L:] = 0.0
    xr = _idct(X, norm="ortho")
    return xr.permute(0, 1, 3, 2).contiguous()


def dft_rope_keep_reconstruct(
    x: torch.Tensor,
    gamma: float,
    rope_base: float = 10000.0,
    rope_thetas: Optional[torch.Tensor] = None,
    is_key: bool = True,
    **_unused,
) -> torch.Tensor:
    """RoPE-matched bandpass: per pair keep ``gamma*N`` bins centered at ``n_i``.

    ``x`` for keys should be the POST-RoPE K. For values pass ``is_key=False``
    (band centered at 0 = plain low-pass).
    """
    x = x.to(torch.float32)
    B, H, N, D = x.shape
    L = _budget_len(gamma, N)
    c = real_pair_to_complex(x)  # [B, H, N, d_pair]
    C = torch.fft.fft(c.permute(0, 1, 3, 2), dim=-1)  # [B, H, d_pair, N]
    d_pair = D // 2

    if is_key:
        thetas = rope_thetas if rope_thetas is not None else default_rope_thetas(D, base=rope_base)
        centers = thetas_to_bin_offsets(thetas.to(x.device), N)  # [d_pair]
    else:
        centers = torch.zeros(d_pair, dtype=torch.long, device=x.device)

    half = L // 2
    offsets = torch.arange(L, device=x.device) - half  # [L]
    keep_idx = (centers.unsqueeze(1) + offsets.unsqueeze(0)) % N  # [d_pair, L]
    mask = torch.zeros(d_pair, N, dtype=torch.bool, device=x.device)
    mask.scatter_(1, keep_idx, True)
    C = C * mask.unsqueeze(0).unsqueeze(0)
    c_r = torch.fft.ifft(C, dim=-1).permute(0, 1, 3, 2)  # [B, H, N, d_pair]
    return complex_to_real_pair(c_r, out_dtype=torch.float32).contiguous()


def wavelet_keep_reconstruct(
    x: torch.Tensor,
    gamma: float,
    wavelet: str = "db4",
    level: Optional[int] = None,
    **_unused,
) -> torch.Tensor:
    """Wavelet hard-threshold: keep the top ``gamma*N`` coefficients per channel."""
    if pywt is None:  # pragma: no cover
        raise RuntimeError("PyWavelets required for wavelet_keep_reconstruct.")
    x = x.to(torch.float32)
    B, H, N, D = x.shape
    if level is None:
        level = max(1, min(pywt.dwt_max_level(N, wavelet), int(np.floor(np.log2(N)))))
    x_np = x.permute(0, 1, 3, 2).reshape(-1, N).cpu().numpy()  # [B*H*D, N]
    out = np.empty_like(x_np)
    for k in range(x_np.shape[0]):
        coeffs = pywt.wavedec(x_np[k], wavelet, level=level, mode="symmetric")
        flat, slices = pywt.coeffs_to_array(coeffs)
        n_keep = _budget_len(gamma, flat.size)
        if n_keep >= flat.size:
            rec = pywt.waverec(coeffs, wavelet, mode="symmetric")
            out[k] = rec[:N]
            continue
        thr = np.partition(np.abs(flat).ravel(), -n_keep)[-n_keep]
        flat_thr = np.where(np.abs(flat) >= thr, flat, 0.0)
        coeffs_thr = pywt.array_to_coeffs(flat_thr, slices, output_format="wavedec")
        rec = pywt.waverec(coeffs_thr, wavelet, mode="symmetric")
        out[k] = rec[:N]
    rec_t = torch.from_numpy(out).reshape(B, H, D, N).permute(0, 1, 3, 2)
    return rec_t.contiguous()


def rst_keep_reconstruct(
    x: torch.Tensor,
    gamma: float,
    alpha: float = 0.7,
    rope_base: float = 10000.0,
    rope_thetas: Optional[torch.Tensor] = None,
    is_key: bool = True,
    residual_domain: str = "time",
    wavelet: str = "db4",
    **_unused,
) -> torch.Tensor:
    """RST hybrid: bandpass bulk (alpha*gamma) + sparse residual ((1-alpha)*gamma).

    The bulk captures the deterministic RoPE frequency comb; the residual
    captures localized events the smooth bulk misses. ``residual_domain``:
        - "time": keep top-k largest residual samples along the sequence axis.
        - "wavelet": keep top-k residual wavelet coefficients per channel.
    """
    x = x.to(torch.float32)
    B, H, N, D = x.shape
    bulk_gamma = max(0.0, alpha) * gamma
    resid_gamma = max(0.0, 1.0 - alpha) * gamma

    if bulk_gamma > 0:
        bulk = dft_rope_keep_reconstruct(
            x, bulk_gamma, rope_base=rope_base, rope_thetas=rope_thetas, is_key=is_key
        )
    else:
        bulk = torch.zeros_like(x)

    resid = x - bulk
    if resid_gamma <= 0:
        return bulk

    if residual_domain == "wavelet":
        resid_kept = wavelet_keep_reconstruct(resid, resid_gamma, wavelet=wavelet)
    else:
        k = _budget_len(resid_gamma, N)
        resid_last = resid.permute(0, 1, 3, 2)  # [B, H, D, N]
        resid_kept = _topk_keep_lastdim(resid_last, k).permute(0, 1, 3, 2)
    return (bulk + resid_kept).contiguous()


RD_CODECS = {
    "dct": dct_keep_reconstruct,
    "dft_rope": dft_rope_keep_reconstruct,
    "wavelet": wavelet_keep_reconstruct,
    "rst": rst_keep_reconstruct,
}


# --------------------------------------------------------------------------
# error metrics
# --------------------------------------------------------------------------


def relative_frobenius_error(true: torch.Tensor, approx: torch.Tensor) -> float:
    """``||true - approx||_F / ||true||_F`` over the whole tensor (float32)."""
    true = true.to(torch.float32)
    approx = approx.to(torch.float32)
    num = torch.linalg.vector_norm(true - approx)
    den = torch.linalg.vector_norm(true).clamp_min(1e-12)
    return float((num / den).item())


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand [B, Hkv, N, D] to [B, Hkv*n_rep, N, D] for GQA."""
    if n_rep == 1:
        return x
    B, Hkv, N, D = x.shape
    return x[:, :, None, :, :].expand(B, Hkv, n_rep, N, D).reshape(B, Hkv * n_rep, N, D)


def causal_attention_output(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """Causal self-attention output, looping over (batch, head) to bound memory.

    Q: [B, Hq, N, D]; K, V: [B, Hkv, N, D] (GQA expanded internally).
    Returns [B, Hq, N, D] float32.
    """
    Q = Q.to(torch.float32)
    K = K.to(torch.float32)
    V = V.to(torch.float32)
    B, Hq, N, D = Q.shape
    Hkv = K.shape[1]
    if Hkv != Hq:
        K = _repeat_kv(K, Hq // Hkv)
        V = _repeat_kv(V, Hq // Hkv)
    scale = 1.0 / (D ** 0.5)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=Q.device), diagonal=1)
    out = torch.empty_like(Q)
    for b in range(B):
        for h in range(Hq):
            scores = (Q[b, h] @ K[b, h].transpose(-1, -2)) * scale  # [N, N]
            scores = scores.masked_fill(mask, float("-inf"))
            A = torch.softmax(scores, dim=-1)
            out[b, h] = A @ V[b, h]
    return out


# --------------------------------------------------------------------------
# E3: per-pair water-filling budget allocation
# --------------------------------------------------------------------------


def pair_energy_curves(post_rope_k: torch.Tensor, rope_base: float = 10000.0) -> np.ndarray:
    """Cumulative energy captured by an L-wide band centered at n_i, per pair.

    Returns ``[d_pair, N]`` array where ``curve[i, L-1]`` = fraction of pair-i
    energy in the ``L`` bins centered at ``n_i`` (so column 0 = single center
    bin, increasing L adds the next-nearest bins).
    """
    x = post_rope_k.to(torch.float32)
    B, H, N, D = x.shape
    d_pair = D // 2
    c = real_pair_to_complex(x).permute(0, 1, 3, 2)  # [B, H, d_pair, N]
    C = torch.fft.fft(c, dim=-1)
    power = (C.real ** 2 + C.imag ** 2).mean(dim=(0, 1))  # [d_pair, N]
    thetas = default_rope_thetas(D, base=rope_base)
    centers = thetas_to_bin_offsets(thetas, N)  # [d_pair]
    curves = np.zeros((d_pair, N), dtype=np.float64)
    p = power.cpu().numpy()
    for i in range(d_pair):
        ci = int(centers[i].item())
        # Order bins by distance from center (0, +1, -1, +2, -2, ...).
        order = [ci % N]
        for off in range(1, N):
            order.append((ci + off) % N)
            order.append((ci - off) % N)
            if len(order) >= N:
                break
        order = order[:N]
        e = p[i, order]
        total = e.sum() + 1e-12
        curves[i] = np.cumsum(e) / total
    return curves


def water_fill_allocation(curves: np.ndarray, total_budget_bins: int) -> np.ndarray:
    """Greedy marginal-gain allocation of bins across pairs.

    Args:
        curves: ``[d_pair, N]`` cumulative captured-energy curves (monotone).
        total_budget_bins: total bins to distribute across all pairs.

    Returns:
        ``[d_pair]`` integer allocation summing to ``total_budget_bins``.
    """
    d_pair, N = curves.shape
    alloc = np.zeros(d_pair, dtype=np.int64)
    # Marginal gain of giving pair i its (L+1)-th bin = curve[i,L] - curve[i,L-1].
    gains = curves[:, 0].copy()  # gain of first bin = curve[:,0]
    import heapq
    heap = [(-gains[i], i) for i in range(d_pair)]
    heapq.heapify(heap)
    for _ in range(min(total_budget_bins, d_pair * N)):
        neg_g, i = heapq.heappop(heap)
        alloc[i] += 1
        nxt = alloc[i]
        if nxt < N:
            marg = curves[i, nxt] - curves[i, nxt - 1]
            heapq.heappush(heap, (-marg, i))
    return alloc


def retained_energy(curves: np.ndarray, alloc: np.ndarray) -> float:
    """Mean fraction of energy retained across pairs for a given allocation."""
    d_pair = curves.shape[0]
    vals = []
    for i in range(d_pair):
        L = int(alloc[i])
        vals.append(curves[i, L - 1] if L > 0 else 0.0)
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# NE1 outlier diagnostics: is the wavelet win driven by localized outliers?
# --------------------------------------------------------------------------


def token_energy_profile(K: torch.Tensor) -> torch.Tensor:
    """Per-token energy ``mean_{B,H,D} K[...,t,:]^2`` -> ``[N]`` (float32)."""
    K = K.to(torch.float32)
    return (K ** 2).mean(dim=(0, 1, 3))


def top_energy_tokens(K: torch.Tensor, m: int) -> torch.Tensor:
    """Indices of the ``m`` highest-energy token positions, shape ``[m]`` (long)."""
    prof = token_energy_profile(K)
    m = max(1, min(m, prof.numel()))
    return torch.topk(prof, m).indices.sort().values


def energy_fraction_in_tokens(K: torch.Tensor, idx: torch.Tensor) -> float:
    """Fraction of total energy contained in token positions ``idx``."""
    prof = token_energy_profile(K)
    return float(prof[idx].sum() / prof.sum().clamp_min(1e-12))


def _excess_kurtosis(x: torch.Tensor, dim: int) -> torch.Tensor:
    mu = x.mean(dim=dim, keepdim=True)
    d = x - mu
    var = (d ** 2).mean(dim=dim)
    m4 = (d ** 4).mean(dim=dim)
    return m4 / var.clamp_min(1e-12) ** 2 - 3.0


def excess_kurtosis_along_seq(K: torch.Tensor) -> float:
    """Mean excess kurtosis of per-channel VALUE sequences (heavy tail => > 0)."""
    return float(_excess_kurtosis(K.to(torch.float32), dim=2).mean().item())


def first_difference_kurtosis(K: torch.Tensor) -> float:
    """Mean excess kurtosis of first differences ``K(t)-K(t-1)`` per channel.

    Stationary AR(1) (rho->1) has light-tailed (Gaussian) differences (~0). A
    piecewise-smooth / bounded-variation signal has sparse, spiky differences
    (large positive kurtosis) -> the signature that motivates a wavelet basis.
    """
    K = K.to(torch.float32)
    dK = K[:, :, 1:, :] - K[:, :, :-1, :]
    return float(_excess_kurtosis(dK, dim=2).mean().item())


def error_localization(true: torch.Tensor, approx: torch.Tensor,
                       idx: torch.Tensor) -> float:
    """Fraction of squared reconstruction error that falls on token rows ``idx``."""
    true = true.to(torch.float32)
    approx = approx.to(torch.float32)
    err2 = ((true - approx) ** 2)  # [B, H, N, D]
    err_per_token = err2.sum(dim=(0, 1, 3))  # [N]
    total = err_per_token.sum().clamp_min(1e-12)
    return float(err_per_token[idx].sum() / total)


def anchor_holdout_reconstruct(
    K: torch.Tensor,
    codec_fn,
    gamma: float,
    anchor_idx: torch.Tensor,
    **codec_kw,
) -> torch.Tensor:
    """Keep ``anchor_idx`` tokens exact; compress the rest at a budget-adjusted rate.

    The anchors cost ``m`` exact token-rows; we deduct that from the budget so the
    total retained coefficient count still equals ``gamma * N`` per channel.
    """
    K = K.to(torch.float32)
    B, H, N, D = K.shape
    m = int(anchor_idx.numel())
    keep_rest = max(1, int(round(gamma * N)) - m)
    gamma_eff = keep_rest / N
    mask = torch.ones(N, device=K.device)
    mask[anchor_idx] = 0.0
    K_rest = K * mask.view(1, 1, N, 1)
    rec = codec_fn(K_rest, gamma_eff, **codec_kw) * mask.view(1, 1, N, 1)
    rec[:, :, anchor_idx, :] = K[:, :, anchor_idx, :]
    return rec
