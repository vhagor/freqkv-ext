"""DFT RoPE-aware bandpass compressor.

Algorithmic claim (see ``docs/METHOD.md``):

By the DFT modulation theorem, applying RoPE in the time domain is equivalent to a
per-pair frequency shift by ``theta_i`` of the pair-complex sequence-axis DFT. Thus the
"post-RoPE energy band" of each hidden-dim pair is centered at bin
``n_i = round(theta_i * N / (2*pi))``, NOT at zero.

This compressor caches the **post-RoPE** spectrum by:

1. Embedding K via LLaMA's RoPE on-the-fly (so the input to this op should be the
   already RoPE-rotated K). For V (no RoPE) we fall back to low-pass.
2. DFT along the sequence axis on pair-complex tensors.
3. **Per-pair bandpass**: for pair ``i`` keep ``L`` bins centered at ``n_i``.
4. IDFT of length L (interpreting the retained bins as a length-L spectrum).
5. Amplitude rescale.

The hypothesis being tested: a RoPE-matched bandpass preserves attention-relevant
content better than a uniform low-pass at the same compression ratio, especially for
pairs with large ``theta_i`` (early hidden indices, which encode short-range positional
detail).

Usage with FreqKV's existing pipeline
-------------------------------------
FreqKV's `dct_compress` is called on **pre-RoPE** K and V. To make this op faithful to
its hypothesis, the caller should rotate K with RoPE before passing it in (when
``kv_type == "key"``). For V, the op silently degenerates to low-pass.

The `freqkv_ext.patch.install("dft_rope")` patch installs a wrapper that handles the
pre/post-RoPE rotation transparently. See ``patch.py``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..rope_utils import (
    complex_to_real_pair,
    default_rope_thetas,
    real_pair_to_complex,
    thetas_to_bin_offsets,
)


def _gather_band_around(C: torch.Tensor, centers: torch.Tensor, bandwidth: int) -> torch.Tensor:
    """For each pair channel, gather `bandwidth` bins of `C` centered at `centers[i]`.

    Args:
        C: complex tensor of shape ``[B, H, d_pair, N]``.
        centers: long tensor of shape ``[d_pair]`` with bin centers in ``[0, N)``.
        bandwidth: number of bins to keep per pair (L).

    Returns:
        Complex tensor of shape ``[B, H, d_pair, L]``.
    """
    B, H, d_pair, N = C.shape
    L = bandwidth
    half_lo = L // 2
    # Bin indices to gather: (centers - half_lo + [0, L)) mod N -> shape [d_pair, L].
    offsets = torch.arange(L, device=C.device) - half_lo  # [L]
    idx = (centers.unsqueeze(1) + offsets.unsqueeze(0)) % N  # [d_pair, L]
    # Broadcast and gather.
    idx_exp = idx.unsqueeze(0).unsqueeze(0).expand(B, H, d_pair, L)
    out = torch.gather(C, dim=-1, index=idx_exp)
    return out


def dft_rope_aware_compress(
    x: torch.Tensor,
    compress_len: int,
    seq_dim: int = 2,
    kv_type: str = "key",
    rope_base: float = 10000.0,
    rope_thetas: Optional[torch.Tensor] = None,
    **_unused,
) -> torch.Tensor:
    """RoPE-aware DFT bandpass compression.

    Args:
        x: ``[B, H, N, head_dim]``. For ``kv_type == "key"``, the caller should pass the
            post-RoPE K. For ``kv_type == "value"``, RoPE is irrelevant; this op acts
            as a plain DFT low-pass.
        compress_len: target sequence length L.
        rope_base: RoPE base if ``rope_thetas`` is not supplied. 10000 for LLaMA-2/3.
        rope_thetas: ``[head_dim // 2]`` tensor of per-pair angles. Overrides ``rope_base``.

    Returns:
        ``[B, H, L, head_dim]``.
    """
    assert seq_dim == 2, "Only seq_dim=2 supported."
    if compress_len == 0:
        return x[:, :, 0:0]
    if compress_len >= x.shape[seq_dim]:
        return x

    bsz, num_heads, q_len, head_dim = x.shape
    assert head_dim % 2 == 0
    d_pair = head_dim // 2
    L = compress_len
    N = q_len

    c = real_pair_to_complex(x)  # [B, H, N, d_pair] complex
    # DFT along sequence (dim=2). Move sequence to last for clean gather.
    C = torch.fft.fft(c.permute(0, 1, 3, 2), dim=-1)  # [B, H, d_pair, N]

    if kv_type == "value":
        # No RoPE -> standard low-pass (use the same gather machinery with centers=0).
        centers = torch.zeros(d_pair, dtype=torch.long, device=x.device)
    else:
        if rope_thetas is None:
            thetas = default_rope_thetas(head_dim, base=rope_base).to(x.device)
        else:
            thetas = rope_thetas.to(x.device)
        centers = thetas_to_bin_offsets(thetas, N)

    C_kept = _gather_band_around(C, centers, L)  # [B, H, d_pair, L]

    # IDFT of length L using the retained bins. To make the resulting time-domain
    # signal an unrotated baseband (so it composes cleanly with downstream RoPE
    # re-application at new positions), demodulate by multiplying with
    # exp(-j * 2*pi * (center / N) * t) before IDFT. This shifts the band back to
    # bin 0 prior to inverse transform.
    #
    # The phase factor depends on the absolute time index of the compressed samples,
    # which is conventionally taken as t = 0..L-1 (the "internal cache position"
    # used by FreqKV's iterative compression).
    if kv_type == "key":
        t = torch.arange(L, dtype=torch.float32, device=x.device)
        # centers is per-pair; phase is per-(pair, t).
        phase = torch.exp(
            -1j * 2.0 * math.pi * (centers.float().unsqueeze(1) / N) * t.unsqueeze(0)
        )  # [d_pair, L]
        c_compressed = torch.fft.ifft(C_kept, dim=-1)  # [B, H, d_pair, L]
        c_compressed = c_compressed * phase.unsqueeze(0).unsqueeze(0)
    else:
        c_compressed = torch.fft.ifft(C_kept, dim=-1)

    c_compressed = c_compressed.permute(0, 1, 3, 2)  # [B, H, L, d_pair]
    c_compressed = c_compressed * math.sqrt(L / N)

    out = complex_to_real_pair(c_compressed, out_dtype=x.dtype)
    return out
