"""DFT low-pass compressor.

This is the DFT analog of FreqKV's DCT-II low-pass. It serves two purposes:

1. **Sanity baseline**: confirms that "DFT instead of DCT" alone is roughly neutral
   (DCT is provably better on smooth signals via KLT-AR(1), but for moderate compression
   ratios the gap should be small). If DFT-lowpass beats DCT, something is wrong;
   if it matches, we've validated the DFT scaffolding.

2. **Phase-domain hook**: unlike DCT, DFT exposes the phase of each frequency bin,
   which is the prerequisite for the RoPE-aware bandpass variant.

Compression strategy
--------------------
- View each adjacent hidden-dim pair ``(d_{2i}, d_{2i+1})`` as a complex channel,
  matching LLaMA's RoPE pairing.
- DFT along the sequence axis -> ``N`` complex bins per pair.
- Keep the ``L`` lowest-frequency bins (centered at bin 0; for a 1D complex DFT this is
  bins ``[0, L/2)`` and ``[N - L/2, N)``).
- Inverse DFT of length ``L`` (using the retained spectrum directly) -> ``L`` complex
  samples = ``L`` real time-domain samples per channel.
- Rescale by ``sqrt(L / N)`` for amplitude parity, matching FreqKV's IDCT rescale.
"""

from __future__ import annotations

import math

import torch

from ..rope_utils import complex_to_real_pair, real_pair_to_complex


def _select_lowpass_bins(C: torch.Tensor, target_len: int) -> torch.Tensor:
    """Return a ``[..., target_len]`` complex tensor of the L lowest-frequency bins.

    For a DFT of length N, "low frequency" means bins near 0 and near N (which equals -1).
    We arrange them so the first ``target_len`` bins of the output IDFT form a coherent
    time-domain signal: gather ``[0..L/2)`` then ``[N - L/2 .. N)``, then IFFT of length L.
    """
    N = C.shape[-1]
    L = target_len
    if L >= N:
        return C
    half_lo = L // 2
    half_hi = L - half_lo  # for odd L, more positive freqs than negative
    # Take [0 .. half_lo) and [N - half_hi .. N).
    lo = C[..., :half_lo]
    hi = C[..., N - half_hi : N]
    return torch.cat([lo, hi], dim=-1)


def dft_lowpass_compress(
    x: torch.Tensor,
    compress_len: int,
    seq_dim: int = 2,
    kv_type: str = "key",
    **_unused,
) -> torch.Tensor:
    """DFT low-pass compression on pair-complex KV states.

    Shape contract matches `dct_compress_baseline`.
    """
    assert seq_dim == 2, "Only seq_dim=2 supported."
    if compress_len == 0:
        return x[:, :, 0:0]
    if compress_len >= x.shape[seq_dim]:
        return x

    bsz, num_heads, q_len, head_dim = x.shape
    assert head_dim % 2 == 0

    # [B, H, N, d] -> complex [B, H, N, d//2].
    c = real_pair_to_complex(x)  # complex64

    # DFT along sequence axis (dim=2).
    C = torch.fft.fft(c, dim=2)

    # Move dim=2 to last for bin selection -> back.
    C = C.permute(0, 1, 3, 2)  # [B, H, d//2, N]
    C_kept = _select_lowpass_bins(C, compress_len)  # [B, H, d//2, L]

    # IDFT of length L using the retained bins as a length-L spectrum.
    c_compressed = torch.fft.ifft(C_kept, dim=-1)  # [B, H, d//2, L]
    c_compressed = c_compressed.permute(0, 1, 3, 2)  # [B, H, L, d//2]

    # Amplitude rescale to match FreqKV's sqrt(L/N).
    c_compressed = c_compressed * math.sqrt(compress_len / q_len)

    # Back to real pairs.
    out = complex_to_real_pair(c_compressed, out_dtype=x.dtype)
    return out
