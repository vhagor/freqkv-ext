"""DCT-II baseline that faithfully reproduces FreqKV's `dct_compress`.

We keep this as a separate file so all compressors share the same external interface
and so that we can A/B test FreqKV vs DSP variants under identical scaffolding.

Reference: LUMIA-Group/FreqKV `llama_attn_replace_dct_mempe.dct`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# --- Low-level DCT-II / IDCT (uses FFT internally, fp32 accumulation). ---


def _dct(x: torch.Tensor, norm: str | None = "ortho") -> torch.Tensor:
    """DCT-II along the last dim. Identical to FreqKV's implementation."""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v.to(torch.float32), dim=1)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    return V


def _idct(X: torch.Tensor, norm: str | None = "ortho") -> torch.Tensor:
    """Inverse DCT-II along the last dim. Identical to FreqKV's implementation."""
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)

    v = torch.fft.ifft(V, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


# --- Public API: drop-in for FreqKV `dct_compress`. ---


def dct_compress_baseline(
    x: torch.Tensor,
    compress_len: int,
    seq_dim: int = 2,
    kv_type: str = "key",
    **_unused,
) -> torch.Tensor:
    """Sequence-axis DCT-II low-pass compression.

    Args:
        x: Tensor of shape ``[bsz, num_heads, seq_len, head_dim]``.
        compress_len: Number of low-frequency coefficients to retain (= output seq len).
        seq_dim: Sequence dim of ``x`` (kept for API parity; must be 2).
        kv_type: ``"key"`` or ``"value"`` (informational only).

    Returns:
        Compressed tensor of shape ``[bsz, num_heads, compress_len, head_dim]``.
    """
    assert seq_dim == 2, "Only seq_dim=2 supported (compatible with FreqKV)."
    if compress_len == 0:
        return x[:, :, 0:0]
    if compress_len >= x.shape[seq_dim]:
        return x

    bsz, num_heads, q_len, head_dim = x.shape
    # Merge heads into a single hidden dim for transform, as FreqKV does.
    x_flat = x.transpose(1, 2).reshape(bsz, q_len, num_heads * head_dim)

    # DCT along sequence: [B, hidden, q_len] -> [B, hidden, q_len]; truncate to compress_len.
    x_dct = _dct(x_flat.transpose(1, 2), norm="ortho")
    x_dct = x_dct[:, :, :compress_len]
    x_idct = _idct(x_dct, norm="ortho").transpose(1, 2) * math.sqrt(compress_len / q_len)

    compressed = x_idct.to(x.dtype)
    return compressed.reshape(bsz, compress_len, num_heads, head_dim).transpose(1, 2)
