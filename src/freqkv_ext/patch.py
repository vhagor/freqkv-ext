"""Monkey-patch FreqKV's `dct_compress` symbol with one of our DSP variants.

This module assumes the FreqKV repo is importable: either present on PYTHONPATH or
sitting at ``../FreqKV`` relative to this package. The module of interest is
``llama_attn_replace_dct_mempe``.

Usage
-----

>>> import freqkv_ext.patch as fkp
>>> fkp.install(method="dft_rope")  # patch FreqKV's dct_compress with DFT-RoPE-aware

After patching, FreqKV's training / eval scripts (``fine-tune.py``, ``eval.py``,
``supervised-fine-tune.py``, etc.) will transparently use the new compressor.

The patch also installs an optional pre-rotation hook: when the chosen method is
``dft_rope`` and the call site is on the **key** branch (FreqKV passes
``kv_type="key"``), we rotate the pre-RoPE key states with RoPE before compressing,
so the compressor operates on post-RoPE K (which is the regime in which the
RoPE-matched bandpass actually makes sense).
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

from .rope_utils import default_rope_thetas
from .transforms import get_compressor

_log = logging.getLogger(__name__)


def _ensure_freqkv_importable(freqkv_root: Optional[str] = None) -> None:
    """Add the FreqKV repo root to ``sys.path`` if not already importable."""
    try:
        importlib.import_module("llama_attn_replace_dct_mempe")
        return
    except ModuleNotFoundError:
        pass

    candidates = []
    if freqkv_root:
        candidates.append(Path(freqkv_root))
    candidates.append(Path(__file__).resolve().parents[3] / "FreqKV")
    candidates.append(Path.cwd() / "FreqKV")
    candidates.append(Path.cwd().parent / "FreqKV")

    for cand in candidates:
        if (cand / "llama_attn_replace_dct_mempe.py").is_file():
            sys.path.insert(0, str(cand))
            return

    raise ModuleNotFoundError(
        "Could not locate FreqKV. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ". Pass `freqkv_root=...` explicitly or set PYTHONPATH."
    )


def _wrap_with_rope_for_key(compressor, head_dim: int, rope_base: float):
    """Wrap a compressor so that when called with `kv_type="key"` on pre-RoPE K,
    it first rotates K with RoPE up to the segment's positions, then compresses.

    This matches the regime expected by the DFT RoPE-aware operator. For the V
    branch, the wrapper is a no-op (RoPE never touches V).

    Note: FreqKV's `dct_compress` is called with K segments that are still
    pre-RoPE (RoPE is applied later inside the attention path). The exact
    positions of the segment are not passed to `dct_compress`, but for FreqKV's
    iterative compression the segment is always
    ``key_states[:, :, sink_size : sink_size + fft_span]`` — i.e. positions
    ``[sink, sink + fft_span)``. We use that convention as the default.

    The positional offset can be overridden via the ``FREQKVEXT_KEY_OFFSET``
    environment variable for non-default cache layouts.
    """
    thetas = default_rope_thetas(head_dim, base=rope_base)
    key_offset = int(os.environ.get("FREQKVEXT_KEY_OFFSET", "0"))

    def wrapped(x: torch.Tensor, compress_len: int, seq_dim: int = 2,
                kv_type: str = "key", **kwargs):
        if kv_type != "key":
            return compressor(x, compress_len, seq_dim=seq_dim, kv_type=kv_type, **kwargs)

        bsz, num_heads, N, hd = x.shape
        assert hd == head_dim, f"head_dim mismatch: wrapper got {hd}, expected {head_dim}"
        positions = torch.arange(N, device=x.device, dtype=torch.float32) + key_offset
        # Apply RoPE to x along sequence: pair adjacent hidden dims as complex,
        # multiply by exp(j theta_i t).
        x_pair = x.reshape(bsz, num_heads, N, head_dim // 2, 2)
        c = torch.view_as_complex(x_pair.to(torch.float32).contiguous())
        # phase: [N, head_dim//2]
        phase = torch.exp(
            1j * positions.unsqueeze(1) * thetas.to(x.device).unsqueeze(0)
        )
        c_rot = c * phase.unsqueeze(0).unsqueeze(0)
        x_rot_pair = torch.view_as_real(c_rot)  # [B, H, N, d/2, 2]
        x_rot = x_rot_pair.reshape(bsz, num_heads, N, head_dim).to(x.dtype)
        return compressor(x_rot, compress_len, seq_dim=seq_dim, kv_type=kv_type,
                          rope_thetas=thetas, **kwargs)

    return wrapped


def install(
    method: str = "dct",
    *,
    freqkv_root: Optional[str] = None,
    head_dim: int = 128,
    rope_base: float = 10000.0,
    rotate_key_before_compress: bool = True,
) -> None:
    """Replace FreqKV's `dct_compress` with our chosen variant.

    Args:
        method: one of ``"dct"``, ``"dft_lowpass"``, ``"dft_rope"``, ``"wavelet"``.
        freqkv_root: explicit path to the FreqKV repo (auto-discovered if None).
        head_dim: model's per-head dimension (128 for LLaMA-2-7B).
        rope_base: RoPE base (10000 for LLaMA-2, 500000 for LLaMA-3 long-ctx).
        rotate_key_before_compress: if True and method is ``dft_rope``, the
            compressor receives **post-RoPE** keys (the regime the algorithm is
            designed for). Set False for clean DFT-only ablation.
    """
    _ensure_freqkv_importable(freqkv_root)
    import llama_attn_replace_dct_mempe as fk

    base = get_compressor(method)
    if method == "dft_rope" and rotate_key_before_compress:
        compressor = _wrap_with_rope_for_key(base, head_dim=head_dim, rope_base=rope_base)
    else:
        compressor = base

    fk.dct_compress = compressor  # noqa: SLF001 (intentional monkey-patch)
    _log.info("Patched FreqKV `dct_compress` -> %s (head_dim=%d, rope_base=%s)",
              method, head_dim, rope_base)
    print(
        f"[freqkv_ext] Installed compressor: method={method}, "
        f"rotate_key_before_compress={rotate_key_before_compress and method == 'dft_rope'}"
    )


def restore() -> None:
    """Reload FreqKV's module to restore its original `dct_compress`."""
    import llama_attn_replace_dct_mempe as fk
    importlib.reload(fk)
    print("[freqkv_ext] Restored original FreqKV `dct_compress`.")
