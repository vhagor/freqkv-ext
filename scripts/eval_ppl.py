"""Perplexity evaluation on PG-19 or Proof-pile with a chosen compressor.

This script reuses **FreqKV's** `eval.py` machinery as much as possible. It does NOT
re-implement loaders; instead it patches `dct_compress` and then dispatches into
FreqKV's eval, exactly as FreqKV's authors do.

This is GPU-heavy and intended for the H100 server. The CLI is a thin wrapper
around FreqKV's argparse interface; see ``test.sh`` in the FreqKV repo for the
original invocation pattern. Add ``--ext-method {dct,dft_lowpass,dft_rope,wavelet}``
to switch compressors.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --- Patch first, then run FreqKV's eval. ---


def _split_ext_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ext-method", default="dct",
                   choices=["dct", "dft_lowpass", "dft_rope", "wavelet"])
    p.add_argument("--ext-rope-base", type=float, default=10000.0)
    p.add_argument("--ext-head-dim", type=int, default=128)
    p.add_argument("--ext-no-rotate-key", action="store_true",
                   help="For dft_rope: disable pre-RoPE rotation wrapper (ablation).")
    p.add_argument("--ext-freqkv-root", default=None,
                   help="Path to FreqKV repo; auto-discovered if omitted.")
    args, rest = p.parse_known_args(argv)
    return args, rest


def main() -> None:
    ext_args, rest = _split_ext_args(sys.argv[1:])

    from freqkv_ext.patch import install
    install(
        method=ext_args.ext_method,
        freqkv_root=ext_args.ext_freqkv_root,
        head_dim=ext_args.ext_head_dim,
        rope_base=ext_args.ext_rope_base,
        rotate_key_before_compress=not ext_args.ext_no_rotate_key,
    )

    # Dispatch into FreqKV's `eval.py`.
    freqkv_root = ext_args.ext_freqkv_root or _autoroot()
    sys.path.insert(0, freqkv_root)
    eval_path = Path(freqkv_root) / "eval.py"
    if not eval_path.is_file():
        raise FileNotFoundError(f"FreqKV eval.py not found at {eval_path}")
    print(f"[freqkv_ext] Dispatching to FreqKV's eval.py at {eval_path}")
    sys.argv = ["eval.py", *rest]
    # Use runpy so FreqKV's __main__ guard fires.
    import runpy
    runpy.run_path(str(eval_path), run_name="__main__")


def _autoroot() -> str:
    here = Path(__file__).resolve().parents[1]  # freqkv-ext/
    candidate = here.parent / "FreqKV"
    if candidate.is_dir():
        return str(candidate)
    raise FileNotFoundError(
        f"Could not auto-locate FreqKV repo. Expected at {candidate}. "
        "Pass --ext-freqkv-root explicitly."
    )


if __name__ == "__main__":
    main()
