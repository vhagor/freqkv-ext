"""Needle-in-a-Haystack evaluation harness.

We reuse https://github.com/gkamradt/LLMTest_NeedleInAHaystack. The script
patches FreqKV, then dispatches to the upstream Needle script. Provide
``NEEDLE_ROOT`` env var (or `--needle-root`).

This eval is the **most informative** for our DSP claim: if DFT-RoPE-aware
compression actually preserves localized post-RoPE information better than
FreqKV's uniform low-pass, the needle accuracy at distances exceeding the
original context window should be markedly higher.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ext-method", default="dft_rope",
                   choices=["dct", "dft_lowpass", "dft_rope", "wavelet"])
    p.add_argument("--ext-rope-base", type=float, default=10000.0)
    p.add_argument("--ext-head-dim", type=int, default=128)
    p.add_argument("--ext-no-rotate-key", action="store_true")
    p.add_argument("--ext-freqkv-root", default=None)
    p.add_argument("--model-path", required=True)
    p.add_argument("--needle-root", default=os.environ.get("NEEDLE_ROOT"))
    p.add_argument("--context-lengths", type=int, nargs="+",
                   default=[1000, 2000, 4000, 8000, 12000, 16000])
    p.add_argument("--depths", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--out", default="./out/needle/results.jsonl")
    args = p.parse_args()

    from freqkv_ext.patch import install
    install(
        method=args.ext_method,
        freqkv_root=args.ext_freqkv_root,
        head_dim=args.ext_head_dim,
        rope_base=args.ext_rope_base,
        rotate_key_before_compress=not args.ext_no_rotate_key,
    )

    if not args.needle_root:
        print(
            "NEEDLE_ROOT not set. "
            "Clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack "
            "and set NEEDLE_ROOT or pass --needle-root."
        )
        sys.exit(2)

    needle_root = Path(args.needle_root)
    sys.path.insert(0, str(needle_root))
    print(f"[needle] dispatching into {needle_root}")
    sys.argv = [
        "run.py",
        "--model_name", args.model_path,
        "--context_lengths", *map(str, args.context_lengths),
        "--depth_percents", *map(str, args.depths),
        "--output_path", args.out,
    ]
    import runpy
    runpy.run_path(str(needle_root / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
