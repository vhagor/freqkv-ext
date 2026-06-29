"""Fine-tune LLaMA with a chosen compressor for context-window extension.

Thin wrapper around FreqKV's ``fine-tune.py`` (PG-19 LM) or ``supervised-fine-tune.py``
(LongAlpaca SFT). Forwards all unknown args to FreqKV.

Example (H100, PG-19 LM at 8K, DFT-RoPE-aware):

    accelerate launch --num_processes 8 scripts/train.py \\
        --ext-method dft_rope \\
        --variant lm \\
        --model_name_or_path /path/to/Llama-2-7b-hf \\
        --bf16 True --output_dir ./out/dft_rope_8k \\
        --model_max_length 8192 --use_flash_attn True \\
        --low_rank_training True --num_train_epochs 1 \\
        --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \\
        --learning_rate 2e-5 --warmup_steps 20 --logging_steps 1 \\
        --deepspeed ../FreqKV/ds_configs/stage2.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ext-method", default="dct",
                   choices=["dct", "dft_lowpass", "dft_rope", "wavelet"])
    p.add_argument("--ext-rope-base", type=float, default=10000.0)
    p.add_argument("--ext-head-dim", type=int, default=128)
    p.add_argument("--ext-no-rotate-key", action="store_true")
    p.add_argument("--ext-freqkv-root", default=None)
    p.add_argument("--variant", choices=["lm", "sft"], default="lm",
                   help="lm: fine-tune.py (RedPajama PG-19 setup), "
                        "sft: supervised-fine-tune.py (LongAlpaca).")
    args, rest = p.parse_known_args()

    from freqkv_ext.patch import install
    install(
        method=args.ext_method,
        freqkv_root=args.ext_freqkv_root,
        head_dim=args.ext_head_dim,
        rope_base=args.ext_rope_base,
        rotate_key_before_compress=not args.ext_no_rotate_key,
    )

    freqkv_root = Path(args.ext_freqkv_root) if args.ext_freqkv_root else _autoroot()
    sys.path.insert(0, str(freqkv_root))
    target = freqkv_root / ("fine-tune.py" if args.variant == "lm"
                            else "supervised-fine-tune.py")
    if not target.is_file():
        raise FileNotFoundError(target)
    print(f"[freqkv_ext] Dispatching to {target}")
    sys.argv = [target.name, *rest]
    import runpy
    runpy.run_path(str(target), run_name="__main__")


def _autoroot() -> Path:
    here = Path(__file__).resolve().parents[1]
    cand = here.parent / "FreqKV"
    if cand.is_dir():
        return cand
    raise FileNotFoundError(
        f"Could not auto-locate FreqKV repo (expected at {cand}). "
        "Pass --ext-freqkv-root explicitly."
    )


if __name__ == "__main__":
    main()
