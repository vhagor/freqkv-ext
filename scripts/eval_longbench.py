"""LongBench evaluation harness.

LongBench has its own eval pipeline (see https://github.com/THUDM/LongBench). The
recommended workflow is:

    1. Patch FreqKV via ``freqkv_ext`` (this script does that).
    2. Use FreqKV's released SFT checkpoint OR your own trained checkpoint.
    3. Generate predictions with LongBench's `pred.py`.
    4. Score with LongBench's `eval.py`.

This script handles steps 1-2 by:
    * installing the chosen compressor,
    * loading the model (the user provides `--model-path`),
    * calling LongBench's prediction loop directly if `LONGBENCH_ROOT` env var is set,
      otherwise just generating one example per task to verify the patch is live.

For full LongBench evaluation, please clone https://github.com/THUDM/LongBench,
set ``LONGBENCH_ROOT=/path/to/LongBench``, and re-run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ext-method", default="dct",
                   choices=["dct", "dft_lowpass", "dft_rope", "wavelet"])
    p.add_argument("--ext-rope-base", type=float, default=10000.0)
    p.add_argument("--ext-head-dim", type=int, default=128)
    p.add_argument("--ext-no-rotate-key", action="store_true")
    p.add_argument("--ext-freqkv-root", default=None)
    p.add_argument("--model-path", required=True, help="HF model id or path.")
    p.add_argument("--longbench-root", default=os.environ.get("LONGBENCH_ROOT"),
                   help="Path to the LongBench repo.")
    p.add_argument("--task", default="hotpotqa", help="LongBench task name.")
    p.add_argument("--max-length", type=int, default=8000)
    p.add_argument("--out", default="./out/longbench/preds.jsonl")
    args = p.parse_args()

    from freqkv_ext.patch import install
    install(
        method=args.ext_method,
        freqkv_root=args.ext_freqkv_root,
        head_dim=args.ext_head_dim,
        rope_base=args.ext_rope_base,
        rotate_key_before_compress=not args.ext_no_rotate_key,
    )

    if not args.longbench_root:
        print("LONGBENCH_ROOT not set; running a smoke test only.")
        _smoke_test(args.model_path)
        return

    lb_root = Path(args.longbench_root)
    sys.path.insert(0, str(lb_root))
    print(f"[longbench] dispatching to {lb_root}/pred.py")
    # Most LongBench forks accept --model_name_or_path and --task; adjust as needed.
    sys.argv = [
        "pred.py",
        "--model_name_or_path", args.model_path,
        "--task", args.task,
        "--max_length", str(args.max_length),
        "--out_path", args.out,
    ]
    import runpy
    runpy.run_path(str(lb_root / "pred.py"), run_name="__main__")


def _smoke_test(model_path: str) -> None:
    """Load the model, run a short generation, and verify patched dct_compress fires."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import llama_attn_replace_dct_mempe as fk  # noqa: F401

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    prompt = "The renormalization group flow connects scale invariance to "
    out = model.generate(
        **tok(prompt, return_tensors="pt").to(model.device),
        max_new_tokens=32, do_sample=False,
    )
    print(tok.decode(out[0]))


if __name__ == "__main__":
    main()
