"""Extract and plot the sequence-axis power spectrum of K states from a LLaMA model.

This is the **central DSP-validation experiment**. It does NOT train, and is light
enough to run on a small GPU (RTX 5060, 8GB) IF the model fits. We support:

    * Vanilla HuggingFace LLaMA models (`--model_name_or_path`).
    * Choosing which layers / heads to plot.
    * Limiting calibration size to keep memory low.

Outputs:
    * One PNG per layer with pre-RoPE and post-RoPE average power spectra,
      annotated with the predicted theta_i shift for selected pairs.
    * A NumPy archive of raw spectra for further offline analysis.

This script is intentionally self-contained and avoids touching FreqKV's
attention monkey-patch — we want the spectra of the *unmodified* model.

Example
-------

CPU sanity check (no real model, uses random init; verifies plumbing):

    uv run python scripts/analyze_spectrum.py --dry-run --seq-len 512

Small GPU (LLaMA-2-7B will OOM on 8GB; pick a smaller model):

    uv run python scripts/analyze_spectrum.py \\
        --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --seq-len 1024 --num-samples 4 \\
        --layers 0 4 8 16 --out-dir ./out/spectrum_tiny

H100:

    uv run python scripts/analyze_spectrum.py \\
        --model_name_or_path meta-llama/Llama-2-7b-hf \\
        --seq-len 4096 --num-samples 16 \\
        --layers 0 4 8 16 31 --out-dir ./out/spectrum_l2_7b
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from freqkv_ext.spectrum import (
    LayerSpectrum,
    SpectrumResult,
    apply_llama_rope_to_key,
    power_spectrum_pair_complex,
)
from freqkv_ext.sparsity import (
    dct_sparsity_per_trace,
    dft_sparsity_per_trace,
    sample_traces,
    wavelet_sparsity_per_trace,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model_name_or_path", default=None,
                   help="HF model id or local path. If omitted, runs in dry-run mode.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip model loading; use a random tensor (sanity only).")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--num-samples", type=int, default=8,
                   help="Number of calibration prompts to average over.")
    p.add_argument("--dataset", default="EleutherAI/pile",
                   help="HF dataset for calibration (only when not --dry-run).")
    p.add_argument("--dataset-split", default="test", help="Dataset split.")
    p.add_argument("--text-field", default="text", help="Field containing input text.")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 31],
                   help="Layer indices to plot.")
    p.add_argument("--heads", type=int, nargs="+", default=None,
                   help="Optional subset of head indices to plot (default: all averaged).")
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--out-dir", default="./out/spectrum")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # ---- Sparsity (wavelet-vs-DCT gate) analysis ----
    p.add_argument("--skip-sparsity", action="store_true",
                   help="Skip the wavelet/DCT/DFT sparsity comparison.")
    p.add_argument("--sparsity-samples", type=int, default=256,
                   help="Number of per-channel sequence-axis traces to sample per layer.")
    p.add_argument("--energy-target", type=float, default=0.95,
                   help="Target fraction of L2 energy for the sparsity comparison.")
    p.add_argument("--wavelet", default="db4",
                   help="PyWavelets family name for the wavelet sparsity analysis.")
    return p.parse_args()


def _make_random_batch(seq_len: int, num_samples: int, head_dim: int,
                       num_heads: int, num_layers: int,
                       device: str, dtype: torch.dtype):
    """Synthetic K states with AR(1)-like correlation along sequence."""
    torch.manual_seed(0)
    out = []
    rho = 0.95
    for _ in range(num_layers):
        x = torch.zeros(num_samples, num_heads, seq_len, head_dim, dtype=torch.float32)
        x[:, :, 0] = torch.randn_like(x[:, :, 0])
        for t in range(1, seq_len):
            x[:, :, t] = rho * x[:, :, t - 1] + math.sqrt(1 - rho**2) * torch.randn_like(x[:, :, 0])
        out.append(x.to(device=device, dtype=dtype))
    return out


def _collect_pre_rope_keys_via_hook(model, tokenizer, prompts, seq_len, device, dtype):
    """Register forward hooks on each `k_proj`; run prompts; return list[num_layers]
    of `[total_samples, num_heads, seq_len, head_dim]` tensors (CPU fp32)."""
    pre_rope = []
    hooks = []

    # Discover number of layers and shapes from the model config.
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

    layer_modules = list(model.model.layers)
    cap = [None] * len(layer_modules)

    for li, layer in enumerate(layer_modules):
        k_proj = layer.self_attn.k_proj

        def _hook(module, inp, out, _li=li):
            # out: [B, N, num_kv_heads * head_dim] (sometimes [B, N, hidden_size])
            B, N, hidden = out.shape
            out_view = out.detach().to(torch.float32).cpu().reshape(B, N, num_heads, head_dim).transpose(1, 2)
            cap[_li] = out_view

        hooks.append(k_proj.register_forward_hook(_hook))

    try:
        all_caps = [[] for _ in layer_modules]
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
                padding="max_length",
            ).to(device)
            with torch.no_grad():
                model(**inputs)
            for li in range(len(layer_modules)):
                all_caps[li].append(cap[li])
        result = [torch.cat(c, dim=0) if c else None for c in all_caps]
    finally:
        for h in hooks:
            h.remove()
    return result, num_heads, head_dim


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    if args.dry_run or args.model_name_or_path is None:
        print("[dry-run] Synthetic AR(1) K states.")
        head_dim, num_heads, num_layers = 128, 32, 32
        keys_per_layer = _make_random_batch(
            args.seq_len, args.num_samples, head_dim, num_heads, num_layers,
            device=args.device, dtype=dtype,
        )
    else:
        print(f"[load] {args.model_name_or_path}")
        # Import here so the dry-run path doesn't need transformers / datasets.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            device_map=args.device,
        )
        model.eval()

        print(f"[data] {args.dataset} / {args.dataset_split}, {args.num_samples} samples.")
        ds = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
        prompts = []
        for ex in ds:
            t = ex.get(args.text_field)
            if not isinstance(t, str) or len(t) < 256:
                continue
            prompts.append(t)
            if len(prompts) >= args.num_samples:
                break
        if len(prompts) < args.num_samples:
            print(f"WARNING: only collected {len(prompts)}/{args.num_samples} samples.")

        keys_per_layer, num_heads, head_dim = _collect_pre_rope_keys_via_hook(
            model, tokenizer, prompts, args.seq_len, args.device, dtype,
        )
        num_layers = len(keys_per_layer)
        # Free model weights ASAP -- we no longer need them.
        del model
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    # Compute spectra per layer.
    result = SpectrumResult(seq_len=args.seq_len, head_dim=head_dim)
    for li, k in enumerate(keys_per_layer):
        if k is None:
            continue
        if li not in args.layers:
            continue
        pre = power_spectrum_pair_complex(k)
        post_k = apply_llama_rope_to_key(k, rope_base=args.rope_base)
        post = power_spectrum_pair_complex(post_k)
        result.layers.append(LayerSpectrum(layer_idx=li, pre_rope_power=pre,
                                          post_rope_power=post))
        # Free.
        del post_k

    # Save raw spectra.
    spectra_npz = out_dir / "spectra.npz"
    save_dict = {
        f"layer{ls.layer_idx}_pre": ls.pre_rope_power for ls in result.layers
    } | {
        f"layer{ls.layer_idx}_post": ls.post_rope_power for ls in result.layers
    }
    np.savez_compressed(spectra_npz, **save_dict)
    (out_dir / "config.json").write_text(json.dumps({
        "seq_len": args.seq_len, "head_dim": head_dim, "num_heads": num_heads,
        "rope_base": args.rope_base, "layers": args.layers,
        "model": args.model_name_or_path or "dry-run-AR1",
    }, indent=2))
    print(f"[save] raw spectra -> {spectra_npz}")

    # Plot. Lazy import: only require matplotlib at output time.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for ls in result.layers:
        pre = ls.pre_rope_power  # [d_pair, N]
        post = ls.post_rope_power
        N = pre.shape[1]
        # Average across pairs (and overlay a few selected pairs).
        bin_axis = np.arange(N)
        # Predicted theta_i bin per pair for overlay.
        i = np.arange(head_dim // 2)
        thetas = args.rope_base ** (-2.0 * i / head_dim)
        bin_centers = (thetas * N / (2 * np.pi)) % N

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        axs[0].plot(bin_axis, pre.mean(axis=0), label="avg over pairs")
        for p in (0, head_dim // 4, head_dim // 2 - 1):
            axs[0].plot(bin_axis, pre[p], alpha=0.4, label=f"pair {p}")
        axs[0].set_title(f"Layer {ls.layer_idx} | pre-RoPE K power")
        axs[0].set_xlabel("DFT bin"); axs[0].set_yscale("log"); axs[0].legend()

        axs[1].plot(bin_axis, post.mean(axis=0), label="avg over pairs")
        for p in (0, head_dim // 4, head_dim // 2 - 1):
            axs[1].plot(bin_axis, post[p], alpha=0.4, label=f"pair {p}")
            axs[1].axvline(bin_centers[p], color="r", linestyle=":", alpha=0.3)
        axs[1].set_title(f"Layer {ls.layer_idx} | post-RoPE K power (red dots = predicted n_i)")
        axs[1].set_xlabel("DFT bin"); axs[1].legend()

        fig.tight_layout()
        png = out_dir / f"layer{ls.layer_idx:02d}.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"[plot] layer {ls.layer_idx} -> {png}")

    # ------------------------------------------------------------------
    # Sparsity analysis: per layer, sample per-channel traces from
    # pre-RoPE K and compute the 95%-energy fraction in DCT / DFT / wavelet.
    # This is the GO/NO-GO gate for the wavelet path.
    # ------------------------------------------------------------------
    if args.skip_sparsity:
        print("[sparsity] skipped via --skip-sparsity.")
        return

    print(f"[sparsity] target={args.energy_target:.2f}, "
          f"samples/layer={args.sparsity_samples}, wavelet={args.wavelet}")
    sparsity_summary: dict[str, dict] = {}
    layer_rows = []
    for li, k in enumerate(keys_per_layer):
        if k is None or li not in args.layers:
            continue
        traces = sample_traces(k, args.sparsity_samples, seed=li)
        dct_s = dct_sparsity_per_trace(traces, target=args.energy_target)
        dft_s = dft_sparsity_per_trace(traces, target=args.energy_target)
        try:
            wav_s = wavelet_sparsity_per_trace(
                traces, target=args.energy_target, wavelet=args.wavelet,
            )
        except RuntimeError as e:
            print(f"[sparsity] wavelet unavailable: {e}")
            wav_s = np.full_like(dct_s, np.nan)

        def _pct(a, q):
            return float(np.nanpercentile(a, q)) if a.size else float("nan")

        sparsity_summary[f"layer{li}"] = {
            "dct":     {"p10": _pct(dct_s, 10), "p50": _pct(dct_s, 50), "p90": _pct(dct_s, 90)},
            "dft":     {"p10": _pct(dft_s, 10), "p50": _pct(dft_s, 50), "p90": _pct(dft_s, 90)},
            "wavelet": {"p10": _pct(wav_s, 10), "p50": _pct(wav_s, 50), "p90": _pct(wav_s, 90)},
        }
        layer_rows.append((li, dct_s, dft_s, wav_s))

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        parts = ax.violinplot(
            [dct_s, dft_s, wav_s[~np.isnan(wav_s)]],
            showmedians=True, showextrema=False,
        )
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["DCT", "DFT (rfft)", f"Wavelet ({args.wavelet})"])
        ax.set_ylabel(f"fraction of coeffs for {int(args.energy_target*100)}% energy")
        ax.set_title(f"Layer {li} sparsity (lower = more compressible)")
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        fig.tight_layout()
        sp_png = out_dir / f"layer{li:02d}_sparsity.png"
        fig.savefig(sp_png, dpi=140)
        plt.close(fig)
        print(f"[sparsity] layer {li}: "
              f"DCT p50={sparsity_summary[f'layer{li}']['dct']['p50']:.3f}  "
              f"DFT p50={sparsity_summary[f'layer{li}']['dft']['p50']:.3f}  "
              f"Wavelet p50={sparsity_summary[f'layer{li}']['wavelet']['p50']:.3f}  "
              f"-> {sp_png.name}")

    if layer_rows:
        layers = [r[0] for r in layer_rows]
        dct_med = [np.median(r[1]) for r in layer_rows]
        dft_med = [np.median(r[2]) for r in layer_rows]
        wav_med = [np.nanmedian(r[3]) for r in layer_rows]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(layers, dct_med, "o-", label="DCT median")
        ax.plot(layers, dft_med, "s-", label="DFT median")
        ax.plot(layers, wav_med, "^-", label=f"Wavelet ({args.wavelet}) median")
        ax.set_xlabel("layer index")
        ax.set_ylabel(f"median fraction of coeffs @ {int(args.energy_target*100)}% energy")
        ax.set_title("Per-layer sparsity comparison (lower = more compressible)")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        sum_png = out_dir / "sparsity_summary.png"
        fig.savefig(sum_png, dpi=140)
        plt.close(fig)
        print(f"[sparsity] summary -> {sum_png}")

    (out_dir / "sparsity.json").write_text(json.dumps(sparsity_summary, indent=2))
    print(f"[sparsity] json -> {out_dir / 'sparsity.json'}")

    # Decision rule printed to stdout.
    if layer_rows:
        dct_p50 = float(np.median([np.median(r[1]) for r in layer_rows]))
        wav_p50 = float(np.nanmedian([np.nanmedian(r[3]) for r in layer_rows]))
        delta = dct_p50 - wav_p50
        print()
        print(f"[gate] Across plotted layers, median p50(DCT)={dct_p50:.3f}, "
              f"p50(Wavelet)={wav_p50:.3f}, delta={delta:+.3f}.")
        if delta > 0.05:
            print("[gate] Wavelet is materially more compressible than DCT on this model. "
                  "Wavelet path has a structural foothold; pursue.")
        elif delta < -0.05:
            print("[gate] DCT is materially more compressible than wavelet on this model. "
                  "Wavelet path has no structural advantage on this data; deprioritize.")
        else:
            print("[gate] DCT and wavelet are within noise on average compressibility. "
                  "Wavelet wins (if any) will need to come from per-channel / per-token "
                  "tail behavior, not the bulk.")


if __name__ == "__main__":
    main()
