"""E2 + E3: offline (training-free) rate-distortion of KV codecs.

This is the GO/NO-GO experiment for the RST hypothesis. It does NOT train.
On real captured K/V it measures, per codec and per budget gamma:

    * K reconstruction relative error,
    * V reconstruction relative error,
    * **attention-output** relative error (what actually matters),

for four codecs: DCT (FreqKV), DFT-RoPE bandpass, wavelet, and RST hybrid.
It also (a) sweeps the RST bulk/residual split ``alpha`` to find the best per
gamma, and (b) runs the E3 per-pair water-filling allocation and compares its
retained energy against uniform allocation.

ALL OUTPUT IS PLAIN TEXT (markdown tables) to stdout and to ``--out-dir`` as
``rate_distortion.md`` + ``rate_distortion.json``. No plotting.

Examples
--------
CPU sanity (synthetic AR(1)+spikes, verifies plumbing & RST mechanics):

    python scripts/rate_distortion.py --dry-run --seq-len 256 --num-samples 2 \\
        --layers 0 4 --gammas 0.5 0.25 0.125 --device cpu --dtype float32

H100 (real model):

    python scripts/rate_distortion.py \\
        --model_name_or_path /workspace/models/Llama-2-7b-hf \\
        --seq-len 2048 --num-samples 4 --layers 0 8 16 31 \\
        --gammas 0.5 0.25 0.125 0.0625 --out-dir results/rd
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from freqkv_ext.rdcodecs import (
    causal_attention_output,
    dct_keep_reconstruct,
    dft_rope_keep_reconstruct,
    pair_energy_curves,
    relative_frobenius_error,
    retained_energy,
    rst_keep_reconstruct,
    water_fill_allocation,
    wavelet_keep_reconstruct,
)
from freqkv_ext.spectrum import apply_llama_rope_to_key

CODECS = ["dct", "dft_rope", "wavelet", "rst"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--dataset", default="EleutherAI/pile")
    p.add_argument("--dataset-split", default="test")
    p.add_argument("--text-field", default="text")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 8, 16, 31])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.25, 0.125, 0.0625])
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Default RST bulk fraction for the main table.")
    p.add_argument("--alpha-sweep", type=float, nargs="+",
                   default=[0.0, 0.5, 0.7, 0.85, 1.0],
                   help="RST bulk fractions to sweep for the best-alpha table.")
    p.add_argument("--residual-domain", default="time", choices=["time", "wavelet"])
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--out-dir", default="results/rd")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--k-domain", default="natural", choices=["natural", "post"],
                   help="'natural': DCT/wavelet compress pre-RoPE then re-apply RoPE; "
                        "DFT-RoPE/RST compress post-RoPE. 'post': all on post-RoPE K.")
    return p.parse_args()


# --------------------------------------------------------------------------
# synthetic data for dry-run
# --------------------------------------------------------------------------


def _ar1_with_spikes(S, H, N, D, device, dtype, rho=0.95, spike_rate=0.01, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(S, H, N, D, dtype=torch.float32)
    x[:, :, 0] = torch.randn(S, H, D, generator=g)
    for t in range(1, N):
        x[:, :, t] = rho * x[:, :, t - 1] + math.sqrt(1 - rho**2) * torch.randn(S, H, D, generator=g)
    # Inject sparse large spikes (localized events) to give the residual something.
    spikes = (torch.rand(S, H, N, D, generator=g) < spike_rate)
    x = x + spikes.float() * 6.0
    return x.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------
# codec dispatch
# --------------------------------------------------------------------------


def _reconstruct_k(name, k_pre, k_post, gamma, alpha, args, thetas=None):
    """Return a post-RoPE reconstruction of K for the given codec."""
    if name == "dct":
        if args.k_domain == "natural":
            return apply_llama_rope_to_key(dct_keep_reconstruct(k_pre, gamma), rope_base=args.rope_base)
        return dct_keep_reconstruct(k_post, gamma)
    if name == "wavelet":
        if args.k_domain == "natural":
            kr = wavelet_keep_reconstruct(k_pre, gamma, wavelet=args.wavelet)
            return apply_llama_rope_to_key(kr, rope_base=args.rope_base)
        return wavelet_keep_reconstruct(k_post, gamma, wavelet=args.wavelet)
    if name == "dft_rope":
        return dft_rope_keep_reconstruct(k_post, gamma, rope_base=args.rope_base, is_key=True)
    if name == "rst":
        return rst_keep_reconstruct(
            k_post, gamma, alpha=alpha, rope_base=args.rope_base, is_key=True,
            residual_domain=args.residual_domain, wavelet=args.wavelet,
        )
    raise ValueError(name)


def _reconstruct_v(name, v, gamma, alpha, args):
    if name == "dct":
        return dct_keep_reconstruct(v, gamma)
    if name == "wavelet":
        return wavelet_keep_reconstruct(v, gamma, wavelet=args.wavelet)
    if name == "dft_rope":
        return dft_rope_keep_reconstruct(v, gamma, is_key=False)
    if name == "rst":
        return rst_keep_reconstruct(
            v, gamma, alpha=alpha, is_key=False,
            residual_domain=args.residual_domain, wavelet=args.wavelet,
        )
    raise ValueError(name)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    dev = args.device

    lines: list[str] = []
    def emit(s=""):
        print(s)
        lines.append(s)

    # ---- gather q/k/v per layer ----
    if args.dry_run or args.model_name_or_path is None:
        emit("[dry-run] synthetic AR(1)+spikes K/V/Q")
        S, Hq, Hkv, D = args.num_samples, 8, 8, 128
        layer_data = {}
        for li in args.layers:
            layer_data[li] = {
                "q_pre": _ar1_with_spikes(S, Hq, args.seq_len, D, dev, dtype, seed=li),
                "k_pre": _ar1_with_spikes(S, Hkv, args.seq_len, D, dev, dtype, seed=li + 100),
                "v": _ar1_with_spikes(S, Hkv, args.seq_len, D, dev, dtype, seed=li + 200),
            }
        head_dim = D
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
        from freqkv_ext.capture import capture_qkv

        emit(f"[load] {args.model_name_or_path}")
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype, device_map=dev)
        model.eval()
        ds = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
        prompts = []
        for ex in ds:
            t = ex.get(args.text_field)
            if isinstance(t, str) and len(t) >= 256:
                prompts.append(t)
            if len(prompts) >= args.num_samples:
                break
        caps, Hq, Hkv, head_dim = capture_qkv(
            model, tok, prompts, args.seq_len, args.layers, dev, dtype)
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        layer_data = {}
        for c in caps:
            layer_data[c.layer_idx] = {
                "q_pre": c.q_pre.to(dev), "k_pre": c.k_pre.to(dev), "v": c.v.to(dev)}

    emit()
    emit(f"# Rate-distortion (E2) + allocation (E3)")
    emit(f"- model: {args.model_name_or_path or 'dry-run'}")
    emit(f"- seq_len={args.seq_len}, samples={args.num_samples}, head_dim={head_dim}")
    emit(f"- k_domain={args.k_domain}, residual_domain={args.residual_domain}, rope_base={args.rope_base}")
    emit()

    results = {}

    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        k_pre = d["k_pre"]
        v = d["v"]
        q_post = apply_llama_rope_to_key(d["q_pre"], rope_base=args.rope_base)
        k_post = apply_llama_rope_to_key(k_pre, rope_base=args.rope_base)
        attn_true = causal_attention_output(q_post, k_post, v)
        attn_true_norm = torch.linalg.vector_norm(attn_true).clamp_min(1e-12)

        emit(f"## Layer {li}")
        emit()
        emit("| gamma | codec | K relerr | V relerr | attn relerr |")
        emit("|------:|:------|---------:|---------:|------------:|")
        results[f"layer{li}"] = {}

        for gamma in args.gammas:
            results[f"layer{li}"][f"gamma{gamma}"] = {}
            for name in CODECS:
                k_hat = _reconstruct_k(name, k_pre, k_post, gamma, args.alpha, args)
                v_hat = _reconstruct_v(name, v, gamma, args.alpha, args)
                k_err = relative_frobenius_error(k_post, k_hat)
                v_err = relative_frobenius_error(v, v_hat)
                attn_hat = causal_attention_output(q_post, k_hat, v_hat)
                a_err = float((torch.linalg.vector_norm(attn_true - attn_hat) / attn_true_norm).item())
                results[f"layer{li}"][f"gamma{gamma}"][name] = {
                    "k": k_err, "v": v_err, "attn": a_err}
                emit(f"| {gamma:g} | {name} | {k_err:.4f} | {v_err:.4f} | {a_err:.4f} |")
        emit()

        # ---- RST alpha sweep (attn relerr) ----
        emit(f"### RST alpha sweep — attn relerr (layer {li})")
        emit()
        header = "| gamma | " + " | ".join(f"a={a:g}" for a in args.alpha_sweep) + " | best |"
        emit(header)
        emit("|------:|" + "|".join(["---:"] * len(args.alpha_sweep)) + "|:----|")
        results[f"layer{li}"]["rst_alpha_sweep"] = {}
        for gamma in args.gammas:
            row = []
            best = (1e9, None)
            for a in args.alpha_sweep:
                k_hat = _reconstruct_k("rst", k_pre, k_post, gamma, a, args)
                v_hat = _reconstruct_v("rst", v, gamma, a, args)
                attn_hat = causal_attention_output(q_post, k_hat, v_hat)
                a_err = float((torch.linalg.vector_norm(attn_true - attn_hat) / attn_true_norm).item())
                row.append(a_err)
                if a_err < best[0]:
                    best = (a_err, a)
            results[f"layer{li}"]["rst_alpha_sweep"][f"gamma{gamma}"] = {
                "errs": row, "best_alpha": best[1]}
            emit(f"| {gamma:g} | " + " | ".join(f"{e:.4f}" for e in row) +
                 f" | a={best[1]:g} ({best[0]:.4f}) |")
        emit()

        # ---- E3 water-filling allocation ----
        curves = pair_energy_curves(k_post, rope_base=args.rope_base)
        d_pair = curves.shape[0]
        emit(f"### E3 water-filling allocation (layer {li}) — mean retained energy")
        emit()
        emit("| gamma | uniform | water-fill | gain |")
        emit("|------:|--------:|-----------:|-----:|")
        results[f"layer{li}"]["allocation"] = {}
        for gamma in args.gammas:
            L = max(1, int(round(gamma * args.seq_len)))
            total_bins = L * d_pair
            uni = np.full(d_pair, L, dtype=np.int64)
            wf = water_fill_allocation(curves, total_bins)
            e_uni = retained_energy(curves, uni)
            e_wf = retained_energy(curves, wf)
            results[f"layer{li}"]["allocation"][f"gamma{gamma}"] = {
                "uniform": e_uni, "waterfill": e_wf}
            emit(f"| {gamma:g} | {e_uni:.4f} | {e_wf:.4f} | {e_wf - e_uni:+.4f} |")
        emit()

    # ---- write artifacts ----
    (out_dir / "rate_distortion.md").write_text("\n".join(lines))
    (out_dir / "rate_distortion.json").write_text(json.dumps(results, indent=2))
    emit(f"[save] {out_dir/'rate_distortion.md'}")
    emit(f"[save] {out_dir/'rate_distortion.json'}")

    # ---- decision hint ----
    emit()
    emit("## GO/NO-GO hint")
    # Aggregate attn relerr: does RST beat DCT at the smallest gamma?
    g_small = min(args.gammas)
    deltas = []
    for li in layer_data:
        r = results[f"layer{li}"][f"gamma{g_small}"]
        deltas.append(r["dct"]["attn"] - r["rst"]["attn"])
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    emit(f"- At gamma={g_small}, mean(attn relerr DCT - RST) = {mean_delta:+.4f} "
         f"(positive => RST better).")
    if mean_delta > 0.01:
        emit("- RST reduces attention error vs DCT at high compression. C2 supported; proceed to E4.")
    elif mean_delta < -0.01:
        emit("- RST does NOT beat DCT. Reconsider the bulk+residual split before training.")
    else:
        emit("- RST ~ DCT at this gamma. Check smaller gamma / needle-like inputs (not PG-19).")


if __name__ == "__main__":
    main()
