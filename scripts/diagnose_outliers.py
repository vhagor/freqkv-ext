"""NE0 + NE1: why does wavelet win? Basis-vs-domain + outlier attribution.

The E2 result (wavelet >> DCT >= DFT-RoPE on attention error) raised two
questions this script answers, entirely OFFLINE and in plain TEXT:

NE0 (basis vs domain)
    In E2's "natural" mode, DCT/wavelet compressed *pre-RoPE* K (smooth, easy)
    while DFT-RoPE/RST compressed *post-RoPE* K (a high-frequency comb, hard).
    So is wavelet's win a better BASIS, or just an easier DOMAIN? We reconstruct
    K in both domains with both bases and decompose the effect.

NE1 (outlier attribution)
    Hypothesis: K is heavy-tailed / outlier-dominated (attention sinks, massive
    activations). Global transforms (DCT) smear a few outlier tokens across all
    frequencies -> large error; a localized basis (wavelet) does not. We measure
    per-token energy concentration, excess kurtosis, where the DCT/wavelet error
    lands, and whether holding the top-m outlier tokens as EXACT anchors closes
    the DCT<->wavelet gap (which would justify a "DCT/wavelet + anchors" method).

Outputs: markdown tables to stdout + ``diagnose.md`` / ``diagnose.json``.

Examples
--------
CPU sanity (synthetic AR(1) + sink at token 0 + spikes):

    python scripts/diagnose_outliers.py --dry-run --seq-len 256 --num-samples 2 \\
        --layers 0 4 --device cpu --dtype float32

H100 (real model):

    python scripts/diagnose_outliers.py --model_name_or_path /root/llama2-7b/ \\
        --seq-len 2048 --num-samples 8 --layers 0 8 16 31 --out-dir results/ne1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from freqkv_ext.rdcodecs import (
    anchor_holdout_reconstruct,
    dct_keep_reconstruct,
    energy_fraction_in_tokens,
    error_localization,
    excess_kurtosis_along_seq,
    relative_frobenius_error,
    top_energy_tokens,
    token_energy_profile,
    wavelet_keep_reconstruct,
)
from freqkv_ext.spectrum import apply_llama_rope_to_key


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--dataset", default="EleutherAI/pile")
    p.add_argument("--dataset-split", default="test")
    p.add_argument("--text-field", default="text")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 8, 16, 31])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    p.add_argument("--anchor-m", type=int, nargs="+", default=[1, 4, 16],
                   help="Numbers of top-energy tokens to treat as outlier anchors.")
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--out-dir", default="results/ne1")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _synth_key(S, H, N, D, device, dtype, seed=0, rho=0.95, sink=8.0, spike_rate=0.005):
    """AR(1) smooth K + a big sink at token 0 + sparse spikes (outliers)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(S, H, N, D, dtype=torch.float32)
    x[:, :, 0] = torch.randn(S, H, D, generator=g)
    for t in range(1, N):
        x[:, :, t] = rho * x[:, :, t - 1] + math.sqrt(1 - rho ** 2) * torch.randn(S, H, D, generator=g)
    x[:, :, 0] += sink  # attention-sink-like massive token 0
    spikes = (torch.rand(S, H, N, D, generator=g) < spike_rate)
    x = x + spikes.float() * 6.0
    return x.to(device=device, dtype=dtype)


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

    # ---- gather pre-RoPE K per layer ----
    if args.dry_run or args.model_name_or_path is None:
        emit("[dry-run] synthetic AR(1)+sink+spikes K")
        S, H, D = args.num_samples, 8, 128
        k_pre_by_layer = {li: _synth_key(S, H, args.seq_len, D, dev, dtype, seed=li)
                          for li in args.layers}
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
        caps, _, _, head_dim = capture_qkv(model, tok, prompts, args.seq_len, args.layers, dev, dtype)
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        k_pre_by_layer = {c.layer_idx: c.k_pre.to(dev) for c in caps}

    emit()
    emit("# NE0 + NE1 diagnosis: why does wavelet win?")
    emit(f"- model: {args.model_name_or_path or 'dry-run'}")
    emit(f"- seq_len={args.seq_len}, samples={args.num_samples}, head_dim={head_dim}, wavelet={args.wavelet}")
    emit()

    results = {}

    for li in sorted(k_pre_by_layer.keys()):
        k_pre = k_pre_by_layer[li]
        k_post = apply_llama_rope_to_key(k_pre, rope_base=args.rope_base)
        results[f"layer{li}"] = {}
        emit(f"## Layer {li}")
        emit()

        # ---- NE0: basis x domain (K reconstruction relerr in each domain) ----
        emit("### NE0 basis × domain — K reconstruction relerr (own domain)")
        emit()
        emit("| gamma | DCT@pre | Wav@pre | DCT@post | Wav@post | dom.effect(DCT) | basis@post |")
        emit("|------:|--------:|--------:|---------:|---------:|----------------:|-----------:|")
        results[f"layer{li}"]["basis_domain"] = {}
        for g in args.gammas:
            dct_pre = relative_frobenius_error(k_pre, dct_keep_reconstruct(k_pre, g))
            wav_pre = relative_frobenius_error(k_pre, wavelet_keep_reconstruct(k_pre, g, wavelet=args.wavelet))
            dct_post = relative_frobenius_error(k_post, dct_keep_reconstruct(k_post, g))
            wav_post = relative_frobenius_error(k_post, wavelet_keep_reconstruct(k_post, g, wavelet=args.wavelet))
            dom = dct_post - dct_pre        # how much harder post is (same basis)
            basis = dct_post - wav_post     # wavelet's basis edge on the SAME (post) domain
            results[f"layer{li}"]["basis_domain"][f"gamma{g}"] = {
                "dct_pre": dct_pre, "wav_pre": wav_pre,
                "dct_post": dct_post, "wav_post": wav_post,
                "domain_effect": dom, "basis_effect_post": basis}
            emit(f"| {g:g} | {dct_pre:.4f} | {wav_pre:.4f} | {dct_post:.4f} | {wav_post:.4f} "
                 f"| {dom:+.4f} | {basis:+.4f} |")
        emit()

        # ---- NE1a: outlier statistics (on pre-RoPE K, the winning domain) ----
        prof = token_energy_profile(k_pre)
        med = float(prof.median())
        peak = float(prof.max())
        argpeak = int(prof.argmax())
        kurt = excess_kurtosis_along_seq(k_pre)
        emit("### NE1 outlier stats (pre-RoPE K)")
        emit()
        emit(f"- excess kurtosis along seq (mean over channels): **{kurt:.2f}**  (heavy tail if >> 0)")
        emit(f"- token energy: peak/median = **{peak/max(med,1e-12):.1f}**, argmax position = **{argpeak}** "
             f"({'SINK at t=0' if argpeak == 0 else 'content token'})")
        emit()
        emit("| top-m tokens | energy fraction | positions (first few) |")
        emit("|------:|----------------:|:----------------------|")
        results[f"layer{li}"]["outlier"] = {"kurtosis": kurt, "argmax": argpeak,
                                            "peak_over_median": peak / max(med, 1e-12), "frac": {}}
        for m in args.anchor_m:
            idx = top_energy_tokens(k_pre, m)
            frac = energy_fraction_in_tokens(k_pre, idx)
            pos = idx[:8].tolist()
            results[f"layer{li}"]["outlier"]["frac"][f"m{m}"] = {"frac": frac, "pos": idx.tolist()}
            emit(f"| {m} | {frac:.4f} | {pos} |")
        emit()

        # ---- NE1b: where does the reconstruction error land? ----
        emit("### NE1 error localization — fraction of squared error on top-m tokens (gamma=0.25)")
        emit()
        emit("| top-m | DCT err on anchors | Wav err on anchors |")
        emit("|------:|-------------------:|-------------------:|")
        g_loc = 0.25 if 0.25 in args.gammas else args.gammas[0]
        dct_rec = dct_keep_reconstruct(k_pre, g_loc)
        wav_rec = wavelet_keep_reconstruct(k_pre, g_loc, wavelet=args.wavelet)
        results[f"layer{li}"]["err_localization"] = {"gamma": g_loc, "rows": {}}
        for m in args.anchor_m:
            idx = top_energy_tokens(k_pre, m)
            d_loc = error_localization(k_pre, dct_rec, idx)
            w_loc = error_localization(k_pre, wav_rec, idx)
            results[f"layer{li}"]["err_localization"]["rows"][f"m{m}"] = {"dct": d_loc, "wav": w_loc}
            emit(f"| {m} | {d_loc:.4f} | {w_loc:.4f} |")
        emit()

        # ---- NE1c: does anchoring outliers close the DCT<->wavelet gap? ----
        emit("### NE1 anchor-holdout — K relerr with top-m outliers kept EXACT (budget-matched)")
        emit()
        emit("| gamma | m | DCT | DCT+anchor | Wavelet | Wav+anchor |")
        emit("|------:|--:|----:|-----------:|--------:|-----------:|")
        results[f"layer{li}"]["anchor_holdout"] = {}
        for g in args.gammas:
            for m in args.anchor_m:
                idx = top_energy_tokens(k_pre, m)
                e_dct = relative_frobenius_error(k_pre, dct_keep_reconstruct(k_pre, g))
                e_dct_a = relative_frobenius_error(
                    k_pre, anchor_holdout_reconstruct(k_pre, dct_keep_reconstruct, g, idx))
                e_wav = relative_frobenius_error(k_pre, wavelet_keep_reconstruct(k_pre, g, wavelet=args.wavelet))
                e_wav_a = relative_frobenius_error(
                    k_pre, anchor_holdout_reconstruct(
                        k_pre, wavelet_keep_reconstruct, g, idx, wavelet=args.wavelet))
                results[f"layer{li}"]["anchor_holdout"][f"g{g}_m{m}"] = {
                    "dct": e_dct, "dct_anchor": e_dct_a, "wav": e_wav, "wav_anchor": e_wav_a}
                emit(f"| {g:g} | {m} | {e_dct:.4f} | {e_dct_a:.4f} | {e_wav:.4f} | {e_wav_a:.4f} |")
        emit()

    # ---- artifacts ----
    (out_dir / "diagnose.md").write_text("\n".join(lines))
    (out_dir / "diagnose.json").write_text(json.dumps(results, indent=2))
    emit(f"[save] {out_dir/'diagnose.md'}")
    emit(f"[save] {out_dir/'diagnose.json'}")

    # ---- automatic read-out ----
    emit()
    emit("## Read-out")
    # Average across layers at gamma=0.25 (or first).
    g = 0.25 if 0.25 in args.gammas else args.gammas[0]
    basis_eff, dom_eff = [], []
    for li in k_pre_by_layer:
        bd = results[f"layer{li}"]["basis_domain"][f"gamma{g}"]
        basis_eff.append(bd["basis_effect_post"])
        dom_eff.append(bd["domain_effect"])
    mb, md = float(np.mean(basis_eff)), float(np.mean(dom_eff))
    emit(f"- NE0 @gamma={g}: mean basis-effect-on-post = {mb:+.4f} "
         f"(>0 => wavelet is a genuinely better BASIS, not just easier domain).")
    emit(f"- NE0 @gamma={g}: mean domain-effect (post vs pre, DCT) = {md:+.4f} "
         f"(>0 => post-RoPE is intrinsically harder to compress).")
    # Anchor gap closure (smallest m).
    m0 = args.anchor_m[0]
    gap, gap_a = [], []
    for li in k_pre_by_layer:
        r = results[f"layer{li}"]["anchor_holdout"][f"g{g}_m{m0}"]
        gap.append(r["dct"] - r["wav"])
        gap_a.append(r["dct_anchor"] - r["wav_anchor"])
    mg, mga = float(np.mean(gap)), float(np.mean(gap_a))
    emit(f"- NE1 @gamma={g}, m={m0}: DCT-Wavelet gap = {mg:+.4f}; after anchoring = {mga:+.4f}.")
    if mg > 0.02 and mga < 0.5 * mg:
        emit("  => Anchoring outliers closes most of the gap: wavelet's win is largely the OUTLIERS. "
             "A 'DCT/transform + exact outlier anchors' method is the cheap path; build NE2.")
    elif mg > 0.02:
        emit("  => Wavelet keeps its edge even after anchoring: the win is in the SMOOTH bulk basis too. "
             "Wavelet itself is the method; anchors are a bonus.")
    else:
        emit("  => Small gap at this gamma; inspect other gamma rows.")


if __name__ == "__main__":
    main()
