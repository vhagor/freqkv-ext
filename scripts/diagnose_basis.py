"""NE2': basis vs selection-rule + smoothness characterization.

NE0/NE1 established that wavelet beats DCT as a *basis* (not just an easier
domain) and that the win is NOT driven by token outliers. Two questions remain
before committing to "wavelet is the method":

A. SELECTION vs BASIS confound.
   FreqKV's DCT keeps the LOWEST L coefficients (low-pass); our wavelet keeps
   the L LARGEST coefficients (adaptive magnitude threshold). Part of wavelet's
   advantage could be the adaptive *selection rule*, not the basis. We compare:
       - DCT low-pass (FreqKV),
       - DCT top-k (adaptive selection, same basis),
       - Wavelet top-k (adaptive selection, different basis).
   (DCT low-pass -> DCT top-k) = selection effect.
   (DCT top-k   -> Wavelet top-k) = pure basis effect.

B. WHY is wavelet a better basis? (smoothness class)
   If K along sequence were stationary AR(1) (rho->1), DCT ~ KLT would be hard
   to beat. The large wavelet win suggests K is piecewise-smooth / bounded
   variation (locally smooth with abrupt boundaries). Signature: VALUE kurtosis
   is small but FIRST-DIFFERENCE kurtosis is large (sparse, spiky edges), and
   wavelet energy concentrates in the approx band + a few detail coefficients.

All output is plain TEXT. Examples
--------
CPU sanity:

    python scripts/diagnose_basis.py --dry-run --seq-len 256 --num-samples 2 \\
        --layers 0 4 --device cpu --dtype float32

H100:

    python scripts/diagnose_basis.py --model_name_or_path /root/llama2-7b/ \\
        --seq-len 2048 --num-samples 8 --layers 0 8 16 31 --out-dir results/ne2b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from freqkv_ext.rdcodecs import (
    dct_keep_reconstruct,
    excess_kurtosis_along_seq,
    first_difference_kurtosis,
    relative_frobenius_error,
    wavelet_keep_reconstruct,
)
from freqkv_ext.sparsity import (
    dct_sparsity_per_trace,
    sample_traces,
    wavelet_band_energy,
    wavelet_sparsity_per_trace,
)


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
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--sparsity-samples", type=int, default=512)
    p.add_argument("--out-dir", default="results/ne2b")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _synth_piecewise(S, H, N, D, device, dtype, seed=0, n_segments=8):
    """Piecewise-smooth K: AR(1) within segments, level jumps at boundaries.

    Designed so the *value* kurtosis is modest but *difference* kurtosis is high
    (the bounded-variation signature) -- to exercise the smoothness diagnostics.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(S, H, N, D, dtype=torch.float32)
    rho = 0.9
    x[:, :, 0] = torch.randn(S, H, D, generator=g)
    bounds = sorted(torch.randint(1, N, (n_segments,), generator=g).tolist())
    jump = torch.zeros(S, H, D)
    for t in range(1, N):
        if t in bounds:
            jump = jump + torch.randn(S, H, D, generator=g) * 3.0
        x[:, :, t] = rho * x[:, :, t - 1] + math.sqrt(1 - rho ** 2) * torch.randn(S, H, D, generator=g)
    # Add piecewise level offsets.
    seg = torch.zeros(S, H, N, D)
    cur = torch.zeros(S, H, D)
    bi = 0
    for t in range(N):
        if bi < len(bounds) and t == bounds[bi]:
            cur = cur + torch.randn(S, H, D, generator=g) * 3.0
            bi += 1
        seg[:, :, t] = cur
    return (x + seg).to(device=device, dtype=dtype)


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

    if args.dry_run or args.model_name_or_path is None:
        emit("[dry-run] synthetic piecewise-smooth K")
        S, H, D = args.num_samples, 8, 128
        k_by_layer = {li: _synth_piecewise(S, H, args.seq_len, D, dev, dtype, seed=li)
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
        k_by_layer = {c.layer_idx: c.k_pre.to(dev) for c in caps}

    emit()
    emit("# NE2' basis-vs-selection + smoothness")
    emit(f"- model: {args.model_name_or_path or 'dry-run'}")
    emit(f"- seq_len={args.seq_len}, samples={args.num_samples}, head_dim={head_dim}, wavelet={args.wavelet}")
    emit()

    results = {}
    sel_gain_all, basis_gain_all = [], []

    for li in sorted(k_by_layer.keys()):
        K = k_by_layer[li]
        results[f"layer{li}"] = {}
        emit(f"## Layer {li}")
        emit()

        # ---- A. selection vs basis (K reconstruction relerr, pre-RoPE) ----
        emit("### A. selection vs basis — K relerr")
        emit()
        emit("| gamma | DCT-lowpass | DCT-topk | Wavelet-topk | selection effect | basis effect |")
        emit("|------:|------------:|---------:|-------------:|-----------------:|-------------:|")
        results[f"layer{li}"]["selection_basis"] = {}
        for g in args.gammas:
            e_lp = relative_frobenius_error(K, dct_keep_reconstruct(K, g, select="lowpass"))
            e_tk = relative_frobenius_error(K, dct_keep_reconstruct(K, g, select="topk"))
            e_wv = relative_frobenius_error(K, wavelet_keep_reconstruct(K, g, wavelet=args.wavelet))
            sel = e_lp - e_tk      # how much adaptive selection helps DCT
            basis = e_tk - e_wv    # pure basis effect (both adaptive)
            sel_gain_all.append(sel)
            basis_gain_all.append(basis)
            results[f"layer{li}"]["selection_basis"][f"gamma{g}"] = {
                "dct_lowpass": e_lp, "dct_topk": e_tk, "wavelet_topk": e_wv,
                "selection_effect": sel, "basis_effect": basis}
            emit(f"| {g:g} | {e_lp:.4f} | {e_tk:.4f} | {e_wv:.4f} | {sel:+.4f} | {basis:+.4f} |")
        emit()

        # ---- B. smoothness class ----
        val_k = excess_kurtosis_along_seq(K)
        dif_k = first_difference_kurtosis(K)
        traces = sample_traces(K, args.sparsity_samples, seed=li)
        approx_frac, detail_fracs = wavelet_band_energy(traces, wavelet=args.wavelet)
        dct_s90 = float(np.median(dct_sparsity_per_trace(traces, target=0.90)))
        wav_s90 = float(np.median(wavelet_sparsity_per_trace(traces, target=0.90, wavelet=args.wavelet)))
        dct_s99 = float(np.median(dct_sparsity_per_trace(traces, target=0.99)))
        wav_s99 = float(np.median(wavelet_sparsity_per_trace(traces, target=0.99, wavelet=args.wavelet)))
        results[f"layer{li}"]["smoothness"] = {
            "value_kurtosis": val_k, "diff_kurtosis": dif_k,
            "approx_energy_frac": approx_frac, "detail_energy_fracs": detail_fracs.tolist(),
            "dct_sparsity90": dct_s90, "wav_sparsity90": wav_s90,
            "dct_sparsity99": dct_s99, "wav_sparsity99": wav_s99}
        emit("### B. smoothness class")
        emit()
        emit(f"- VALUE kurtosis = **{val_k:.2f}**, FIRST-DIFFERENCE kurtosis = **{dif_k:.2f}**  "
             f"(diff >> value => piecewise-smooth / bounded-variation edges)")
        emit(f"- wavelet energy: approx band = **{approx_frac:.3f}**, "
             f"detail (coarse->fine) = {[round(float(x), 3) for x in detail_fracs]}")
        emit(f"- n-term sparsity @90% energy: DCT={dct_s90:.3f}, Wavelet={wav_s90:.3f}  |  "
             f"@99%: DCT={dct_s99:.3f}, Wavelet={wav_s99:.3f}  (lower=better)")
        emit()

    # ---- artifacts + read-out ----
    (out_dir / "diagnose_basis.md").write_text("\n".join(lines))
    (out_dir / "diagnose_basis.json").write_text(json.dumps(results, indent=2))
    emit(f"[save] {out_dir/'diagnose_basis.md'}")
    emit(f"[save] {out_dir/'diagnose_basis.json'}")

    emit()
    emit("## Read-out")
    ms = float(np.mean(sel_gain_all))
    mb = float(np.mean(basis_gain_all))
    emit(f"- mean selection effect (DCT-lowpass -> DCT-topk) = {ms:+.4f}")
    emit(f"- mean pure basis effect (DCT-topk -> Wavelet-topk) = {mb:+.4f}")
    if mb > 1.5 * max(ms, 1e-6):
        emit("  => Most of the win is the BASIS. Wavelet is the method; adaptive DCT is not enough.")
    elif ms > 1.5 * max(mb, 1e-6):
        emit("  => Most of the win is the SELECTION RULE. An adaptive (top-k) DCT may suffice — "
             "cheaper than wavelet and keeps FreqKV's fast transform. Test adaptive-DCT end to end.")
    else:
        emit("  => Selection and basis contribute comparably. Use adaptive selection AND a wavelet basis.")


if __name__ == "__main__":
    main()
