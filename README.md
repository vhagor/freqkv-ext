# freqkv-ext

DSP-extended KV cache compression: drop-in **DFT (RoPE-aware)** and **wavelet**
compressors that replace FreqKV's DCT-II operator. The goal is to test two
hypotheses against FreqKV on equal footing:

1. **DFT-RoPE-aware bandpass**: by the DFT modulation theorem, applying RoPE to
   pair-complex K is a per-pair frequency shift by $\theta_i$. A RoPE-matched
   bandpass should preserve attention-relevant content better than FreqKV's
   uniform low-pass on the same compression budget — particularly for
   hidden-dim pairs with large $\theta_i$ (early hidden indices that encode
   short-range positional detail).

2. **Wavelet + adaptive thresholding**: a time-frequency localized basis should
   recover the local-detail signal (needle, code symbols, formulas) that
   FreqKV's global DCT smears away.

See [`docs/METHOD.md`](docs/METHOD.md) for the DSP derivation and the precise
algorithms.

## Layout

```
freqkv-ext/
├── pyproject.toml                  # uv-managed deps (CPU / GPU extras split)
├── src/freqkv_ext/
│   ├── transforms/
│   │   ├── dct_baseline.py         # faithful re-impl of FreqKV's DCT
│   │   ├── dft_lowpass.py          # DFT analog low-pass (sanity baseline)
│   │   ├── dft_rope_aware.py       # per-pair theta-centered bandpass (the new op)
│   │   └── wavelet.py              # DWT + hard thresholding
│   ├── rope_utils.py               # theta_i, bin offsets, real/complex pair view
│   ├── patch.py                    # monkey-patch FreqKV's dct_compress
│   └── spectrum.py                 # K spectrum extraction utilities
├── scripts/
│   ├── analyze_spectrum.py         # Experiment 1 (small): pre/post RoPE spectra
│   ├── eval_ppl.py                 # Experiment 2 (H100): PG-19 / Proof-pile PPL
│   ├── eval_longbench.py           # Experiment 3 (H100): LongBench
│   ├── eval_needle.py              # Experiment 4 (H100): Needle-in-a-Haystack
│   └── train.py                    # H100 only: fine-tune w/ chosen compressor
├── tests/                          # CPU-only unit tests
└── docs/
    ├── METHOD.md                   # DSP theory + algorithm derivations
    ├── RUN_LOCAL.md                # local sanity steps
    └── RUN_H100.md                 # H100 run book
```

## TL;DR for the H100 server operator

1. Clone both repos:

   ```bash
   git clone https://github.com/LUMIA-Group/FreqKV /workspace/FreqKV
   git clone <THIS-REPO>             /workspace/freqkv-ext
   ```

2. Install with uv (GPU extras):

   ```bash
   cd /workspace/freqkv-ext
   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install -e ".[gpu]"
   uv pip install -r /workspace/FreqKV/requirements.txt
   uv pip install flash-attn --no-build-isolation
   ```

3. Validate the DSP claim (small, fast):

   ```bash
   PYTHONPATH=src python scripts/analyze_spectrum.py \
       --model_name_or_path meta-llama/Llama-2-7b-hf \
       --seq-len 4096 --num-samples 16 \
       --layers 0 4 8 16 31 --out-dir ./out/spectrum_l2_7b
   ```

4. Train a checkpoint with each compressor, then evaluate. See
   [`docs/RUN_H100.md`](docs/RUN_H100.md) for full commands.

## What to expect

- **Spectrum analysis** should show the post-RoPE K spectrum of pair $i$
  peaks near bin $n_i = \mathrm{round}(\theta_i N / (2\pi))$. The modulation
  prediction is
  marked on each plot as a red dotted line — if these lines line up with the
  actual peaks, the DFT-RoPE hypothesis is on the right side of the data.

- **PG-19 PPL** should rank ``dft_rope ~ dct < dft_lowpass`` (DCT is provably
  near-optimal for AR(1) smooth signals via KLT, but DFT-RoPE has a structural
  reason to match it without paying the pre-RoPE caching cost). Wavelet may be
  slightly worse on PG-19 (which is smooth) but should win on needle/code.

- **LongBench QA / code subtasks** should show ``dft_rope > dct`` for the cases
  where FreqKV loses ground (HotpotQA, code, RULER NIAH at long context).

- **Needle-in-a-Haystack** is the decisive test for the DSP claim. If
  ``dft_rope`` does not improve needle recall at distances ≥ 8K, the
  bandpass-vs-lowpass story is empirically refuted; rotate effort toward the
  wavelet path or back-propagate the negative result.

## Author note

This repo deliberately keeps the FreqKV scaffolding untouched. All scripts
monkey-patch ``llama_attn_replace_dct_mempe.dct_compress`` with one of our
operators (see ``src/freqkv_ext/patch.py``) and then dispatch to FreqKV's own
``fine-tune.py`` / ``eval.py``. This makes the comparison apples-to-apples and
keeps surprise-induced bugs isolated to the compression operator.
