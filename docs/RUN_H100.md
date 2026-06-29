# H100 run book

This document is the playbook for running ``freqkv-ext`` experiments on a
multi-H100 server. It assumes:

- 8 × H100-80GB or similar.
- Linux with CUDA 12 toolkit and a recent NVIDIA driver.
- HuggingFace access to ``meta-llama/Llama-2-7b-hf`` (or local weights).
- Network egress to PyPI and (optionally) HuggingFace Hub.

All commands below assume the working directory is ``/workspace`` and that
``FreqKV`` and ``freqkv-ext`` sit side by side there.

## 0. Layout

```
/workspace/
├── FreqKV/                         # cloned from LUMIA-Group/FreqKV
├── freqkv-ext/                     # this repo
├── models/                         # local model snapshots (optional)
├── data/                           # PG-19 / Proof-pile bins from FreqKV
├── LongBench/                      # cloned from THUDM/LongBench (optional)
└── LLMTest_NeedleInAHaystack/      # cloned from gkamradt/... (optional)
```

## 1. Install

```bash
cd /workspace
git clone https://github.com/LUMIA-Group/FreqKV.git
# (You will already have freqkv-ext on the box; copy from your local checkout.)

# uv (recommended) for fast reproducible installs:
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/freqkv-ext
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install our package + GPU extras + FreqKV's deps. FreqKV pins
# transformers==4.43.0 which is critical for the attention monkey-patch to
# match. uv will install the CUDA torch wheel as long as a CUDA driver is
# detected; force the index if needed.
uv pip install -e ".[gpu]"
uv pip install -r /workspace/FreqKV/requirements.txt

# flash-attn must be built with --no-build-isolation. This is a long install
# (10-30 min on H100; downloads CUDA includes from PyPI).
uv pip install flash-attn --no-build-isolation
```

Verify:

```bash
python -c "
import torch, transformers, llama_attn_replace_dct_mempe, freqkv_ext
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('FreqKV module loaded; freqkv_ext version', freqkv_ext.__version__)
"
```

The last line should print ``0.1.0`` and not raise.

## 2. Validate the DSP claim (gate the rest of the H100 spend)

Plot pre- and post-RoPE K spectra of LLaMA-2-7B. This consumes one GPU for ~5
minutes:

```bash
cd /workspace/freqkv-ext
PYTHONPATH=src python scripts/analyze_spectrum.py \
    --model_name_or_path /path/to/Llama-2-7b-hf \
    --seq-len 4096 --num-samples 16 \
    --layers 0 4 8 16 31 \
    --out-dir ./out/spectrum_l2_7b
```

**Decision rule**: open the post-RoPE PNGs. Each red dotted line is a predicted
$n_i = \mathrm{round}(\theta_i N / (2\pi))$. The per-pair traces should peak
near those lines. Inspect at least pair 0 (high $\theta$ $\to$ high bin) and
pair 31 (low $\theta$ $\to$ bin 0):

- **Lines hit the peaks**: proceed with the full experiment matrix.
- **Lines don't hit peaks**: post-RoPE spectrum dominated by other structure
  (RMSNorm scaling, attention sinks). Re-run with longer seq, or pre-register
  this as a negative result and pivot to wavelet path only.

## 3. Train checkpoints for each compressor

Each variant needs its own LoRA SFT pass to adapt the model to the compressor
(FreqKV does this; we follow the same convention). We use FreqKV's
``train-flash.sh`` settings as a baseline.

For ``method ∈ {dct, dft_lowpass, dft_rope, wavelet}``:

```bash
cd /workspace/freqkv-ext
accelerate launch --num_processes 8 scripts/train.py \
    --ext-method ${method} \
    --variant lm \
    --model_name_or_path /workspace/models/Llama-2-7b-hf \
    --bf16 True \
    --output_dir /workspace/ckpts/${method}_8k \
    --model_max_length 8192 \
    --use_flash_attn True \
    --low_rank_training True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_steps 20 \
    --logging_steps 1 \
    --save_strategy steps --save_steps 200 \
    --deepspeed /workspace/FreqKV/ds_configs/stage2.json
```

Notes:

- The DCT baseline checkpoint should match FreqKV's released numbers; this is
  the sanity-check anchor for the other three.
- Wavelet's CPU DWT will be the bottleneck. Either accept the speed penalty
  for a one-time training run, or pre-cache the wavelet basis as a stacked
  conv (TODO).
- For longer ``--model_max_length``, use ``ds_configs/stage3.json``.

After training:

```bash
cd /workspace/ckpts/${method}_8k && \
    python /workspace/FreqKV/zero_to_fp32.py . pytorch_model.bin && \
    bash /workspace/FreqKV/merge.sh
```

## 4. Evaluation

### 4.1 Perplexity on PG-19 / Proof-pile

```bash
for method in dct dft_lowpass dft_rope wavelet; do
  for seq in 8192 16384 32768; do
    PYTHONPATH=src python scripts/eval_ppl.py \
        --ext-method ${method} \
        --base_model /workspace/ckpts/${method}_8k/merged \
        --seq_len ${seq} \
        --context_size 8192 \
        --data_path /workspace/FreqKV/data/pg19/test.bin \
        --output_dir /workspace/out/ppl/${method}_${seq}
  done
done
```

The full set of CLI flags is the union of our ``--ext-*`` flags and FreqKV's
``eval.py`` flags. ``--ext-method`` patches the compressor; everything else is
forwarded to FreqKV's eval.

### 4.2 LongBench

```bash
git clone https://github.com/THUDM/LongBench /workspace/LongBench
export LONGBENCH_ROOT=/workspace/LongBench
for method in dct dft_lowpass dft_rope wavelet; do
  PYTHONPATH=src python scripts/eval_longbench.py \
      --ext-method ${method} \
      --model-path /workspace/ckpts/${method}_8k/merged \
      --task all  # or per-task: hotpotqa, gov_report, ...
done
# Then run LongBench's `eval.py` to score the predictions.
```

### 4.3 Needle-in-a-Haystack (the decisive test for the DFT-RoPE claim)

```bash
git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack \
    /workspace/LLMTest_NeedleInAHaystack
export NEEDLE_ROOT=/workspace/LLMTest_NeedleInAHaystack
for method in dct dft_lowpass dft_rope wavelet; do
  PYTHONPATH=src python scripts/eval_needle.py \
      --ext-method ${method} \
      --model-path /workspace/ckpts/${method}_8k/merged \
      --context-lengths 1000 2000 4000 8000 12000 16000 \
      --depths 0.0 0.25 0.5 0.75 1.0 \
      --out /workspace/out/needle/${method}.jsonl
done
```

## 5. Reporting

Suggested table layout for the writeup (mirrors FreqKV Table 2 / 3 / 4):

| Compressor | PG-19 PPL @ 32K | LongBench avg | NIAH @ 16K |
|---|---|---|---|
| DCT (FreqKV) | (baseline) | (baseline) | (baseline) |
| DFT low-pass | should match DCT | should match DCT | should match DCT |
| DFT RoPE-aware | match or improve | improve on QA / code | **expected improvement** |
| Wavelet | possibly worse | similar | possibly worse on average, better on code/needle |

Any deviation from "should match" indicates either a bug or genuine
information-theoretic content. Either is publishable, but they look very
different in the prose.

## 6. Common gotchas

- **Position offset for the DFT-RoPE wrapper**: ``patch.py`` defaults to
  ``positions = arange(N)`` when rotating pre-RoPE K. In FreqKV's iterative
  compression, the pre-RoPE K segment occupies original positions
  ``[sink_size, sink_size + fft_span)``. If you want exact post-RoPE
  positioning, export ``FREQKVEXT_KEY_OFFSET=<sink_size>`` before running.
  This matters for non-DC pair channels.

- **transformers version**: FreqKV's monkey-patch targets the
  ``LlamaAttention.forward`` signature of transformers 4.43.0. Newer
  transformers refactored the cache and position embeddings; do not upgrade
  unless you also adapt ``llama_attn_replace_dct_mempe.py``.

- **flash-attn build**: requires CUDA 12. If you get cryptic compile errors,
  ``pip install flash-attn==2.6.3 --no-build-isolation`` is a known-good pin
  for the FreqKV stack.

- **deepspeed + low_rank_training**: when training fails with a deepspeed
  shape mismatch on LoRA parameters, set ``"zero_optimization": {"stage": 2}``
  and avoid ZeRO-3 unless you also set ``"prefetch_bucket_size": auto``.
