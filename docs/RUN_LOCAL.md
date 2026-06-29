# Local development

The local workflow is **non-GPU-heavy**: code edits, unit tests, dry-run plots.
All heavy inference / training is for the H100 box.

## Prereqs

- uv (auto-installed at ``~/.local/bin/uv``).
- Python 3.11 (uv will install one if missing).
- ~250MB free for a CPU-only torch wheel.

## Bootstrap

```bash
cd /home/vhagor/workbench/freqkv-ext

# Create a CPU-only venv (already done if this directory has .venv-cpu/).
uv venv --python 3.11 .venv-cpu
uv pip install --python .venv-cpu/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    numpy scipy pywavelets pytest matplotlib einops 'torch>=2.1'

# Or install via the package extras:
# uv pip install --python .venv-cpu/bin/python -e ".[cpu]" \
#     --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run the unit tests

```bash
PYTHONPATH="$PWD/src" ./.venv-cpu/bin/python -m pytest tests -v
```

Expected: **24 passed**. The most important test is
``test_theta_bin_offsets_match_modulation``: it numerically verifies that RoPE
applied to a baseband signal shifts its DFT peak to bin
$\theta_i N / (2\pi)$. If this test ever fails on a code change, the
RoPE-frequency-shift identity has been broken and the DFT-RoPE-aware compressor
is no longer correct.

## Dry-run spectrum analysis

This validates the plotting pipeline without loading any LLM. Uses an AR(1)
synthetic signal as a stand-in for K states.

```bash
PYTHONPATH="$PWD/src" ./.venv-cpu/bin/python scripts/analyze_spectrum.py \
    --dry-run --seq-len 256 --num-samples 2 --layers 0 4 8 \
    --device cpu --dtype float32 --out-dir ./out/spectrum_dryrun
```

Outputs to ``out/spectrum_dryrun/layerNN.png``. The pre-RoPE plot should show
log-scale energy concentrated at low bins (AR(1) has DCT/DFT energy near 0).
The post-RoPE plot should show clean per-pair peaks at the red dotted lines.

## Optional: run on RTX 5060 with a tiny model

The 8GB VRAM cannot fit LLaMA-2-7B, but small LLaMA-architecture models do
fit and give a real-data spectrum analysis (no training, just forward pass on
a handful of prompts):

```bash
# Activate the GPU venv if you've also set one up; otherwise use a separate
# venv with the CUDA torch wheel.
PYTHONPATH="$PWD/src" python scripts/analyze_spectrum.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --seq-len 1024 --num-samples 4 \
    --layers 0 4 8 16 \
    --dtype float16 --device cuda \
    --out-dir ./out/spectrum_tinyllama
```

Caveats:
- TinyLlama uses RoPE with base=10000 (same as LLaMA-2). Theta predictions on
  the plots therefore apply.
- TinyLlama's RMSNorm scaling differs from LLaMA-2; absolute peak heights are
  not directly comparable between models, but bin-position predictions are.

## Workflow tips

- Edit ``src/freqkv_ext/transforms/*.py``. Run ``pytest tests`` after each
  meaningful change.
- ``patch.py`` cannot be unit-tested locally (it needs the FreqKV repo on
  PYTHONPATH and a working attention path). Tag changes there for H100 review.
- New compression operators: add a function with the standard signature
  ``(x, compress_len, seq_dim=2, kv_type='key', **kwargs) -> Tensor``, register
  it in ``src/freqkv_ext/transforms/__init__.py::METHODS``, and write tests.
