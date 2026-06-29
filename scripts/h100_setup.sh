#!/usr/bin/env bash
# H100 one-shot environment bootstrap for freqkv-ext.
#
# Idempotent: safe to re-run. Each step checks "already done" before acting.
#
# Inputs (via env vars, all optional):
#     WORKSPACE        default /workspace
#     PYTHON_VERSION   default 3.11
#     FREQKV_REPO      default https://github.com/LUMIA-Group/FreqKV.git
#     TORCH_INDEX      default https://download.pytorch.org/whl/cu121
#     FLASH_ATTN_PIN   default "flash-attn"  (set to e.g. "flash-attn==2.6.3" if needed)
#
# Outputs:
#     $WORKSPACE/FreqKV         (cloned reference repo)
#     $WORKSPACE/freqkv-ext     (this repo; must be rsync'd here before running)
#     $WORKSPACE/freqkv-ext/.venv  (uv-managed venv with all deps)
#
# After this script: `source $WORKSPACE/freqkv-ext/.venv/bin/activate` and run h100_run_all.sh.

set -euo pipefail

# ---- Defaults ----
WORKSPACE=${WORKSPACE:-/workspace}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
FREQKV_REPO=${FREQKV_REPO:-https://github.com/LUMIA-Group/FreqKV.git}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}
FLASH_ATTN_PIN=${FLASH_ATTN_PIN:-flash-attn}

log() { printf '\n\033[1;36m[h100_setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[h100_setup WARN]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[h100_setup ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. uv ----

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv install failed."
log "uv: $(uv --version)"

# ---- 2. Workspace layout ----

mkdir -p "$WORKSPACE"
if [ ! -d "$WORKSPACE/freqkv-ext" ]; then
    die "Missing $WORKSPACE/freqkv-ext. rsync your local checkout into place first."
fi
if [ ! -f "$WORKSPACE/freqkv-ext/pyproject.toml" ]; then
    die "$WORKSPACE/freqkv-ext exists but pyproject.toml is missing. Bad rsync."
fi

# ---- 3. Clone FreqKV reference repo ----

cd "$WORKSPACE"
if [ ! -d "$WORKSPACE/FreqKV" ]; then
    log "Cloning FreqKV reference repo..."
    git clone --depth 1 "$FREQKV_REPO" "$WORKSPACE/FreqKV"
else
    log "FreqKV repo already present."
fi
[ -f "$WORKSPACE/FreqKV/llama_attn_replace_dct_mempe.py" ] || \
    die "FreqKV repo present but llama_attn_replace_dct_mempe.py missing."

# ---- 4. Create / re-use venv ----

cd "$WORKSPACE/freqkv-ext"
if [ ! -d .venv ]; then
    log "Creating venv with Python $PYTHON_VERSION..."
    uv venv --python "$PYTHON_VERSION" .venv
else
    log "venv already exists at .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python_exec="$(pwd)/.venv/bin/python"
log "Python: $($python_exec --version)"

# ---- 5. Core dependencies ----

# (5a) torch with CUDA wheel. We install torch *first* with the CUDA index so
# subsequent installs don't pull a CPU wheel.
log "Installing torch (CUDA from $TORCH_INDEX)..."
uv pip install --python "$python_exec" \
    --extra-index-url "$TORCH_INDEX" \
    "torch>=2.1"

# (5b) Our package + GPU extras.
log "Installing freqkv-ext[gpu]..."
uv pip install --python "$python_exec" -e ".[gpu]"

# (5c) FreqKV's pinned deps (transformers==4.43.0 is critical for the monkey-patch).
log "Installing FreqKV's requirements.txt..."
uv pip install --python "$python_exec" -r "$WORKSPACE/FreqKV/requirements.txt"

# ---- 6. flash-attn (the long one) ----
# Has to be built against installed torch, so install last and with --no-build-isolation.

if "$python_exec" -c "import flash_attn" >/dev/null 2>&1; then
    log "flash-attn already importable, skipping build."
else
    log "Building flash-attn (10-30 min)..."
    if ! uv pip install --python "$python_exec" "$FLASH_ATTN_PIN" --no-build-isolation; then
        warn "flash-attn default install failed; retrying with pinned 2.6.3..."
        uv pip install --python "$python_exec" "flash-attn==2.6.3" --no-build-isolation
    fi
fi

# ---- 7. Sanity check ----

log "Sanity check..."
PYTHONPATH="$WORKSPACE/FreqKV:$WORKSPACE/freqkv-ext/src" "$python_exec" - <<'PY'
import sys
import torch
import transformers
import freqkv_ext
import llama_attn_replace_dct_mempe as fk  # noqa: F401

print("torch         :", torch.__version__)
print("torch.cuda    :", torch.cuda.is_available())
print("cuda devices  :", torch.cuda.device_count())
print("transformers  :", transformers.__version__)
print("freqkv_ext    :", freqkv_ext.__version__)
print("FreqKV module : OK (llama_attn_replace_dct_mempe imported)")

try:
    import flash_attn
    print("flash_attn    :", flash_attn.__version__)
except Exception as e:
    print("flash_attn    : NOT IMPORTABLE", repr(e))
    sys.exit(2)
PY

log "All set. Activate the env with: source $WORKSPACE/freqkv-ext/.venv/bin/activate"
log "Then run: bash scripts/h100_run_all.sh"
