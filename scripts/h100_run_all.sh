#!/usr/bin/env bash
# Full freqkv-ext experiment pipeline on H100.
#
# Stages (each independently skippable via env vars):
#     STAGE_SPECTRUM=1   spectrum analysis (the DSP GO/NO-GO gate)
#     STAGE_TRAIN=1      fine-tune each compressor variant
#     STAGE_PPL=1        PG-19 / Proof-pile PPL eval on each ckpt
#     STAGE_LONGBENCH=0  LongBench (requires LONGBENCH_ROOT env var)
#     STAGE_NEEDLE=0     Needle-in-a-Haystack (requires NEEDLE_ROOT env var)
#
# Other knobs:
#     METHODS                space-separated, default "dct dft_lowpass dft_rope wavelet"
#     MODEL_PATH             HF id or local path to base model (default meta-llama/Llama-2-7b-hf)
#     OUTPUT_DIR             default $WORKSPACE/freqkv-ext/out
#     CKPT_DIR               default $WORKSPACE/ckpts
#     NUM_GPUS               default 8
#     SEQ_LEN                training context len (default 8192)
#     EVAL_SEQS              space-separated eval lengths (default "8192 16384 32768")
#     ROPE_BASE              default 10000.0
#     HEAD_DIM               default 128
#     KEY_OFFSET             FREQKVEXT_KEY_OFFSET for DFT-RoPE wrapper (default 0)
#
# Run: source the venv first.
#     source $WORKSPACE/freqkv-ext/.venv/bin/activate
#     bash scripts/h100_run_all.sh

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE/freqkv-ext/out}
CKPT_DIR=${CKPT_DIR:-$WORKSPACE/ckpts}
METHODS=${METHODS:-"dct dft_lowpass dft_rope wavelet"}
MODEL_PATH=${MODEL_PATH:-meta-llama/Llama-2-7b-hf}
NUM_GPUS=${NUM_GPUS:-8}
SEQ_LEN=${SEQ_LEN:-8192}
EVAL_SEQS=${EVAL_SEQS:-"8192 16384 32768"}
ROPE_BASE=${ROPE_BASE:-10000.0}
HEAD_DIM=${HEAD_DIM:-128}
KEY_OFFSET=${KEY_OFFSET:-0}

STAGE_SPECTRUM=${STAGE_SPECTRUM:-1}
STAGE_TRAIN=${STAGE_TRAIN:-1}
STAGE_PPL=${STAGE_PPL:-1}
STAGE_LONGBENCH=${STAGE_LONGBENCH:-0}
STAGE_NEEDLE=${STAGE_NEEDLE:-0}

export PYTHONPATH="$WORKSPACE/FreqKV:$WORKSPACE/freqkv-ext/src${PYTHONPATH:+:$PYTHONPATH}"
export FREQKVEXT_KEY_OFFSET="$KEY_OFFSET"

mkdir -p "$OUTPUT_DIR" "$CKPT_DIR"
cd "$WORKSPACE/freqkv-ext"

log() { printf '\n\033[1;36m[run_all]\033[0m %s\n' "$*"; }
sep() { printf '\033[1;34m----------------------------------------\033[0m\n'; }

# ===========================================================================
# Stage 0: Spectrum analysis (GO/NO-GO gate, ~5 min on 1 GPU)
# ===========================================================================
if [ "$STAGE_SPECTRUM" = "1" ]; then
    sep; log "STAGE 0: spectrum analysis"
    python scripts/analyze_spectrum.py \
        --model_name_or_path "$MODEL_PATH" \
        --seq-len 4096 --num-samples 16 \
        --layers 0 4 8 16 31 \
        --rope-base "$ROPE_BASE" \
        --dtype float16 \
        --out-dir "$OUTPUT_DIR/spectrum"
    log "Spectrum plots: $OUTPUT_DIR/spectrum/layer*.png"
    log "OPEN THE PNGS. If post-RoPE peaks do NOT line up with the red dotted lines,"
    log "STOP HERE and reconsider before spending H100 hours on training."
fi

# ===========================================================================
# Stage 1: Train each compressor variant
# ===========================================================================
if [ "$STAGE_TRAIN" = "1" ]; then
    sep; log "STAGE 1: training"
    for method in $METHODS; do
        out="$CKPT_DIR/${method}_${SEQ_LEN}"
        if [ -f "$out/pytorch_model.bin" ] || [ -d "$out/merged" ]; then
            log "[$method] checkpoint exists at $out, skipping."
            continue
        fi
        log "[$method] starting training, output=$out"
        accelerate launch --num_processes "$NUM_GPUS" scripts/train.py \
            --ext-method "$method" \
            --ext-rope-base "$ROPE_BASE" \
            --ext-head-dim "$HEAD_DIM" \
            --variant lm \
            --model_name_or_path "$MODEL_PATH" \
            --bf16 True \
            --output_dir "$out" \
            --model_max_length "$SEQ_LEN" \
            --use_flash_attn True \
            --low_rank_training True \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --learning_rate 2e-5 \
            --warmup_steps 20 \
            --logging_steps 1 \
            --save_strategy steps --save_steps 200 \
            --deepspeed "$WORKSPACE/FreqKV/ds_configs/stage2.json"

        # Merge LoRA into full-precision weights.
        log "[$method] merging LoRA weights..."
        (cd "$out" && python "$WORKSPACE/FreqKV/zero_to_fp32.py" . pytorch_model.bin) || \
            log "[$method] zero_to_fp32 failed; skipping merge"
        # FreqKV's merge.sh expects specific paths; user may need to edit.
        log "[$method] merged checkpoint should be under $out/merged"
    done
fi

# ===========================================================================
# Stage 2: PPL eval
# ===========================================================================
if [ "$STAGE_PPL" = "1" ]; then
    sep; log "STAGE 2: perplexity evaluation"
    for method in $METHODS; do
        ckpt="$CKPT_DIR/${method}_${SEQ_LEN}/merged"
        if [ ! -d "$ckpt" ]; then
            log "[$method] no merged ckpt at $ckpt, skipping PPL"
            continue
        fi
        for seq in $EVAL_SEQS; do
            out="$OUTPUT_DIR/ppl/${method}_${seq}"
            mkdir -p "$out"
            log "[$method @ ${seq}] -> $out"
            python scripts/eval_ppl.py \
                --ext-method "$method" \
                --ext-rope-base "$ROPE_BASE" \
                --ext-head-dim "$HEAD_DIM" \
                --base_model "$ckpt" \
                --seq_len "$seq" \
                --context_size "$SEQ_LEN" \
                --data_path "$WORKSPACE/FreqKV/data/pg19/test.bin" \
                --output_dir "$out" 2>&1 | tee "$out/log.txt" || \
                log "[$method @ ${seq}] eval failed (continuing)"
        done
    done
fi

# ===========================================================================
# Stage 3: LongBench (optional, requires LONGBENCH_ROOT)
# ===========================================================================
if [ "$STAGE_LONGBENCH" = "1" ]; then
    sep; log "STAGE 3: LongBench"
    if [ -z "${LONGBENCH_ROOT:-}" ]; then
        log "LONGBENCH_ROOT not set. Clone https://github.com/THUDM/LongBench and export the path."
    else
        for method in $METHODS; do
            ckpt="$CKPT_DIR/${method}_${SEQ_LEN}/merged"
            [ -d "$ckpt" ] || { log "[$method] no merged ckpt, skipping"; continue; }
            out="$OUTPUT_DIR/longbench/$method"
            mkdir -p "$out"
            python scripts/eval_longbench.py \
                --ext-method "$method" \
                --ext-rope-base "$ROPE_BASE" \
                --ext-head-dim "$HEAD_DIM" \
                --model-path "$ckpt" \
                --longbench-root "$LONGBENCH_ROOT" \
                --task all \
                --out "$out/preds.jsonl"
        done
    fi
fi

# ===========================================================================
# Stage 4: Needle (optional, requires NEEDLE_ROOT)
# ===========================================================================
if [ "$STAGE_NEEDLE" = "1" ]; then
    sep; log "STAGE 4: Needle-in-a-Haystack"
    if [ -z "${NEEDLE_ROOT:-}" ]; then
        log "NEEDLE_ROOT not set. Clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack."
    else
        for method in $METHODS; do
            ckpt="$CKPT_DIR/${method}_${SEQ_LEN}/merged"
            [ -d "$ckpt" ] || { log "[$method] no merged ckpt, skipping"; continue; }
            out="$OUTPUT_DIR/needle/${method}.jsonl"
            python scripts/eval_needle.py \
                --ext-method "$method" \
                --ext-rope-base "$ROPE_BASE" \
                --ext-head-dim "$HEAD_DIM" \
                --model-path "$ckpt" \
                --needle-root "$NEEDLE_ROOT" \
                --context-lengths 1000 2000 4000 8000 12000 16000 \
                --depths 0.0 0.25 0.5 0.75 1.0 \
                --out "$out"
        done
    fi
fi

sep; log "All requested stages done."
log "Spectrum -> $OUTPUT_DIR/spectrum"
log "Checkpoints -> $CKPT_DIR"
log "PPL -> $OUTPUT_DIR/ppl"
log "LongBench -> $OUTPUT_DIR/longbench"
log "Needle -> $OUTPUT_DIR/needle"
