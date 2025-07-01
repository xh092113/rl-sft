#!/usr/bin/env bash
# Iterate over every checkpoint dir and hand it off to eval_all_math.sh
# ────────────────────────────────────────────────────────────────────
set -euo pipefail
set -x  # echo commands for easier debugging

# ── Arguments (with defaults) ───────────────────────────────────────
# ROOT_DIR="/homes/gws/lxh22/rl-sft/one-input-sft/save-old/Qwen2.5-Math-1.5B"
DIR1="/local1/lxh/save/one_shot_1.5b_t0.5"
DIR2="/local1/lxh/save/Qwen2.5-Math-1.5B"
DIR3="/local1/lxh/save/Qwen2.5-Math-1.5B-filter1"
DIR4="/local1/lxh/save/Qwen2.5-Math-7B-filter1"
ROOT_DIR="/homes/gws/lxh22/rl-sft/one-shot-em/checkpoints/Qwen2.5-Math-1.5B/one_shot_1.5b_t0.5"
EVAL_SCRIPT="sh/eval_zyr.sh"   # path to the script you already have

# ── Natural-sort helper so step_2 < step_10 ─────────────────────────
natsort() { LC_COLLATE=C sort -t_ -k2,2n; }

# ── Main loop ───────────────────────────────────────────────────────
find "$ROOT_DIR" -maxdepth 1 -type d \( -name "step*" -o -name "final" \) \
  | natsort \
  | while read -r CKPT; do
        export MODEL_NAME_OR_PATH="$CKPT"
        # export OUTPUT_DIR="$CKPT/temp04/math500-eval"
        export OUTPUT_DIR="$CKPT/temp02/amc-eval"
        mkdir -p "$OUTPUT_DIR"
        bash "$EVAL_SCRIPT"
    done

echo "✅  Finished submitting all checkpoints to eval_all_math.sh"
