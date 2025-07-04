#!/usr/bin/env bash
# mush be run under evaluation/ directory, not under evaluation/sh/
set -euo pipefail
set -x  

# ── Arguments (with defaults) ───────────────────────────────────────
ROOT_DIR="/homes/gws/lxh22/rl-sft/one-shot-em/checkpoints/Qwen2.5-Math-1.5B/one_shot/"
EVAL_SCRIPT="./sh/eval_all_math.sh"   

# ── Natural-sort helper so step_2 < step_10 ─────────────────────────
natsort() { LC_COLLATE=C sort -t_ -k2,2n; }

find "$ROOT_DIR" -maxdepth 1 -type d \( -name "step_*" -o -name "final" \) \
  | natsort \
  | while read -r CKPT; do
        export MODEL_NAME_OR_PATH="$CKPT"
        export OUTPUT_DIR="$CKPT/temp00/eval"
        mkdir -p "$OUTPUT_DIR"
        bash "$EVAL_SCRIPT"
    done

echo "✅  Finished submitting all checkpoints to eval_all_math.sh"