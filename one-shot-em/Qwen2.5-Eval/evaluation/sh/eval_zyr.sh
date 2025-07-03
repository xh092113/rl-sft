set -x
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="/homes/gws/lxh22/models/Qwen2.5-Math-7B"
OUTPUT_DIR="/homes/gws/lxh22/rl-sft/one-shot-em/7B-eval-pi1+to"
mkdir -p $OUTPUT_DIR
PROMPT_TYPE="qwen25-math-cot"
MAX_TOKENS_PER_CALL="3072"
SPLIT="test"
NUM_TEST_SAMPLE=-1
#DATA_NAMES="amc23x8,minerva_math,olympiadbench,math500"
DATA_NAMES="math500-modified" # change this
IFS=',' read -ra DATASETS <<< "$DATA_NAMES"
ALL_EXIST=true

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_zyr.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAMES} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
    --overwrite 
