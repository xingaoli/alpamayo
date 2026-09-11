#!/bin/bash
# Evaluate a CoC-trained Stage 1 or Stage 2 checkpoint.
# Usage: bash scripts/eval_trajs_with_coc.sh
#   All arguments have sensible defaults; override via env vars or CLI args.

set -euo pipefail

STAGE=${STAGE:-2}
CHECKPOINT=${CHECKPOINT:-outputs/output_coc_stage2_2b/checkpoint-10}
STAGE1_VLM_CKPT=${STAGE1_VLM_CKPT:-outputs/output_coc_stage1_2b/checkpoint-1000}
COC_JSONL=${COC_JSONL:-data/coc.jsonl}

cd /home/xingao/code/NVlabs-alpamayo/alpamayo

export PYTHONPATH=/home/xingao/code/NVlabs-alpamayo/alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

NUM_GPUS=${NUM_GPUS:-4}
VLM=${VLM:-ckpts/Qwen3-VL-2B-Instruct}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
OUTPUT_JSONL=${OUTPUT_JSONL:-"outputs/$(basename $(dirname "$CHECKPOINT"))_coc_eval.jsonl"}

torchrun --nproc_per_node=${NUM_GPUS} \
    tools/eval_ar1/eval_ckpt_coc.py \
    --stage "$STAGE" \
    --checkpoint "$CHECKPOINT" \
    --coc_jsonl "$COC_JSONL" \
    --vlm "$VLM" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --output_jsonl "$OUTPUT_JSONL" \
    --stage1_vlm_checkpoint_path "$STAGE1_VLM_CKPT"
