#!/bin/bash
# Stage 1: Train VLM with discrete trajectory tokens (Qwen3-VL-2B-Instruct)

cd /home/xingao/code/Alpamayo

export PYTHONPATH=/home/xingao/code/Alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3}
# export PROCESS_TITLE=Stage1-2B
# export PROCESS_TITLE_TEMPLATE="[{title}] rank-{rank}/{world_size}"
# export RESUME_FROM_CHECKPOINT=outputs/output_stage1_2b/checkpoint-4000

NUM_GPUS=${NUM_GPUS:-3}
CONFIG_PATH=/home/xingao/code/Alpamayo/finetune/sft/configs
CONFIG_NAME=sft_stage1_qwen3vl2b_local
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_ARGS+=(resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}")
fi

torchrun --nproc_per_node=${NUM_GPUS} \
    finetune/sft/train_hf.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    "${RESUME_ARGS[@]}" \
    "$@"
