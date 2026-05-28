#!/bin/bash
# Stage 1 low-memory training with ZeRO-3 CPU offload.

cd /home/xingao/code/Alpamayo

export PYTHONPATH=/home/xingao/code/Alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# export PROCESS_TITLE=Stage1-2B-Offload

NUM_GPUS=${NUM_GPUS:-2}
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
    trainer.deepspeed=finetune/sft/configs/deepspeed/zero3_offload_cpu.json \
    +trainer.torch_empty_cache_steps=1 \
    "${RESUME_ARGS[@]}" \
    "$@"
