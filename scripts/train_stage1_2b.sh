#!/bin/bash
# Stage 1: Train VLM with discrete trajectory tokens (Qwen3-VL-2B-Instruct)

cd /home/xingao/code/NVlabs-alpamayo/alpamayo

# Load environment variables (ALPAMAYO_WORKER_NUM_THREADS etc.)
source "$(dirname "$0")/../.env" 2>/dev/null || true

export PYTHONPATH=/home/xingao/code/NVlabs-alpamayo/alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3}

# Limit per-worker threads and malloc arenas to prevent memory fragmentation.
# These must be set BEFORE process start; os.environ in Python is too late for glibc.
export ALPAMAYO_WORKER_NUM_THREADS=${ALPAMAYO_WORKER_NUM_THREADS:-2}
export OMP_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export MKL_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export OPENBLAS_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export TORCH_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export MALLOC_ARENA_MAX=$ALPAMAYO_WORKER_NUM_THREADS
# export PROCESS_TITLE=Stage1-2B
# export PROCESS_TITLE_TEMPLATE="[{title}] rank-{rank}/{world_size}"
# export RESUME_FROM_CHECKPOINT=outputs/output_stage1_2b/checkpoint-4000

NUM_GPUS=${NUM_GPUS:-3}
CONFIG_PATH=/home/xingao/code/NVlabs-alpamayo/alpamayo/finetune/sft/configs
CONFIG_NAME=sft_stage1_qwen3vl2b
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
