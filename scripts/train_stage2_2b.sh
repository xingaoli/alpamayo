#!/bin/bash
# Stage 2: Freeze VLM, train expert module (Qwen3-VL-2B-Instruct)

cd /home/xingao/code/Alpamayo

# Load environment variables (ALPAMAYO_WORKER_NUM_THREADS etc.)
source "$(dirname "$0")/../.env" 2>/dev/null || true

export PYTHONPATH=/home/xingao/code/Alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3}

# Limit per-worker threads and malloc arenas to prevent memory fragmentation.
# These must be set BEFORE process start; os.environ in Python is too late for glibc.
export ALPAMAYO_WORKER_NUM_THREADS=${ALPAMAYO_WORKER_NUM_THREADS:-2}
export OMP_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export MKL_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export OPENBLAS_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export TORCH_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export MALLOC_ARENA_MAX=$ALPAMAYO_WORKER_NUM_THREADS

# export RESUME_FROM_CHECKPOINT=outputs/output_stage2_2b/checkpoint-XXXX

NUM_GPUS=${NUM_GPUS:-3}
CONFIG_PATH=/home/xingao/code/Alpamayo/finetune/sft/configs
CONFIG_NAME=sft_stage2_qwen3vl2b
STAGE1_CKPT=${STAGE1_CKPT:-/home/xingao/code/Alpamayo/outputs/output_stage1_2b/checkpoint-10}
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_ARGS+=(resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}")
fi

MASTER_PORT=${MASTER_PORT:-29501}

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} --max-restarts=0 \
    finetune/sft/train_hf.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.stage1_checkpoint_path=${STAGE1_CKPT} \
    "${RESUME_ARGS[@]}" \
    "$@"
