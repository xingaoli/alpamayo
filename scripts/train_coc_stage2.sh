#!/bin/bash
# Stage 2: Train expert module with CoC reasoning (Qwen3-VL-2B-Instruct)

cd /home/xingao/code/NVlabs-alpamayo/alpamayo

# Load environment variables (ALPAMAYO_WORKER_NUM_THREADS etc.)
source "$(dirname "$0")/../.env" 2>/dev/null || true

export PYTHONPATH=/home/xingao/code/NVlabs-alpamayo/alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Limit per-worker threads and malloc arenas to prevent memory fragmentation.
# These must be set BEFORE process start; os.environ in Python is too late for glibc.
export ALPAMAYO_WORKER_NUM_THREADS=${ALPAMAYO_WORKER_NUM_THREADS:-8}
export OMP_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export MKL_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export OPENBLAS_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export TORCH_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export MALLOC_ARENA_MAX=$ALPAMAYO_WORKER_NUM_THREADS

NUM_GPUS=${NUM_GPUS:-4}
CONFIG_PATH=/home/xingao/code/NVlabs-alpamayo/alpamayo/finetune/sft/configs
CONFIG_NAME=sft_coc_stage2
STAGE2_NO_COC_CKPT=${STAGE2_NO_COC_CKPT:-/home/xingao/code/NVlabs-alpamayo/alpamayo/ckpts/Alpamayo-R1-10B-from-1.5}
STAGE1_COC_CKPT=${STAGE1_COC_CKPT:-/home/xingao/code/NVlabs-alpamayo/alpamayo/outputs/output_coc_stage1/checkpoint-50}

torchrun --nproc_per_node=${NUM_GPUS} \
    finetune/sft/train_hf.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.pretrained_model_name_or_path=${STAGE2_NO_COC_CKPT} \
    model.stage1_vlm_checkpoint_path=${STAGE1_COC_CKPT} \
    "$@"
