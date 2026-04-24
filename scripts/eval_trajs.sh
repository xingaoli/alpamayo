#!/bin/bash
# Stage 1: Train VLM with discrete trajectory tokens (Qwen3-VL-2B-Instruct)

cd /home/xingao/code/Alpamayo

export PYTHONPATH=/home/xingao/code/Alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

NUM_GPUS=${NUM_GPUS:-4}

torchrun --nproc_per_node=${NUM_GPUS} \
    tools/eval_checkpoint.py \
    --stage 1 \
    --checkpoint outputs/output_stage1_2b/checkpoint-8982 \
    --batch_size 8 \
    --num_workers 16 \
    --vlm ckpts/Qwen3-VL-2B-Instruct
