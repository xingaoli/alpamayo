#!/bin/bash
# Stage 1: Train VLM with discrete trajectory tokens (Qwen3-VL-2B-Instruct)

cd /home/xingao/code/NVlabs-alpamayo/alpamayo

export PYTHONPATH=/home/xingao/code/NVlabs-alpamayo/alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

NUM_GPUS=${NUM_GPUS:-4}

torchrun --nproc_per_node=${NUM_GPUS} \
    tools/eval_ar1/eval_checkpoint.py \
    --stage 2 \
    --checkpoint outputs/output_stage2_2b/checkpoint-1497 \
    --batch_size 8 \
    --num_workers 16 \
    --vlm ckpts/Qwen3-VL-2B-Instruct \
    --max_eval_samples 9999 \
    --stage1_vlm_checkpoint_path outputs/output_stage1_2b/checkpoint-2994
