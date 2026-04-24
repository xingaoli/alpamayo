#!/bin/bash
# Stage 1: Train VLM with discrete trajectory tokens + CoC reasoning (Qwen3-VL-2B-Instruct)

cd /home/xingao/code/Alpamayo

export PYTHONPATH=/home/xingao/code/Alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3}

NUM_GPUS=${NUM_GPUS:-3}
CONFIG_PATH=/home/xingao/code/Alpamayo/finetune/sft/configs
CONFIG_NAME=sft_coc_stage1_qwen3vl2b_local

torchrun --nproc_per_node=${NUM_GPUS} \
    finetune/sft/train_hf.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    "$@"
