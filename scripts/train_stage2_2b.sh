#!/bin/bash
# Stage 2: Freeze VLM, train expert module (Qwen3-VL-2B-Instruct)

cd /home/xingao/code/Alpamayo

export PYTHONPATH=/home/xingao/code/Alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3}

NUM_GPUS=${NUM_GPUS:-3}
CONFIG_PATH=/home/xingao/code/Alpamayo/finetune/sft/configs
CONFIG_NAME=sft_stage2_qwen3vl2b
STAGE1_CKPT=${STAGE1_CKPT:-/home/xingao/code/Alpamayo/outputs/output_stage1_2b/checkpoint-10}

torchrun --nproc_per_node=${NUM_GPUS} \
    finetune/sft/train_hf.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.stage1_checkpoint_path=${STAGE1_CKPT} \
    "$@"
