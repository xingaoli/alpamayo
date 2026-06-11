#!/bin/bash
# Stage 2: Train expert module with CoC reasoning (Qwen3-VL-2B-Instruct)

cd /home/xingao/code/Alpamayo

export PYTHONPATH=/home/xingao/code/Alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3}

NUM_GPUS=${NUM_GPUS:-3}
CONFIG_PATH=/home/xingao/code/Alpamayo/finetune/sft/configs
CONFIG_NAME=sft_coc_stage2_qwen3vl2b
STAGE2_NO_COC_CKPT=${STAGE2_NO_COC_CKPT:-/home/xingao/code/Alpamayo/outputs/output_stage2_2b/checkpoint-6000}
STAGE1_COC_CKPT=${STAGE1_COC_CKPT:-/home/xingao/code/Alpamayo/outputs/output_coc_stage1_2b/checkpoint-10}

torchrun --nproc_per_node=${NUM_GPUS} \
    finetune/sft/train_hf.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.pretrained_model_name_or_path=${STAGE2_NO_COC_CKPT} \
    model.stage1_vlm_checkpoint_path=${STAGE1_COC_CKPT} \
    "$@"
