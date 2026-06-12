#!/bin/bash
# Stage 1: Train VLM with discrete trajectory tokens (Qwen3-VL-2B-Instruct)
# Usage: bash scripts/eval_trajs_local.sh 4000 5000 6000
#   Or:  bash scripts/eval_trajs_local.sh 4000,5000,6000

set -euo pipefail

cd /home/xingao/code/Alpamayo

export PYTHONPATH=/home/xingao/code/Alpamayo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

NUM_GPUS=${NUM_GPUS:-2}
CHECKPOINT_DIR="outputs/output_stage1_2b"
LOG_FILE="${CHECKPOINT_DIR}/eval_results_$(date +%Y%m%d_%H%M%S).log"

# Parse checkpoint numbers from args (supports both space-separated and comma-separated)
if [ $# -eq 0 ]; then
    echo "Usage: bash $0 <ckpt_num1> [ckpt_num2] [...]" | tee -a "$LOG_FILE"
    echo "  e.g. bash $0 4000 5000 6000" | tee -a "$LOG_FILE"
    echo "  e.g. bash $0 4000,5000,6000" | tee -a "$LOG_FILE"
    exit 1
fi

IFS=', ' read -r -a CKPT_NUMS <<< "$*"

echo "============================================" | tee -a "$LOG_FILE"
echo "Evaluation started at $(date)" | tee -a "$LOG_FILE"
echo "Checkpoints to evaluate: ${CKPT_NUMS[*]}" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

for ckpt_num in "${CKPT_NUMS[@]}"; do
    ckpt_num=$(echo "$ckpt_num" | xargs)  # trim whitespace
    if [ -z "$ckpt_num" ]; then
        continue
    fi

    CKPT_PATH="${CHECKPOINT_DIR}/save-checkpoint-${ckpt_num}"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "[$(date)] WARNING: checkpoint not found: $CKPT_PATH, skipping..." | tee -a "$LOG_FILE"
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "--------------------------------------------" | tee -a "$LOG_FILE"
    echo "[$(date)] Evaluating checkpoint: save-checkpoint-${ckpt_num}" | tee -a "$LOG_FILE"
    echo "--------------------------------------------" | tee -a "$LOG_FILE"

    torchrun --nproc_per_node=${NUM_GPUS} \
        tools/eval_ar1/eval_checkpoint.py \
        --stage 1 \
        --checkpoint "$CKPT_PATH" \
        --batch_size 4 \
        --num_workers 8 \
        --vlm ckpts/Qwen3-VL-2B-Instruct \
        --max_eval_samples 20 2>&1 | tee -a "$LOG_FILE"

    echo "[$(date)] Finished evaluating save-checkpoint-${ckpt_num}" | tee -a "$LOG_FILE"

    # Sleep 60s between checkpoints to release GPU memory
    if [ "$ckpt_num" != "${CKPT_NUMS[-1]}" ]; then
        echo "[$(date)] Waiting 60s before next checkpoint..." | tee -a "$LOG_FILE"
        sleep 60
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "All evaluations completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
