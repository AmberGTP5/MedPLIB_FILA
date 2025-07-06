#!/bin/bash
# Stage 2: VQA Expert SFT Launch Script (Simplified with Hardcoded Paths in Python)

set -euo pipefail

# --- 配置区 ---
# 只需要定义输出目录和超参数
OUTPUT_DIR="../runs/stage2-vqa-deepspeed-$(date +%Y%m%d_%H%M%S)"
LEARNING_RATE="2e-4"
EPOCHS=1
BATCH_SIZE_PER_GPU=2
GRAD_ACCUM_STEPS=32
SAVE_STEPS=500

# --- DeepSpeed 配置 ---
DS_CONFIG_PATH="../ds_config_stage2.json"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
MASTER_PORT=$(shuf -i 10000-65000 -n 1)

# --- 启动命令 (已简化) ---
deepspeed --include=localhost:$(seq -s, 0 $((GPU_COUNT-1))) --master_port=${MASTER_PORT} \
  ../train_stage2_vqa_sft.py \
  --output_dir=${OUTPUT_DIR} \
  --num_train_epochs=${EPOCHS} \
  --learning_rate=${LEARNING_RATE} \
  --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
  --gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
  --save_steps=${SAVE_STEPS} \
  --logging_steps=10 \
  --bf16 True \
  --save_strategy "steps" \
  --save_total_limit 3 \
  --report_to "tensorboard" \
  --deepspeed ${DS_CONFIG_PATH}

echo "✅ Stage 2 SFT (DeepSpeed) finished. LoRA adapters saved in ${OUTPUT_DIR}"