#!/bin/bash
# Stage 3: Single-Step Training Simulation Script

set -euo pipefail
export TRANSFORMERS_OFFLINE=1

# --- 核心脚本与配置路径 ---
# ❗️我们现在调用的是我们自己的、整合好的训练脚本
TRAIN_SCRIPT_PATH="../train_stage3_grounding_sft.py" 
DS_CONFIG_PATH="../ds_config_stage3.json"

# --- 训练超参数 ---
OUTPUT_DIR="../runs/stage3-simulation-$(date +%Y%m%d_%H%M%S)"
LEARNING_RATE="5e-5"
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=16
SFT_MODULES="text_hidden_fcs,visual_model.mask_decoder"

# --- DeepSpeed 系统设置 ---
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
MASTER_PORT=$(shuf -i 10000-65000 -n 1)

echo "🚀 Starting Stage 3: Single-Step Simulation..."
echo "   Output Directory: ${OUTPUT_DIR}"

# --- 启动命令 ---
deepspeed --include=localhost:$(seq -s, 0 $((GPU_COUNT-1))) --master_port=${MASTER_PORT} \
  ${TRAIN_SCRIPT_PATH} \
  --output_dir=${OUTPUT_DIR} \
  --sft_modules="${SFT_MODULES}" \
  --learning_rate=${LEARNING_RATE} \
  --per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
  --bf16 True \
  --deepspeed=${DS_CONFIG_PATH} \
  --logging_steps 1 \
  --save_strategy "no" \
  --max_steps 1 # <-- 关键：只训练一步

echo "✅ Stage 3 Simulation finished."