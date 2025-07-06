#!/bin/bash
# Stage 3: Grounding Expert SFT Launch Script (Final Corrected Version)

set -euo pipefail
export TRANSFORMERS_OFFLINE=1

# --- 核心脚本与配置路径 ---
TRAIN_SCRIPT_PATH="../train_stage3_grounding_sft.py"
DS_CONFIG_PATH="../ds_config_stage3.json"


# --- 路径配置 (仅供参考，不再作为参数传递) ---
# 以下路径已被硬编码到 train_stage3_grounding_sft.py 中
# BASE_LLM_PATH="/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
# VISION_TOWER_PATH="/root/autodl-tmp/MedPLIB_FILA/models/clip-vit-large-patch14-336/"
# STAGE1_WEIGHTS_PATH="/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/scripts/runs/fila-stage1-20250614_000152/checkpoint-44000/stage1_projector_cvfm.bin"
# SAM_PRETRAINED_PATH="/root/autodl-tmp/MedPLIB_FILA/models/SA-Med2D/sam-med2d_b.pth"
# DATA_PATH="/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_3/subset_MeCoVQA_Grounding_CLEAN.json"
# DATA_BASE_DIR="/root/autodl-tmp/MedPLIB_FILA/datasets/SA-Med2D/"


# --- SFT 超参数 ---
LEARNING_RATE="5e-5"
EPOCHS=3
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=16


# --- DeepSpeed 系统设置 ---
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
MASTER_PORT=$(shuf -i 10000-65000 -n 1)

# [关键修改] 定义输出目录，这是唯一需要通过命令行传入的路径相关参数
OUTPUT_DIR="../runs/stage3-grounding-sft-$(date +%Y%m%d_%H%M%S)"
echo "🚀 Starting Stage 3: Grounding Expert SFT..."
echo "   Output Directory: ${OUTPUT_DIR}"


# --- 启动命令 (已简化) ---
deepspeed --include=localhost:$(seq -s, 0 $((GPU_COUNT-1))) --master_port=${MASTER_PORT} \
  ${TRAIN_SCRIPT_PATH} \
  --output_dir=${OUTPUT_DIR} \
  --num_train_epochs=${EPOCHS} \
  --learning_rate=${LEARNING_RATE} \
  --per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
  --bf16 True \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_steps 10 \
  --report_to "tensorboard" \
  --deepspeed ${DS_CONFIG_PATH} \
  --sft_modules "mask_decoder,text_hidden_fcs"
  # 移除了所有 --version, --vision_tower, --data_path 等路径参数

echo "✅ Stage 3 SFT finished. LoRA adapters saved in ${OUTPUT_DIR}"