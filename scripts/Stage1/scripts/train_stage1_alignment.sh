#!/bin/bash
# FILA Stage 1 Alignment Training Launch Script

set -euo pipefail

# --- Get the absolute path of the directory where the script is located ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TRAIN_SCRIPT="${SCRIPT_DIR}/../train_stage1_alignment.py" # The python script is one level up

# --- Color Definitions ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- Configuration (ADJUST THESE PATHS) ---
LLM_PATH="/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
VISION_TOWER_PATH="/root/autodl-tmp/MedPLIB_FILA/models/clip-vit-large-patch14-336"
DATA_PATH="/root/autodl-tmp/MedPLIB_FILA/datasets/final_alignment_data_fixed.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/MedPLIB_FILA/datasets/pretrain_images/"
OUTPUT_DIR="./runs/fila-stage1-$(date +%Y%m%d_%H%M%S)"

# --- Training Hyperparameters ---
EPOCHS=1
LEARNING_RATE="1e-3"
MAX_STEPS=20000
BATCH_SIZE_PER_GPU=2
GRAD_ACCUM_STEPS=32
SAVE_STEPS=1000

# --- System Setup ---
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
DEEPSPEED_CMD="deepspeed --include=localhost:$(seq -s, 0 $((GPU_COUNT-1)))"
MASTER_PORT=$(shuf -i 10000-65000 -n 1)

echo -e "${BLUE}🚀 Starting FILA Stage 1 Training...${NC}"
echo -e "   Output Directory: ${YELLOW}${OUTPUT_DIR}${NC}"

mkdir -p "$OUTPUT_DIR"

# --- Launch Training ---
${DEEPSPEED_CMD} --master_port=${MASTER_PORT} ${TRAIN_SCRIPT} \
  --version="${LLM_PATH}" \
  --vision_tower="${VISION_TOWER_PATH}" \
  --data_path="${DATA_PATH}" \
  --image_folder="${IMAGE_FOLDER}" \
  --output_dir="${OUTPUT_DIR}" \
  --epochs=${EPOCHS} \
  --lr=${LEARNING_RATE} \
  --batch_size=${BATCH_SIZE_PER_GPU} \
  --grad_accumulation_steps=${GRAD_ACCUM_STEPS} \
  --max_steps=${MAX_STEPS} \
  --save_steps=${SAVE_STEPS} \
  --gradient_checkpointing 2>&1 | tee "${OUTPUT_DIR}/training.log"

echo -e "${GREEN}🎉 Training complete! Checkpoints are in ${OUTPUT_DIR}${NC}"