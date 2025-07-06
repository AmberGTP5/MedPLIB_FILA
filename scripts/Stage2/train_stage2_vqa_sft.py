# file: scripts/Stage2/train_stage2_vqa_sft.py
#
# 第二阶段：VQA专家监督微调（SFT）的完整训练脚本

import argparse
import os
import sys
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from torchvision import transforms

# --- 路径设置 ---
# 将项目根目录 (MedPLIB_FILA) 添加到 sys.path，以确保可以正确导入其他模块
# 此脚本位于 MedPLIB_FILA/scripts/Stage2/
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

STAGE1_ROOT = os.path.join(PROJECT_ROOT, "scripts", "Stage1")
sys.path.insert(0, STAGE1_ROOT)

# 导入我们自己项目中的模块
from scripts.Stage1.model.fila_lisa import FILAForCausalLM
from scripts.Stage2.datasets.vqa_dataset import VQADataset

# ----------------------------------------------------------------------------------
# 图像处理器：封装了第一阶段使用的图像变换，确保一致性
# ----------------------------------------------------------------------------------
class FILAImageProcessor:
    """A callable class to process images for both ViT and ConvNeXt branches."""
    def __init__(self):
        # ViT 分支的图像变换 (来自 Stage 1)
        self.vit_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        # ConvNeXt 分支的图像变换 (来自 Stage 1)
        self.convnext_transform = transforms.Compose([
            transforms.Resize((768, 768), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        # 返回一个包含两种变换结果的元组
        return self.vit_transform(image), self.convnext_transform(image)

# ----------------------------------------------------------------------------------
# 主训练函数
# ----------------------------------------------------------------------------------
def train():
    # ==============================================================================
    # --- ⚙️ 硬编码路径配置区 ---
    # --- 请在此处填入您的实际固定路径 ---
    # ==============================================================================
    BASE_LLM_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
    VISION_TOWER_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/clip-vit-large-patch14-336/"
    STAGE1_WEIGHTS_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/scripts/runs/fila-stage1-20250614_000152/checkpoint-44000/stage1_projector_cvfm.bin"
    DATA_PATH = "/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_2/subset_MeCoVQA_Region_CLEAN.json"
    IMAGE_FOLDER = "/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_2/subset_images_region_CLEAN/"
    # ==============================================================================

    # --- 1. 解析命令行参数 (已修改，移除了路径参数) ---
    parser = argparse.ArgumentParser(description="Stage 2: VQA Expert SFT with DeepSpeed")
    
    # -- 训练超参数和输出目录 --
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--bf16", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    
    # -- DeepSpeed 兼容性参数 --
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)

    args = parser.parse_args()

    # --- 2. 初始化模型和Tokenizer ---
    print("🚀 Initializing model and tokenizer for Stage 2 SFT...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_PATH, use_fast=False, model_max_length=2048) # [修正]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(BASE_LLM_PATH, trust_remote_code=True) # [修正]
    config.mm_vision_tower = VISION_TOWER_PATH # [修正]
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_hidden_size = 1024
    config.max_sample_point = 4096

    model = FILAForCausalLM.from_pretrained(
        BASE_LLM_PATH, # [修正]
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    if tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]}) > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    model.get_model().vision_tower.vision_tower.load_model()
    model.get_model().vision_tower.vision_tower.tokenizer = tokenizer

    # --- 3. 加载第一阶段权重并应用LoRA ---
    print(f"🔧 Loading Stage 1 weights from: {STAGE1_WEIGHTS_PATH}")
    stage1_weights = torch.load(STAGE1_WEIGHTS_PATH, map_location='cpu') # [修正]
    model.load_state_dict(stage1_weights, strict=False)
    print("✅ Stage 1 weights loaded successfully.")
    
    print("🧊 Freezing original parameters and applying LoRA...")
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. 准备数据集 ---
    print("📚 Preparing VQA dataset...")
    image_processor = FILAImageProcessor()
    train_dataset = VQADataset(
        data_path=DATA_PATH, # [修正]
        image_folder=IMAGE_FOLDER, # [修正]
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    # --- 5. 设置并启动训练 ---
    print("🔥 Setting up Trainer and starting SFT...")
    
    # ❗️❗️❗️注意：这里的变量名需要与启动脚本中的命令行参数名完全对应❗️❗️❗️
    print("🔥 Setting up Trainer and starting SFT...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"🎉 Stage 2 SFT finished. Final LoRA adapter saved in {args.output_dir}")


if __name__ == "__main__":
    train()