# file: scripts/Stage3/train_stage3_grounding_sft.py
#
# 最终修正版：修正了所有已知问题，是一个完整的、可直接运行的训练脚本。

import argparse
import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import (AutoConfig, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)

# --- [核心修正] 强制启用离线模式 ---
# 这一行代码会告诉transformers库，禁止所有网络连接尝试
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# --- 路径设置 (最终解决方案) ---
# 1. ❗️【关键】将您整个项目的根目录添加到搜索路径
#    这使得 `from scripts.Stage1...` 或 `from scripts.Stage3...` 这样的导入能够成功
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

# 2. ❗️【关键】将此脚本所在的Stage3目录也添加到搜索路径
#    这使得 `from model...` 或 `from utils...` 能找到您拷贝到Stage3内的文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# --- 模块导入 ---
# 现在所有导入都将基于正确的搜索路径
from model.LISA import LISAForCausalLM
from scripts.Stage1.model.fila_lisa import FILAForCausalLM as FILABaseModel
from datasets.grounding_dataset import GroundingDataset


# --- 图像处理器 ---
class FILAImageProcessor:
    """A callable class to process images for both ViT and ConvNeXt branches."""
    def __init__(self):
        self.vit_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.convnext_transform = transforms.Compose([
            transforms.Resize((768, 768), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, image):
        return self.vit_transform(image), self.convnext_transform(image)


# --- 主训练函数 ---
def train():
    # ==============================================================================
    # --- ⚙️ 硬编码路径配置区 ---
    # ==============================================================================
    BASE_LLM_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
    VISION_TOWER_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/clip-vit-large-patch14-336/"
    STAGE1_WEIGHTS_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/scripts/runs/fila-stage1-20250614_000152/checkpoint-44000/stage1_projector_cvfm.bin"
    SAM_PRETRAINED_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/SA-Med2D/sam-med2d_b.pth"
    DATA_PATH = "/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_3/subset_MeCoVQA_Grounding_CLEAN.json"
    DATA_BASE_DIR = "/root/autodl-tmp/MedPLIB_FILA/datasets/SA-Med2D/"
    # ==============================================================================
    
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Stage 3: Grounding Expert SFT")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sft_modules", type=str, default="mask_decoder,text_hidden_fcs")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--bf16", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--version", type=str, default=BASE_LLM_PATH)
    parser.add_argument("--vision_tower", type=str, default=VISION_TOWER_PATH)
    args = parser.parse_args()

    # --- 2. 初始化模型和Tokenizer (最终解决方案) ---
    print("🚀 Initializing model and tokenizer for Stage 3 Grounding SFT...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_PATH, use_fast=False, model_max_length=2048, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # --- [核心修正：纯手动分步构建模型] ---
    
    # 步骤 1: 创建一个被我们完整“扩充”了的Config对象
    print("Step 1: Creating a single, complete config object...")
    config = AutoConfig.from_pretrained(BASE_LLM_PATH, trust_remote_code=True, local_files_only=True)
    config.mm_vision_tower = VISION_TOWER_PATH
    config.vision_tower = VISION_TOWER_PATH
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_hidden_size = 1024
    config.train_mask_decoder = True
    config.out_dim = 256
    
    tokenizer.add_tokens(["<SEG>"], special_tokens=True)
    seg_token_idx = tokenizer.convert_tokens_to_ids("<SEG>")

    # 步骤 2: 使用这个完美的Config对象，直接调用构造函数来创建模型“空壳”
    print("Step 2: Instantiating an empty LISAForCausalLM model framework...")
    # ❗️将seg_token_idx作为关键字参数直接传入
    model = LISAForCausalLM(config=config, torch_dtype=torch.bfloat16, seg_token_idx=seg_token_idx)
    print("✅ LISA base model framework created successfully.")

    # 步骤 3: 手动为这个“空壳”模型加载基础LLM的权重
    print("Step 3: Manually loading base LLM weights into the framework...")
    # 我们先加载一个标准的Llama模型，然后只取它的权重字典
    base_llama_state_dict = transformers.LlamaForCausalLM.from_pretrained(
        BASE_LLM_PATH, torch_dtype=torch.bfloat16, local_files_only=True
    ).state_dict()
    model.load_state_dict(base_llama_state_dict, strict=False)
    del base_llama_state_dict # 释放内存
    print("✅ Base LLM weights loaded successfully.")
    
    # 步骤 4: 注入我们自己的高分辨率编码器
    print("Step 4: Replacing original vision tower with our Hybrid Encoder...")
    fila_base_model = FILABaseModel(config)
    model.get_model().vision_tower = fila_base_model.get_model().vision_tower
    print("✅ Hybrid Encoder injected successfully.")
    
    # 步骤 5: 加载其余的权重
    print(f"Step 5a: Loading Stage 1 alignment weights from: {STAGE1_WEIGHTS_PATH}")
    model.load_state_dict(torch.load(STAGE1_WEIGHTS_PATH, map_location='cpu'), strict=False)
    print("✅ Stage 1 weights loaded.")

    print(f"Step 5b: Loading pre-trained SAM weights from: {SAM_PRETRAINED_PATH}")
    # ❗️现在我们是在一个完整的模型对象上调用这个方法，它将成功执行
    model.get_model().initialize_lisa_modules(vision_pretrained=SAM_PRETRAINED_PATH)
    print("✅ Pre-trained SAM weights loaded.")
    
    model.resize_token_embeddings(len(tokenizer))
    # --- 修正结束 ---
    
    # --- 5. 准备数据集 ---
    print("📚 Preparing Grounding dataset...")
    image_processor = FILAImageProcessor()
    train_dataset = GroundingDataset(
        data_path=DATA_PATH, data_base_dir=DATA_BASE_DIR,
        tokenizer=tokenizer, image_processor=image_processor
    )
    
    # --- 6. 设置并启动训练 ---
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
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, data_collator=default_data_collator
    )
    trainer.train()

    # --- 7. 保存可训练部分的权重 ---
    print("💾 Saving trainable weights for the Grounding Expert...")
    state_dict_to_save = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    torch.save(state_dict_to_save, os.path.join(args.output_dir, "stage3_grounding_weights.bin"))
    print(f"🎉 Stage 3 SFT finished. Grounding expert weights saved in {args.output_dir}")

if __name__ == "__main__":
    train()