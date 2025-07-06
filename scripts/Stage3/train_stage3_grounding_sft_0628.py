# file: scripts/Stage3/train_stage3_grounding_sft.py
#
# 最终的、统一的、自给自足的第三阶段训练脚本
# 集成了所有依赖类和经过验证的初始化逻辑

import argparse
import os
import sys
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import transformers
from transformers import (AutoConfig, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torchvision import transforms
from transformers import default_data_collator

# --- 路径设置 ---
# 将项目根目录添加到搜索路径，以导入Stage1的模型
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from datasets.grounding_dataset import GroundingDataset # 导入修改后的数据集类
from model.fila_lisa_for_stage3 import FILAForCausalLM as FILABaseModel
# 导入MedPLIB中的SAM模型构造器，假设它在我们的Stage3/model目录下
from model.segment_anything_med2d import build_sam_vit_b

# ==============================================================================
# --- 第1部分: 所有必需的类定义 ---
# ==============================================================================

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

# --- 模型定义 ---
def dice_loss(inputs, targets, eps=1e-6):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    intersection = (inputs * targets).sum(1)
    union = inputs.sum(1) + targets.sum(1)
    dice_score = (2. * intersection + eps) / (union + eps)
    return 1 - dice_score.mean()

class UniMedVLMForGrounding(FILABaseModel):
    def __init__(self, config, sam_pretrained_path=None, seg_token_idx=0):
        super().__init__(config)
        
        self.seg_token_idx = seg_token_idx
        
        print("Initializing SAM-Med2D sub-module...")
        sam_model = build_sam_vit_b(sam_checkpoint=sam_pretrained_path)
        self.visual_model = sam_model # 集成整个SAM模型
        
        llm_hidden_size = config.hidden_size
        sam_feature_dim = 256
        self.text_hidden_fcs = nn.Linear(llm_hidden_size, sam_feature_dim)
        
    def forward(self, images_vit=None, images_convnext=None, sam_images=None, input_ids=None, labels=None, gt_masks=None, **kwargs):
        if self.training: # 只在训练时打印，避免推理时也输出
            print("\n" + "="*20 + " SHAPE CHECK 5 (Inside Forward Call) " + "="*20)
            print(f"Shape of pos_embed right before use: {self.visual_model.image_encoder.pos_embed.shape}")
            print("="*70 + "\n")
        # ---------------------------

        outputs = super().forward(
            images_vit=images_vit, images_convnext=images_convnext,
            input_ids=input_ids, labels=labels,
            output_hidden_states=True, return_dict=True
        )
        
        lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)
        
        seg_loss = torch.tensor(0.0, device=input_ids.device)

        if gt_masks is not None:
            last_hidden_state = outputs.hidden_states[-1]
            seg_token_mask = (input_ids == self.seg_token_idx)
            if torch.sum(seg_token_mask) > 0:
                seg_token_features = last_hidden_state[seg_token_mask]
                prompt_embeddings = self.text_hidden_fcs(seg_token_features)
                with torch.no_grad():
                    # [最终核心修正] 必须将为SAM准备的sam_images传入SAM自己的编码器！
                    image_embeddings = self.visual_model.image_encoder(sam_images)

                # [核心修正] 修正对CLIP视觉模型的调用方式
                # # 1. 首先拿到 CLIPVisionModel 对象
                # clip_vision_model = self.get_model().vision_tower.vision_tower.vision_tower
                # # 2. 直接调用它来编码图像，并获取其 last_hidden_state
                # image_embeddings = self.visual_model.image_encoder(sam_images)

                low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=prompt_embeddings.unsqueeze(1),
                    dense_prompt_embeddings=self.visual_model.prompt_encoder.get_dense_pe(),
                    multimask_output=False,
                )
                
                predicted_mask_upsampled = F.interpolate(
                    low_res_masks, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False
                )
                
                bce = F.binary_cross_entropy_with_logits(predicted_mask_upsampled.squeeze(1), gt_masks)
                dice = dice_loss(predicted_mask_upsampled.squeeze(1), gt_masks)
                seg_loss = bce + dice

        total_loss = lm_loss + seg_loss 
        return CausalLMOutputWithPast(loss=total_loss, logits=outputs.logits)

# --- 主训练函数 ---
def main():
    # --- ⚙️ 硬编码路径配置区 ---
    # ==============================================================================
    BASE_LLM_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
    VISION_TOWER_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/clip-vit-large-patch14-336/"
    STAGE1_WEIGHTS_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/scripts/runs/fila-stage1-20250614_000152/checkpoint-44000/stage1_projector_cvfm.bin"
    SAM_PRETRAINED_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/SA-Med2D/sam-med2d_b.pth"
    DATA_PATH = "/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_3/subset_MeCoVQA_Grounding_CLEAN.json"
    DATA_BASE_DIR = "/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_3/"
    # ==============================================================================
    
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Stage 3: Grounding Expert SFT")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sft_modules", type=str, default="mask_decoder,text_hidden_fcs")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: overrides num_train_epochs.")
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

    # --- 初始化模型和Tokenizer ---
    # print("🚀 Initializing model and tokenizer for Stage 3 Grounding SFT...")
    # tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_PATH, use_fast=False, local_files_only=True)
    # if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_tokens(["<SEG>"], special_tokens=True)
    # seg_token_idx = tokenizer.convert_tokens_to_ids("<SEG>")

    # [核心修正] 采用与test_dataset.py中验证通过的、完全相同的初始化方式
    print("🚀 Initializing model and tokenizer for Stage 3 Grounding SFT...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_LLM_PATH,
        use_fast=False,         # ❗️确保为False
        local_files_only=True,  # 确保离线
        add_bos_token=False,    # ❗️确保为False
        add_eos_token=False     # ❗️确保为False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_tokens(["<SEG>"], special_tokens=True)
    # seg_token_idx 的获取也移到这里
    seg_token_idx = tokenizer.convert_tokens_to_ids("<SEG>")
    
    print("✅ Tokenizer initialized with correct settings.")

    # Register the <image> token
    special_tokens_dict = {'additional_special_tokens': ['<image>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
        
    # --- [最终解决方案：手动、分步、可控地构建模型] ---
    print("Step 1: Creating a complete, custom 'config' object...")
    config = AutoConfig.from_pretrained(BASE_LLM_PATH, local_files_only=True)
    config.mm_vision_tower = VISION_TOWER_PATH
    config.vision_tower = VISION_TOWER_PATH
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_hidden_size = 1024
    config.train_mask_decoder = True
    config.out_dim = 256
    config.max_sample_point = 4096 # <-- 补上这个缺失的属性
    print("✅ Config object created and fully populated.")
    
    print("Step 2: Instantiating our custom UniMedVLMForGrounding model framework...")
    model = UniMedVLMForGrounding(config, sam_pretrained_path=SAM_PRETRAINED_PATH, seg_token_idx=seg_token_idx)
    
    # 监控点 1: 模型刚刚初始化后
    print("\n" + "="*20 + " SHAPE CHECK 1 (After Model Init) " + "="*20)
    print(f"Shape of pos_embed: {model.visual_model.image_encoder.pos_embed.shape}")
    print("="*68 + "\n")

    # 修复点 A: 动态调整SAM的位置编码
    print("🧊 Dynamically adjusting SAM's positional embedding for 1024x1024 input...")
    target_embedding_size = (64, 64)
    original_pos_embed = model.visual_model.image_encoder.pos_embed
    if original_pos_embed.shape[1:3] != target_embedding_size:
        original_pos_embed_transposed = original_pos_embed.permute(0, 3, 1, 2)
        new_pos_embed_transposed = F.interpolate(
            original_pos_embed_transposed, size=target_embedding_size, mode='bilinear', align_corners=False
        )
        new_pos_embed = new_pos_embed_transposed.permute(0, 2, 3, 1)
        model.visual_model.image_encoder.pos_embed = nn.Parameter(new_pos_embed)
        print("✅ SAM's positional embedding resized successfully.")
    else:
        print("✅ SAM's positional embedding already has the correct size.")
    
    # 监控点 2: 手动修正 pos_embed 之后
    print("\n" + "="*20 + " SHAPE CHECK 2 (After pos_embed Fix) " + "="*20)
    print(f"Shape of pos_embed: {model.visual_model.image_encoder.pos_embed.shape}")
    print("="*68 + "\n")

    # 修复点 B: 动态调整SAM的Mask Decoder
    print("🧊 Dynamically adjusting SAM's Mask Decoder output upsampling...")
    mask_decoder = model.visual_model.mask_decoder
    mask_decoder.output_upscaling = nn.Sequential(
        nn.ConvTranspose2d(256, 256 // 4, kernel_size=2, stride=2),
        nn.LayerNorm(256 // 4),
        nn.GELU(),
        nn.ConvTranspose2d(256 // 4, 256 // 8, kernel_size=2, stride=2),
        nn.GELU(),
    )
    mask_decoder.output_upscaling.to(device=model.device, dtype=model.dtype)
    print("✅ SAM's Mask Decoder upscaling layer re-initialized.")

    # 修复点 C: 强制重新初始化mm_projector
    print("🧊 Re-initializing mm_projector to ensure correct dimensions [1024 -> 4096]...")
    mm_hidden_size = model.config.mm_hidden_size
    hidden_size = model.config.hidden_size
    model.model.mm_projector = nn.Sequential(
        nn.Linear(mm_hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size)
    )
    model.model.mm_projector.to(device=model.device, dtype=model.dtype)
    print(f"✅ mm_projector re-initialized successfully.")

    print("Step 3: Loading base LLM weights...")
    model.load_state_dict(
        transformers.LlamaForCausalLM.from_pretrained(BASE_LLM_PATH, torch_dtype=torch.bfloat16, local_files_only=True).state_dict(),
        strict=False
    )
    
    # 监控点 3: 加载完LLM权重之后
    print("\n" + "="*20 + " SHAPE CHECK 3 (After LLM Load) " + "="*20)
    print(f"Shape of pos_embed: {model.visual_model.image_encoder.pos_embed.shape}")
    print("="*68 + "\n")

    print(f"Step 4: Loading Stage 1 weights from: {STAGE1_WEIGHTS_PATH}")
    model.load_state_dict(torch.load(STAGE1_WEIGHTS_PATH, map_location='cpu'), strict=False)

    # 监控点 4: 加载完Stage1权重之后
    print("\n" + "="*20 + " SHAPE CHECK 4 (After Stage1 Load) " + "="*20)
    print(f"Shape of pos_embed: {model.visual_model.image_encoder.pos_embed.shape}")
    print("="*68 + "\n")
    
    # 您已经注释掉了 resize_token_embeddings，这是正确的
    # model.resize_token_embeddings(len(tokenizer))
    # 为确保万无一失，我们使用手动版本
    print("🧊 Manually resizing token embeddings and output layer for new tokens...")
    new_num_tokens = len(tokenizer)
    input_embeddings = model.get_model().embed_tokens
    old_num_tokens, old_embedding_dim = input_embeddings.weight.shape
    if old_num_tokens != new_num_tokens:
        new_input_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim, device=model.device, dtype=model.dtype)
        new_input_embeddings.weight.data[:old_num_tokens, :] = input_embeddings.weight.data.clone()
        avg_embedding = input_embeddings.weight.data.mean(dim=0)
        new_input_embeddings.weight.data[old_num_tokens:, :] = avg_embedding
        model.get_model().embed_tokens = new_input_embeddings
    output_embeddings = model.lm_head
    if output_embeddings.weight.shape[0] != new_num_tokens:
        new_lm_head = nn.Linear(output_embeddings.in_features, new_num_tokens, bias=False, device=model.device, dtype=model.dtype)
        new_lm_head.weight.data[:old_num_tokens, :] = output_embeddings.weight.data.clone()
        if 'avg_embedding' in locals():
            new_lm_head.weight.data[old_num_tokens:, :] = avg_embedding
        model.lm_head = new_lm_head
    model.config.vocab_size = new_num_tokens
    print(f"✅ Final embeddings and lm_head resized from {old_num_tokens} to {new_num_tokens}.")

    print("✅ All components assembled and weights loaded successfully.")
    
    # --- 精确冻结/解冻参数 ---
    print("🧊 Freezing parameters for Stage 3 SFT...")
    for param in model.parameters(): param.requires_grad = False
    
    trainable_modules = args.sft_modules.split(',')
    for name, param in model.named_parameters():
        for tm in trainable_modules:
            if tm in name:
                param.requires_grad = True
                break
    
    # --- [核心修正] ---
    # 手动计算并打印可训练参数
    print("Calculating trainable parameters...")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Trainable params: {trainable_params:,}")
    print(f"  - All params: {total_params:,}")
    print(f"  - Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    # --- 准备数据集 ---
    image_processor = FILAImageProcessor()
    train_dataset = GroundingDataset(
        data_path=DATA_PATH, data_base_dir=DATA_BASE_DIR,
        tokenizer=tokenizer, image_processor=image_processor
    )

    # --- [新增调试信息] ---
    print("\n" + "="*30)
    print("--- [DEBUG] COLLATOR PREPARATION ---")
    
    # 步骤 1: 创建 DataCollatorForSeq2Seq 实例
    print("[DEBUG] Step 1: Creating DataCollatorForSeq2Seq instance...")
    data_collator = default_data_collator
    # 步骤 2: 打印即将传入Trainer的collator的类型
    print(f"[DEBUG] Step 2: The object being passed to Trainer as 'data_collator' is of type: {type(data_collator)}")
    print("="*30 + "\n")
    # --- 调试信息结束 ---


    # --- 设置并启动训练 ---
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
        # 如果您的 .sh 脚本中也传入了 max_steps, 这里也需要加上
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )
    print("🔥 Starting SFT...")

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, data_collator=data_collator
    )

    # 然后再调用 .train() 方法
    trainer.train()

    # --- 保存模型 ---
    state_dict_to_save = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    torch.save(state_dict_to_save, os.path.join(args.output_dir, "stage3_grounding_weights.bin"))
    print(f"🎉 Stage 3 SFT finished. Grounding expert weights saved in {args.output_dir}")

if __name__ == "__main__":
    main()