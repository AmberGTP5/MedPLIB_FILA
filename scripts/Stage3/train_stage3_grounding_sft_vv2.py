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
from transformers import DataCollatorForSeq2Seq

# --- 路径设置 ---
# 将项目根目录添加到搜索路径，以导入Stage1的模型
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from scripts.Stage1.model.fila_lisa import FILAForCausalLM as FILABaseModel
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

# --- 数据集脚本 ---
class GroundingDataset(Dataset):
    """Dataset for Stage 3: Grounding Expert SFT."""
    def __init__(self, data_path, data_base_dir, tokenizer, image_processor, model_max_length=2048):
        print(f"Initializing GroundingDataset...")
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)
        self.data_base_dir = data_base_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_max_length = model_max_length
        self.mask_transform = transforms.ToTensor()
        self.DEFAULT_IMAGE_TOKEN = "<image>"
        self.IGNORE_INDEX = -100

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        try:
            image_path_from_json = item['image']
            base_filename = os.path.basename(image_path_from_json)
            image_path = os.path.join(self.data_base_dir, "images", base_filename)
            input_image = Image.open(image_path).convert('RGB')
            images_vit, images_convnext = self.image_processor(input_image)

            answer = item['conversations'][1]['value']
            mask_path_match = re.search(r'<mask>(.*?)<\/mask>', answer)
            if not mask_path_match: return None
            
            mask_relative_path = mask_path_match.group(1)
            mask_path = os.path.join(self.data_base_dir, mask_relative_path)
            gt_mask_image = Image.open(mask_path).convert('L')
            gt_masks_tensor = self.mask_transform(gt_mask_image)
            gt_masks_tensor = (gt_masks_tensor > 0.5).float()

            instruction = item['conversations'][0]['value'].replace("<image>\n", "")
            prompt = f"USER: {self.DEFAULT_IMAGE_TOKEN}\n{instruction}\nASSISTANT: <SEG>"
            
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=self.model_max_length).input_ids[0]
            labels = torch.full_like(input_ids, self.IGNORE_INDEX)
            
            return dict(images_vit=images_vit, images_convnext=images_convnext, input_ids=input_ids, labels=labels, gt_masks=gt_masks_tensor.squeeze(0))
        except Exception: return None

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
        
    def forward(self, images_vit=None, images_convnext=None, input_ids=None, labels=None, gt_masks=None, **kwargs):
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
                    image_embeddings = self.get_model().vision_tower.vision_tower.vision_tower.image_encoder(images_vit)

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
    
    print("Step 3: Loading base LLM weights...")
    model.load_state_dict(
        transformers.LlamaForCausalLM.from_pretrained(BASE_LLM_PATH, torch_dtype=torch.bfloat16, local_files_only=True).state_dict(),
        strict=False
    )
    
    print(f"Step 4: Loading Stage 1 weights from: {STAGE1_WEIGHTS_PATH}")
    model.load_state_dict(torch.load(STAGE1_WEIGHTS_PATH, map_location='cpu'), strict=False)
    
    print("✅ All components assembled and weights loaded.")
    
    model.resize_token_embeddings(len(tokenizer))
    
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
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
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