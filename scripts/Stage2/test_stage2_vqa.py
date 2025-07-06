# file: test_stage2_vqa.py
#
# 第二阶段VQA专家模型的完整测试脚本。
# 它会先加载基础模型和第一阶段权重，然后再加载第二阶段训练的LoRA适配器。

import os
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoTokenizer, logging as transformers_logging
from peft import PeftModel

# --- 路径设置 ---
# 确保可以正确导入我们项目中的模块
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)
STAGE1_ROOT = os.path.join(PROJECT_ROOT, "scripts", "Stage1")
sys.path.insert(0, STAGE1_ROOT)

from scripts.Stage1.model.fila_lisa import FILAForCausalLM
from scripts.Stage1.utils.utils import DEFAULT_IMAGE_TOKEN
from scripts.Stage1.model.medplib.model.language_model.medplib_llama import LlamaForCausalLM
from scripts.Stage1.model.medplib import conversation as conversation_lib

transformers_logging.set_verbosity_error()

# ==============================================================================
# --- ⚙️ 配置区 (HARDCODED CONFIGURATION) ---
# --- 请在此处填入您的实际路径和要测试的内容 ---
# ==============================================================================
BASE_LLM_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
VISION_TOWER_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/clip-vit-large-patch14-336"

# 关键：第一阶段产出的权重文件路径
STAGE1_WEIGHTS_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/scripts/runs/fila-stage1-20250614_000152/checkpoint-44000/stage1_projector_cvfm.bin"

# ❗️❗️❗️【新增】第二阶段训练产出的LoRA适配器文件夹路径❗️❗️❗️
# 这个路径应该指向包含 adapter_model.bin 和 adapter_config.json 的文件夹
STAGE2_LORA_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage2/runs/stage2-vqa-deepspeed-20250621_134100"

# 您想要测试的图片
IMAGE_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage2/test/ct_00--MSD_Liver--liver_83--y_0343.png"

# 您想要提出的问题
PROMPT = "What major organs are visible in this CT scan?"

DEVICE = "cuda"
# ==============================================================================
# --- 脚本主逻辑 ---
# ==============================================================================

# 图像处理器：封装了第一阶段使用的图像变换，确保一致性
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

def load_sft_model(base_llm_path, vision_tower_path, stage1_weights_path, stage2_lora_path, device):
    """
    加载完整的、经过SFT的VQA专家模型。
    流程: 基础模型 -> 加载Stage1权重 -> 加载Stage2 LoRA适配器
    """
    print("🚀 Initializing base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_llm_path, use_fast=False, model_max_length=2048)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(base_llm_path)
    config.mm_vision_tower = vision_tower_path
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_hidden_size = 1024
    config.max_sample_point = 4096
    
    # 1. 加载基础模型
    model = FILAForCausalLM.from_pretrained(
        base_llm_path, config=config, torch_dtype=torch.bfloat16
    ).to(device)
    
    if tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]}) > 0:
        model.resize_token_embeddings(len(tokenizer))

    # 2. 加载第一阶段权重
    print(f"🔧 Loading Stage 1 weights from: {stage1_weights_path}")
    if os.path.exists(stage1_weights_path):
        stage1_weights = torch.load(stage1_weights_path, map_location='cpu')
        model.load_state_dict(stage1_weights, strict=False)
        print("✅ Stage 1 weights loaded successfully.")
    else:
        print(f"⚠️ Warning: Stage 1 weights not found at {stage1_weights_path}. Skipping.")

    # 3. 【核心】加载并应用第二阶段的LoRA适配器
    print(f"✨ Loading Stage 2 LoRA adapter from: {stage2_lora_path}")
    model = PeftModel.from_pretrained(model, stage2_lora_path)
    print("✅ Stage 2 LoRA adapter loaded successfully.")
    
    # 为了更快的推理速度，可以将LoRA权重合并到基础模型中
    print("Merging LoRA weights for faster inference...")
    model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer

def prepare_vqa_inputs(image_path, prompt, tokenizer, device):
    """为VQA任务准备输入，使用与训练时一致的对话模板。"""
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'.")
        sys.exit(1)
    
    # 使用与训练时相同的图像处理器
    image_processor = FILAImageProcessor()
    images_vit, images_convnext = image_processor(image)
    images_vit = images_vit.unsqueeze(0).to(torch.bfloat16).to(device)
    images_convnext = images_convnext.unsqueeze(0).to(torch.bfloat16).to(device)
    
    # 对话模板处理
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    question = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None) # 答案部分留空，让模型生成
    
    full_prompt_string = conv.get_prompt()
    input_ids = tokenizer(full_prompt_string, return_tensors="pt").input_ids.to(device)
    
    return {"images_vit": images_vit, "images_convnext": images_convnext, "input_ids": input_ids}


def run_manual_generation(model, tokenizer, inputs, prompt):
    """
    手动实现的、可靠的文本生成循环。
    """
    print("\n💬 Performing manual generation...")
    print("-----------------------------------")
    print(f"PROMPT: {prompt}")

    # --- 1. 准备第一次 forward pass 的输入 ---
    input_ids = inputs['input_ids']
    images_vit = inputs['images_vit']
    images_convnext = inputs['images_convnext']
    
    image_features = model.encode_images(images_vit, images_convnext)
    input_embeds = model.get_model().embed_tokens(input_ids)
    
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    image_token_indices = torch.where(input_ids[0] == image_token_id)[0]
    image_token_start_index = image_token_indices[0]
    
    current_embeds = torch.cat([
        input_embeds[:, :image_token_start_index],
        image_features,
        input_embeds[:, image_token_start_index + 1:]
    ], dim=1)
    
    attention_mask = torch.ones(current_embeds.shape[:2], dtype=torch.long, device=DEVICE)
    
    # --- 2. 初始化生成循环所需变量 ---
    max_new_tokens = 512
    generated_ids = []
    past_key_values = None
    
    with torch.inference_mode():
        for i in range(max_new_tokens):
            # --- 3. 执行前向传播 ---
            if i == 0:
                outputs = LlamaForCausalLM.forward(model, inputs_embeds=current_embeds, attention_mask=attention_mask, use_cache=True)
            else:
                outputs = LlamaForCausalLM.forward(model, input_ids=next_token, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=True)

            # --- 4. 获取下一个词元 ---
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            
            next_token_logits = logits[:, -1, :]
            # 这里使用贪心解码 (greedy decoding)
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # --- 5. 保存并检查是否结束 ---
            token_id = next_token.item()
            
            # 检查是否为结束符
            if token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id) or (hasattr(tokenizer, 'eot_id') and token_id == tokenizer.eot_id):
                break
                
            generated_ids.append(token_id)
            
            # --- 6. 准备下一次循环的输入 ---
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=DEVICE)], dim=1)

    # --- 7. 解码并打印结果 ---
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    clean_response = response.strip()
    print(f"\nMODEL RESPONSE:\n{clean_response}")
    print("-----------------------------------")


def main():
    """主执行函数"""
    print("--- Starting Stage 2 VQA Model Test ---")
    model, tokenizer = load_sft_model(BASE_LLM_PATH, VISION_TOWER_PATH, STAGE1_WEIGHTS_PATH, STAGE2_LORA_PATH, DEVICE)
    inputs = prepare_vqa_inputs(IMAGE_PATH, PROMPT, tokenizer, DEVICE)
    run_manual_generation(model, tokenizer, inputs, PROMPT)
    print("--- Inference finished ---")

if __name__ == "__main__":
    main()