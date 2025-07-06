# file: run_generation_final.py
#
# 最终版：一个完整、干净、可用的推理脚本。
# 它通过手动实现生成循环，彻底绕开了与 model.generate() 的兼容性问题。

import os
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoTokenizer, logging as transformers_logging

# 确保这个路径是正确的
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 

from model.fila_lisa import FILAForCausalLM
from utils.utils import DEFAULT_IMAGE_TOKEN
from model.medplib.model.language_model.medplib_llama import LlamaForCausalLM

transformers_logging.set_verbosity_error()

# ==============================================================================
# --- ⚙️ 配置区 (HARDCODED CONFIGURATION) ---
# ==============================================================================
BASE_LLM_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
VISION_TOWER_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/clip-vit-large-patch14-336"
STAGE1_WEIGHTS_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/scripts/runs/fila-stage1-20250614_000152/checkpoint-44000/stage1_projector_cvfm.bin"
IMAGE_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/test/pexels-photo-32477995.jpeg"
PROMPT = "Describe this image in detail."
DEVICE = "cuda"
# ==============================================================================
# --- 脚本主逻辑 ---
# ==============================================================================

def load_model(base_llm_path, vision_tower_path, stage1_weights_path, device):
    print("🚀 Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_llm_path, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(base_llm_path)
    config.mm_vision_tower = vision_tower_path; config.mm_vision_select_layer = -2; config.mm_vision_select_feature = "patch"; config.mm_hidden_size = 1024; config.max_sample_point = 4096
    model = FILAForCausalLM.from_pretrained(
        base_llm_path, config=config, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
    ).to(device)
    if tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]}) > 0:
        model.resize_token_embeddings(len(tokenizer))
    print(f"🔧 Loading trained weights from: {stage1_weights_path}")
    if not os.path.exists(stage1_weights_path):
        print(f"❌ Error: Weight file not found at '{stage1_weights_path}'."); sys.exit(1)
    stage1_weights = torch.load(stage1_weights_path, map_location='cpu')
    model.load_state_dict(stage1_weights, strict=False)
    print("✅ Weights loaded successfully.")
    model.eval()
    return model, tokenizer

def prepare_inputs(image_path, prompt, tokenizer, device):
    try: image = Image.open(image_path).convert("RGB")
    except FileNotFoundError: print(f"❌ Error: Image file not found at '{image_path}'."); sys.exit(1)
    transform_vit = transforms.Compose([transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    transform_convnext = transforms.Compose([transforms.Resize((768, 768), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    images_vit = transform_vit(image).unsqueeze(0).to(torch.bfloat16).to(device)
    images_convnext = transform_convnext(image).unsqueeze(0).to(torch.bfloat16).to(device)
    full_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
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
            # 第一次使用 embeds，后续使用 input_ids
            if i == 0:
                outputs = LlamaForCausalLM.forward(model, inputs_embeds=current_embeds, attention_mask=attention_mask, use_cache=True)
            else:
                outputs = LlamaForCausalLM.forward(model, input_ids=next_token, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=True)

            # --- 4. 获取下一个词元 ---
            logits = outputs.logits
            past_key_values = outputs.past_key_values # 保存缓存以备下次使用
            
            next_token_logits = logits[:, -1, :]
            # 这里使用贪心解码 (greedy decoding)，也可以换成采样
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # --- 5. 保存并检查是否结束 ---
            token_id = next_token.item()
            generated_ids.append(token_id)
            
            if token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                break
            
            # --- 6. 准备下一次循环的输入 ---
            # 扩展 attention_mask 以包含新生成的词元
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=DEVICE)], dim=1)

    # --- 7. 解码并打印结果 ---
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nMODEL RESPONSE:\n{response}")
    print("-----------------------------------")


def main():
    """主执行函数"""
    print("--- Starting FINAL inference script ---")
    model, tokenizer = load_model(BASE_LLM_PATH, VISION_TOWER_PATH, STAGE1_WEIGHTS_PATH, DEVICE)
    inputs = prepare_inputs(IMAGE_PATH, PROMPT, tokenizer, DEVICE)
    run_manual_generation(model, tokenizer, inputs, PROMPT)
    print("--- Inference finished ---")

if __name__ == "__main__":
    main()