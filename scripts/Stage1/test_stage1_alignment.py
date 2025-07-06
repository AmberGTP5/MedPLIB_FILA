# file: run_inference_fixed.py
#
# 最终解决方案: 使用"猴子补丁"技术在不修改模型文件的前提下修复推理错误。
# 它会在运行时动态替换模型的 forward 方法，使其能够兼容推理流程。

import os
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoTokenizer, logging as transformers_logging
from functools import wraps

# 将项目根目录添加到系统路径，以便导入自定义模块
# 确保这个路径是正确的，指向您项目的根目录
# 例如，如果此脚本在 a/b/c/下，而模型代码在 a/model/下，则路径应为 Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 

from model.fila_lisa import FILAForCausalLM
from utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from model.medplib.model.language_model.medplib_llama import LlamaForCausalLM

# 抑制不必要的警告，使输出更整洁
transformers_logging.set_verbosity_error()

# ==============================================================================
# --- ⚙️ 配置区 (HARDCODED CONFIGURATION) ---
# --- 请在此处填入您的实际路径 ---
# ==============================================================================
BASE_LLM_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
VISION_TOWER_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/clip-vit-large-patch14-336"
STAGE1_WEIGHTS_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/scripts/runs/fila-stage1-20250613_172540/checkpoint-500/stage1_projector_cvfm.bin"
IMAGE_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/test/pexels-photo-32477995.jpeg"
PROMPT = "Describe this image in detail."
DEVICE = "cuda"
# ==============================================================================
# --- 脚本主逻辑 (大部分无需修改) ---
# ==============================================================================

def monkey_patch_forward(model):
    """
    这是一个猴子补丁函数。它会动态替换原始模型的 forward 方法。
    """
    # 1. 保存对原始 forward 方法的引用
    original_forward = model.forward

    # 2. 定义一个新的、更完善的 forward 方法
    @wraps(original_forward)
    def patched_forward(**kwargs):
        """
        推理阶段已在脚本外完成图像特征拼接，之后只需纯文本语言模型。
        先删除 LlamaForCausalLM 不认识的多模态参数，再调用其 forward。
        """
        # ⚠️ 清理多模态相关残余参数，保持签名兼容
        for invalid in (
            "images", "images_vit", "images_convnext",
            "region_masks", "valid_region_masks_bool"
        ):
            kwargs.pop(invalid, None)
        return LlamaForCausalLM.forward(model, **kwargs)

    # 3. 用我们新定义的 forward 方法替换掉模型实例的原始方法
    model.forward = patched_forward
    print("✅ Model's forward method has been successfully monkey-patched for inference.")
    return model

# load_model, prepare_inputs, 和 run_inference 函数与上一版修复中的代码完全相同
def load_model(base_llm_path, vision_tower_path, stage1_weights_path, device):
    print("🚀 正在初始化模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(base_llm_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(base_llm_path)
    config.mm_vision_tower = vision_tower_path
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_hidden_size = 1024
    config.max_sample_point = 4096
    model = FILAForCausalLM.from_pretrained(
        base_llm_path, config=config, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
    ).to(device)
    if tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]}) > 0:
        model.resize_token_embeddings(len(tokenizer))
    print(f"🔧 正在从以下路径加载训练好的权重: {stage1_weights_path}")
    if not os.path.exists(stage1_weights_path):
        print(f"❌ 错误: 权重文件不存在于 '{stage1_weights_path}'。请检查路径。")
        sys.exit(1)
    stage1_weights = torch.load(stage1_weights_path, map_location='cpu')
    model.load_state_dict(stage1_weights, strict=False)
    print("✅ 权重加载成功。")
    model.eval()
    return model, tokenizer

def prepare_inputs(image_path, prompt, tokenizer, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 错误: 图像文件不存在于 '{image_path}'。请检查路径。")
        sys.exit(1)
    transform_vit = transforms.Compose([
        transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    transform_convnext = transforms.Compose([
        transforms.Resize((768, 768), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images_vit = transform_vit(image).unsqueeze(0).to(torch.bfloat16).to(device)
    images_convnext = transform_convnext(image).unsqueeze(0).to(torch.bfloat16).to(device)
    full_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    return {"images_vit": images_vit, "images_convnext": images_convnext, "input_ids": input_ids}

def run_inference(model, tokenizer, inputs, prompt):
    print("\n💬 正在生成回应...")
    print("-------------------------")
    print(f"提示 (PROMPT): {prompt}")
    input_ids = inputs['input_ids']
    images_vit = inputs['images_vit']
    images_convnext = inputs['images_convnext']
    image_features = model.encode_images(images_vit, images_convnext)
    input_embeds = model.get_model().embed_tokens(input_ids)
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    image_token_indices = torch.where(input_ids[0] == image_token_id)[0]
    if len(image_token_indices) == 0:
        print("❌ 错误: 在 prompt 中没有找到 <image> 标记。")
        return
    image_token_start_index = image_token_indices[0]
    final_inputs_embeds = torch.cat([
        input_embeds[:, :image_token_start_index],
        image_features,
        input_embeds[:, image_token_start_index + 1:]
    ], dim=1)
    attention_mask = torch.ones(final_inputs_embeds.shape[:2], dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        output_ids = model.generate(
            inputs_embeds=final_inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            use_cache=True
        )
    decoded_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    response = decoded_output.split(prompt)[-1].strip()
    print(f"\n模型回应 (MODEL RESPONSE):\n{response}")
    print("-------------------------")

def main():
    """主执行函数"""
    print("--- 开始执行推理脚本 (带猴子补丁修复) ---")
    model, tokenizer = load_model(BASE_LLM_PATH, VISION_TOWER_PATH, STAGE1_WEIGHTS_PATH, DEVICE)
    
    # 🔥 在这里应用猴子补丁
    model = monkey_patch_forward(model)
    
    inputs = prepare_inputs(IMAGE_PATH, PROMPT, tokenizer, DEVICE)
    run_inference(model, tokenizer, inputs, PROMPT)
    print("--- 推理结束 ---")

if __name__ == "__main__":
    main()