# file: scripts/Stage3/test_model_init.py
#
# 一个独立的、用于验证UniMedVLMForGrounding模型能否被正确初始化的脚本。

import os
import sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
import transformers

# --- 路径设置 ---
# 这个设置是为了让此脚本能找到同级目录下的 model/ 和 utils/ 文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model.unimed_vlm import UniMedVLMForGrounding

def main():
    # --- [请修改] 配置您的模型和权重路径 ---
    BASE_LLM_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
    VISION_TOWER_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/clip-vit-large-patch14-336/"
    STAGE1_WEIGHTS_PATH = "/root/autodl-tmp/MedPLIB_FILA/scripts/Stage1/scripts/runs/fila-stage1-20250614_000152/checkpoint-44000/stage1_projector_cvfm.bin"
    SAM_PRETRAINED_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/SA-Med2D/sam-med2d_b.pth"
    
    print("--- [Model Init Test] Starting test ---")

    try:
        # 步骤 1: 初始化Tokenizer和Config
        print("\n[TEST STEP 1] Initializing Tokenizer and base Config...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_PATH, use_fast=False, local_files_only=True)
        config = AutoConfig.from_pretrained(BASE_LLM_PATH, local_files_only=True)
        print("✅ Base Tokenizer and Config initialized.")

        # --- [核心修正] ---
        # 步骤 2: 在一个地方，一次性地为Config对象注入所有必需的多模态属性
        print("\n[TEST STEP 2] Injecting ALL multimodal attributes into config object...")
        config.mm_vision_tower = VISION_TOWER_PATH
        config.vision_tower = VISION_TOWER_PATH
        config.mm_vision_select_layer = -2
        config.mm_vision_select_feature = "patch"
        config.mm_hidden_size = 1024
        config.train_mask_decoder = True
        config.out_dim = 256
        config.max_sample_point = 4096 # <-- [核心修正] 补上这个缺失的属性
        print("✅ All attributes injected.")

        # --- [新增的调试信息] ---
        print("\n--- [DEBUG] Verifying config object before model creation ---")
        print(f"  - Does config have 'mm_vision_tower'?        {'mm_vision_tower' in config.__dict__}")
        print(f"  - Does config have 'mm_vision_select_layer'?  {'mm_vision_select_layer' in config.__dict__}")
        print(f"  - Does config have 'mm_vision_select_feature'? {'mm_vision_select_feature' in config.__dict__}")
        print(f"  - Does config have 'mm_hidden_size'?         {'mm_hidden_size' in config.__dict__}")
        print("---------------------------------------------------------")
        # --- 调试信息结束 ---
        
        # 步骤 3: 实例化我们自己的模型，并传入这个“完美”的config
        print("\n[TEST STEP 3] Instantiating UniMedVLMForGrounding model framework...")
        # 准备kwargs
        tokenizer.add_tokens(["<SEG>"], special_tokens=True)
        model_kwargs = {"vision_pretrained": SAM_PRETRAINED_PATH, "seg_token_idx": tokenizer.convert_tokens_to_ids("<SEG>")}
        model = UniMedVLMForGrounding(config, **model_kwargs)
        print("✅ Custom model framework created successfully.")
        
        # 步骤 4: 手动为这个“空壳”模型加载基础LLM的权重
        print("\n[TEST STEP 4] Manually loading base LLM weights...")
        # 注意：这里我们加载一个标准的LlamaForCausalLM来获取state_dict，以确保兼容性
        base_llama_state_dict = transformers.LlamaForCausalLM.from_pretrained(
            BASE_LLM_PATH, torch_dtype=torch.bfloat16, local_files_only=True
        ).state_dict()
        model.load_state_dict(base_llama_state_dict, strict=False)
        del base_llama_state_dict
        print("✅ Base LLM weights loaded successfully.")
        
        # 步骤 5: 加载Stage1权重
        print(f"\n[TEST STEP 5] Loading Stage 1 weights from: {STAGE1_WEIGHTS_PATH}")
        model.load_state_dict(torch.load(STAGE1_WEIGHTS_PATH, map_location='cpu'), strict=False)
        print("✅ Stage 1 weights loaded.")
        
        # 步骤 5: 冻结与解冻参数 (已验证成功)
        print("\n[TEST STEP 5] Freezing and unfreezing parameters...")
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        trainable_modules = ["text_hidden_fcs", "visual_model.mask_decoder"]
        for name, param in model.named_parameters():
            for trainable_module in trainable_modules:
                if trainable_module in name:
                    param.requires_grad = True
                    break
        print("✅ Parameters configured for training.")

        # --- [核心修正] ---
        # 步骤 6: 手动计算并打印可训练参数，替换掉 model.print_trainable_parameters()
        print("\n[TEST STEP 6] Calculating and printing trainable parameters...")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - Trainable params: {trainable_params:,}")
        print(f"  - All params: {total_params:,}")
        print(f"  - Trainable %: {100 * trainable_params / total_params:.4f}%")
        # --- 修正结束 ---
        
        print("\n" + "="*30)
        print("✅ [SUCCESS] Model assembly and power-on test completed successfully!")
        print("="*30)

    except Exception as e:
        print(f"\n❌ [FAILURE] Test failed. Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()