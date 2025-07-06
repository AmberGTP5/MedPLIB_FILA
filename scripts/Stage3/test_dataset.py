# file: scripts/Stage3/test_dataset.py
#
# 一个独立的、用于验证GroundingDataset是否正常工作的小脚本

import os
import sys
from pathlib import Path
from torchvision import transforms
import torch
from transformers import AutoTokenizer

# --- 路径设置 ---
# 这个设置是为了让此脚本能找到同级目录下的 datasets/ 和 model/ 文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from datasets.grounding_dataset import GroundingDataset

# --- 图像处理器 (直接从Stage2脚本复制过来，确保一致性) ---
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

def main():
    # --- [配置您的路径] (与上一版完全相同) ---
    TOKENIZER_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"
    DATA_PATH = "/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_3/subset_MeCoVQA_Grounding_CLEAN.json"
    DATA_BASE_DIR = "/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_3/"

    print("--- [Dataset Test] Starting test ---")

    # 1. 初始化组件
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        use_fast=False,
        local_files_only=True,
        add_bos_token=False, # ❗️关键：禁止自动添加句子起始符
        add_eos_token=False  # ❗️关键：禁止自动添加句子结束符
    )
    
    # --- [核心修正] ---
    # 告诉测试用的tokenizer认识<SEG>这个特殊token
    tokenizer.add_tokens(["<SEG>"], special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer initialized and special token '<SEG>' added.")
    # --- 修正结束 ---

    image_processor = FILAImageProcessor()

    # 2. 创建数据集实例
    dataset = GroundingDataset(
        data_path=DATA_PATH,
        data_base_dir=DATA_BASE_DIR,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    print(f"\n✅ Dataset instance created. Total samples: {len(dataset)}")

    # 3. 尝试获取第一个样本并验证
    print("\n--- [Dataset Test] Attempting to fetch first sample (index 0)... ---")
    
    # 使用一个循环来找到第一个有效的样本，以防万一还有其他问题
    first_sample = None
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is not None:
            first_sample = sample
            print(f"--- Found first valid sample at index: {i} ---")
            break

    if first_sample:
        print("\n✅ Successfully fetched a valid sample!")
        print("--- Sample Content ---")
        for key, value in first_sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key: '{key}', Type: torch.Tensor, Shape: {value.shape}, Dtype: {value.dtype}")
            elif isinstance(value, list) and all(isinstance(i, torch.Tensor) for i in value):
                print(f"  - Key: '{key}', Type: List of Tensors")
                for j, t in enumerate(value):
                    print(f"    - Tensor {j}: Shape: {t.shape}, Dtype: {t.dtype}")
            else:
                print(f"  - Key: '{key}', Type: {type(value)}")
    else:
        print("\n❌ Failed to fetch any valid sample from the dataset.")

if __name__ == "__main__":
    main()