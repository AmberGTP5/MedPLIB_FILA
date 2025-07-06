# file: test_dataset_full_scan.py (Corrected Version)
#
# 一个用于对整个GroundingDataset进行全面健康检查的脚本
# 使用经过验证的、正确的路径配置

import os
import sys
from pathlib import Path
from torchvision import transforms
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# --- 路径设置 ---
# 这个设置是为了让此脚本能找到同级目录下的 train_stage3_grounding_sft.py
# 并从中导入 GroundingDataset 和 FILAImageProcessor
# [保持不变]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 为了能导入正确的GroundingDataset，需要将Stage3目录加入sys.path
# train_stage3_grounding_sft.py就在当前目录，所以我们直接从它导入
sys.path.insert(0, os.path.dirname(SCRIPT_DIR)) # 添加 'scripts' 目录
from Stage3.train_stage3_grounding_sft import GroundingDataset, FILAImageProcessor

def full_dataset_scan():
    """
    对整个数据集进行一次完整的扫描，找出所有处理失败的样本。
    """
    print("--- [Dataset Full Scan] Starting a full health check of the dataset... ---")

    # --- 1. [核心修正] 使用与您验证通过的test_dataset.py完全相同的路径配置 ---
    DATA_PATH = "/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_3/subset_MeCoVQA_Grounding_CLEAN.json"
    DATA_BASE_DIR = "/root/autodl-tmp/MedPLIB_FILA/datasets/datasets_subset_3/"
    BASE_LLM_PATH = "/root/autodl-tmp/MedPLIB_FILA/models/Llama-3.1-8B-Instruct"

    print(f"  - Using Annotation File: {DATA_PATH}")
    print(f"  - Using Base Directory for Images/Masks: {DATA_BASE_DIR}")

    # --- 2. 初始化 Tokenizer 和 Image Processor ---
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_LLM_PATH,
        use_fast=False,
        local_files_only=True,
        add_bos_token=False,
        add_eos_token=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<SEG>"], special_tokens=True)
    image_processor = FILAImageProcessor()
    print("✅ Tokenizer and Image Processor initialized.")

    # --- 3. 创建数据集实例 ---
    # 我们需要在GroundingDataset的__getitem__中使用上次修改的、能暴露真实错误的版本
    dataset = GroundingDataset(
        data_path=DATA_PATH,
        data_base_dir=DATA_BASE_DIR,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    print(f"✅ Dataset instance created. Total samples to check: {len(dataset)}")

    # --- 4. 遍历检查所有样本 ---
    bad_samples = []
    for i in tqdm(range(len(dataset)), desc="Scanning samples"):
        try:
            # 尝试获取并处理样本
            sample = dataset[i]
            if sample is None:
                # 记录由显式 `return None` 导致的坏样本
                bad_samples.append({'index': i, 'error': 'Dataset class explicitly returned None (likely no <mask> tag).'})
        except Exception as e:
            # 记录由其他异常（如FileNotFound）导致的坏样本
            bad_samples.append({'index': i, 'error': f"Exception: {type(e).__name__} - {e}"})

    # --- 5. 打印最终的“体检报告” ---
    print("\n--- [Full Scan Report] ---")
    if not bad_samples:
        print("🎉 Congratulations! All 12380 samples in the dataset were processed successfully!")
    else:
        print(f"🚨 Found {len(bad_samples)} problematic samples out of {len(dataset)}.")
        print("--- List of Bad Samples (showing first 20) ---")
        for bad_sample in bad_samples[:20]: # 只打印前20个，避免刷屏
            print(f"  - Index: {bad_sample['index']}, Error: {bad_sample['error']}")
        if len(bad_samples) > 20:
            print(f"  ... and {len(bad_samples) - 20} more.")
        print("\n[ACTION REQUIRED] Please review the list above to clean your dataset.")

    print("--- Scan Finished ---")

if __name__ == "__main__":
    full_dataset_scan()