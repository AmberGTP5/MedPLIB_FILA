# file: create_clean_subset.py (FINAL OPTIMIZED VERSION)
#
# 最终优化版：增加了对JSON中缺少.png后缀的路径的自动处理能力。

import json
import os
import shutil
from tqdm import tqdm

# ==============================================================================
# --- ⚙️ 配置区 (HARDCODED CONFIGURATION) ---
# --- 请在此处填入您的实际路径和参数 ---
# ==============================================================================
SOURCE_JSON_PATH = "E:/MeCoVQA/train/MeCoVQA-Region.json"
SOURCE_BASE_DIR = "E:/SAMed2Dv1" 
DEST_JSON_PATH = "D:/SoftData/VSCODE/顶会论文/MedPLIB-main/20250607/datasets_subset_2/subset_MeCoVQA_Region_CLEAN.json"
DEST_IMAGES_DIR = "D:/SoftData/VSCODE/顶会论文/MedPLIB-main/20250607/datasets_subset_2/subset_images_region_CLEAN/"
NUM_SAMPLES_TO_PROCESS = 10000 
MISSING_FILES_LOG_PATH = "D:/SoftData/VSCODE/顶会论文/MedPLIB-main/20250607/datasets_subset_2/missing_files_CLEAN.txt"
# ==============================================================================
# --- 脚本主逻辑 (已优化) ---
# ==============================================================================

def create_dataset_subset():
    print("--- Starting Dataset Subset Creation (Optimized Logic for .png extension) ---")
    
    # ... (前面的检查和目录创建逻辑保持不变) ...
    os.makedirs(os.path.dirname(DEST_JSON_PATH), exist_ok=True)
    os.makedirs(DEST_IMAGES_DIR, exist_ok=True)
    
    print(f"Reading original annotation file from: {SOURCE_JSON_PATH}")
    with open(SOURCE_JSON_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    data_to_process = all_data[:min(NUM_SAMPLES_TO_PROCESS, len(all_data))]
    print(f"Processing first {len(data_to_process)} samples from the source file...")
    
    valid_samples = []
    missing_files_list = []

    for item in tqdm(data_to_process, desc="Validating and copying images"):
        relative_path = item.get('image')
        if not relative_path:
            continue

        # --- 核心优化在此 ---
        source_path = os.path.join(SOURCE_BASE_DIR, relative_path)
        valid_source_path = None

        # 1. 首先，检查原始路径是否存在
        if os.path.exists(source_path):
            valid_source_path = source_path
        else:
            # 2. 如果不存在，尝试添加 .png 后缀再检查
            source_path_with_ext = source_path + ".png"
            if os.path.exists(source_path_with_ext):
                valid_source_path = source_path_with_ext

        # --- 优化结束 ---
        
        if valid_source_path:
            # 如果找到了有效路径，则添加样本并复制图片
            valid_samples.append(item)
            
            base_filename = os.path.basename(valid_source_path)
            destination_path = os.path.join(DEST_IMAGES_DIR, base_filename)
            if not os.path.exists(destination_path):
                 shutil.copy2(valid_source_path, destination_path)
        else:
            # 如果两种情况都找不到，则记录为缺失
            missing_files_list.append(relative_path)

    print(f"\nSaving {len(valid_samples)} valid samples to: {DEST_JSON_PATH}")
    with open(DEST_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(valid_samples, f, indent=2)
    
    if missing_files_list:
        print(f"Writing list of {len(missing_files_list)} missing files to: {MISSING_FILES_LOG_PATH}")
        with open(MISSING_FILES_LOG_PATH, 'w', encoding='utf-8') as log_file:
            for missing_path in missing_files_list:
                log_file.write(missing_path + '\n')

    print("\n--- Subset Creation Finished ---")
    print(f"✅ Total valid samples found and saved: {len(valid_samples)}")
    print(f"✅ Total images copied: {len(valid_samples)}")
    print(f"⚠️ Total samples skipped due to missing images: {len(missing_files_list)}")
    if missing_files_list:
        print(f"👉 A log of missing files has been saved to: {MISSING_FILES_LOG_PATH}")

if __name__ == "__main__":
    create_dataset_subset()