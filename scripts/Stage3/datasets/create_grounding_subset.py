# file: create_grounding_subset.py
#
# 第三阶段专用：创建一个包含“标注、图像、掩码”三者对应的纯净数据集子集。

import json
import os
import shutil
import re
from tqdm import tqdm

# ==============================================================================
# --- ⚙️ 配置区 (HARDCODED CONFIGURATION) ---
# --- 请在此处填入您的实际路径和参数 ---
# ==============================================================================

# 1. 原始的、完整的 Grounding 标注文件路径
SOURCE_JSON_PATH = r"E:/MeCoVQA/train/MeCoVQA-Grounding.json"

# 2. 存放`images`和`masks`文件夹的“基础目录”的路径
SOURCE_BASE_DIR = r"E:/SAMed2Dv1" 

# 3. 【输出】您希望保存“子集标注文件”的完整路径
DEST_JSON_PATH = r"D:/SoftData/VSCODE/顶会论文/MedPLIB-main/20250607/datasets_subset_stage3/subset_MeCoVQA_Grounding_CLEAN.json"

# 4. 【输出】您希望保存“子集图像”的新建文件夹路径
DEST_IMAGES_DIR = r"D:/SoftData/VSCODE/顶会论文/MedPLIB-main/20250607/datasets_subset_stage3/images/"

# 5. 【输出】【新增】您希望保存“子集掩码”的新建文件夹路径
DEST_MASKS_DIR = r"D:/SoftData/VSCODE/顶会论文/MedPLIB-main/20250607/datasets_subset_stage3/masks/"

# 6. 您希望从原始文件中处理的样本数量（脚本会从这么多样本中筛选出有效的）
NUM_SAMPLES_TO_PROCESS = 15000 

# 7. 【输出】保存“未找到文件列表”的日志文件路径
MISSING_FILES_LOG_PATH = "D:/SoftData/VSCODE/顶会论文/MedPLIB-main/20250607/datasets_subset_stage3/missing_files_log.txt"

# ==============================================================================
# --- 脚本主逻辑 (无需修改) ---
# ==============================================================================

def check_path(base_dir, relative_path):
    """一个智能检查路径的辅助函数，会自动尝试添加.png后缀"""
    path_attempt = os.path.join(base_dir, relative_path)
    if os.path.exists(path_attempt):
        return path_attempt
    
    path_with_ext = path_attempt + ".png"
    if os.path.exists(path_with_ext):
        return path_with_ext
        
    return None

def create_dataset_subset():
    print("--- Starting Stage 3 Dataset Subset Creation ---")
    
    # 创建所有输出目录
    os.makedirs(os.path.dirname(DEST_JSON_PATH), exist_ok=True)
    os.makedirs(DEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(DEST_MASKS_DIR, exist_ok=True)
    
    print(f"Reading original annotation file: {SOURCE_JSON_PATH}")
    with open(SOURCE_JSON_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    data_to_process = all_data[:min(NUM_SAMPLES_TO_PROCESS, len(all_data))]
    print(f"Attempting to process first {len(data_to_process)} samples...")
    
    valid_samples = []
    missing_log = []

    # --- 第一遍：验证数据，收集有效样本和有效路径 ---
    for item in tqdm(data_to_process, desc="Validating data (images and masks)"):
        image_path_from_json = item.get('image')
        answer = item.get('conversations', [{}, {}])[1].get('value')

        if not image_path_from_json or not answer:
            continue
            
        mask_path_match = re.search(r'<mask>(.*?)<\/mask>', answer)
        if not mask_path_match:
            continue
        
        mask_path_from_json = mask_path_match.group(1)
        
        valid_image_path = check_path(SOURCE_BASE_DIR, image_path_from_json)
        valid_mask_path = check_path(SOURCE_BASE_DIR, mask_path_from_json)
        
        if valid_image_path and valid_mask_path:
            # 只有当图像和掩码都存在时，才视为有效样本
            # 存储清理过的相对路径，以备复制
            item['__valid_image_path__'] = valid_image_path
            item['__valid_mask_path__'] = valid_mask_path
            valid_samples.append(item)
        else:
            if not valid_image_path:
                missing_log.append(f"Image not found: {image_path_from_json}")
            if not valid_mask_path:
                missing_log.append(f"Mask not found: {mask_path_from_json}")

    print(f"\nValidation finished. Found {len(valid_samples)} complete and valid samples.")

    # --- 第二遍：保存纯净的标注文件并复制文件 ---
    # 从样本中移除我们添加的临时路径键
    for sample in valid_samples:
        valid_image_path = sample.pop('__valid_image_path__')
        valid_mask_path = sample.pop('__valid_mask_path__')

        # 复制图像
        image_dest = os.path.join(DEST_IMAGES_DIR, os.path.basename(valid_image_path))
        if not os.path.exists(image_dest):
            shutil.copy2(valid_image_path, image_dest)
            
        # 复制掩码
        mask_dest = os.path.join(DEST_MASKS_DIR, os.path.basename(valid_mask_path))
        if not os.path.exists(mask_dest):
            shutil.copy2(valid_mask_path, mask_dest)

    print(f"Copying files for {len(valid_samples)} samples finished.")

    with open(DEST_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(valid_samples, f, indent=2)
    print(f"Clean annotation subset saved to: {DEST_JSON_PATH}")

    # --- 写入缺失文件日志 ---
    if missing_log:
        print(f"Writing {len(missing_log)} missing file entries to: {MISSING_FILES_LOG_PATH}")
        with open(MISSING_FILES_LOG_PATH, 'w', encoding='utf-8') as log_f:
            for line in missing_log:
                log_f.write(line + '\n')

    print("\n--- Subset Creation Finished ---")
    print(f"✅ Total valid samples (image+mask): {len(valid_samples)}")
    print(f"⚠️ Total invalid/incomplete samples skipped: {len(data_to_process) - len(valid_samples)}")


if __name__ == "__main__":
    create_dataset_subset()