# file: scripts/Stage2/datasets/vqa_dataset.py
#
# 最终版：一个完整、健壮的VQA数据集处理脚本
# 它能正确处理图片路径，并为SFT精确地创建标签掩码

import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 假定 MedPLIB 的工具代码位于 Stage1 的目录中
# 请确保您的项目结构与此处的导入路径匹配
from scripts.Stage1.model.medplib.mm_utils import tokenizer_image_token
from scripts.Stage1.utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from scripts.Stage1.model.medplib import conversation as conversation_lib


class VQADataset(Dataset):
    """Dataset for Stage 2: VQA Supervised Fine-Tuning."""
    
    def __init__(self, data_path, image_folder, tokenizer, image_processor, conv_mode="llava_v1"):
        """
        Args:
            data_path (str): Path to the VQA JSON data file.
            image_folder (str): Path to the folder containing images.
            tokenizer: The tokenizer.
            image_processor (callable): A function/callable that takes a PIL image and returns two tensors (images_vit, images_convnext).
            conv_mode (str): The conversation template mode.
        """
        print(f"Initializing VQADataset...")
        print(f"  - Loading annotations from: {data_path}")
        try:
            with open(data_path, "r", encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            raise IOError(f"❌ Could not read or parse data file: {data_path}. Error: {e}")

        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        print(f"✅ VQADataset Initialized. Found {len(self.data)} samples. Image folder: '{self.image_folder}'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        
        try:
            # --- [核心优化] ---
            # 智能处理图片路径，兼容缺少 .png 后缀的情况
            image_relative_path_from_json = item['image']
            
            # 无论原始路径是什么格式，都先提取出基本文件名/路径
            base_filename = os.path.basename(image_relative_path_from_json)
            
            # 构造一个不带.png后缀的潜在路径（以应对json中就没有.png的情况）
            path_attempt = os.path.join(self.image_folder, base_filename)
            
            valid_image_path = None

            # 1. 首先，检查这个构造出的路径是否存在
            if os.path.exists(path_attempt):
                valid_image_path = path_attempt
            else:
                # 2. 如果不存在，尝试在文件名后添加 .png 后缀再检查
                path_with_ext = path_attempt + ".png"
                if os.path.exists(path_with_ext):
                    valid_image_path = path_with_ext
            
            if not valid_image_path:
                # 如果两种方式都找不到，则此样本无效
                # print(f"Warning: Skipping sample {i}. Image not found for entry: {image_relative_path_from_json}")
                return None
            # --- 优化结束 ---

            image = Image.open(valid_image_path).convert('RGB')
            images_vit, images_convnext = self.image_processor(image)

        except Exception as e:
            # print(f"Warning: Skipping sample {i} due to image loading error: {e}")
            return None

        # --- 对话处理和标签掩码（保持不变） ---
        conv = conversation_lib.conv_templates[self.conv_mode].copy()
        
        # 假设VQA只有一轮问答
        if not (isinstance(item['conversations'], list) and len(item['conversations']) >= 2):
            return None # 过滤掉格式不正确的对话

        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        
        question_with_token = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], question_with_token)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        
        image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, image_token_id, return_tensors='pt').squeeze(0)
        labels = input_ids.clone()

        # Masking: 只对答案部分计算 Loss
        sep = conv.sep + conv.roles[1] + ":"
        
        parts = prompt.split(sep)
        if len(parts) != 2:
            return None # 格式不正确，过滤掉

        parts[0] += sep
        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer, image_token_id)) - 1

        labels[:instruction_len] = IGNORE_INDEX
        
        return dict(
            images_vit=images_vit,
            images_convnext=images_convnext,
            input_ids=input_ids,
            labels=labels
        )