# file: scripts/Stage3/datasets/grounding_dataset.py (Enhanced Debug Version)

import os
import json
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 为了让这个脚本能在后续被主训练脚本正确导入，我们先做一些路径假设
# 真正的路径设置将在主训练脚本的开头完成
try:
    from model.medplib.mm_utils import tokenizer_image_token
    from utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
    from model.medplib import conversation as conversation_lib
except ImportError:
    print("Warning: Could not import from MedPLIB source. This is expected if running this script standalone.")
    DEFAULT_IMAGE_TOKEN = "<image>"
    IGNORE_INDEX = -100


class GroundingDataset(Dataset):
    """Dataset for Stage 3: Grounding Expert SFT."""

    def __init__(self, data_path, data_base_dir, tokenizer, image_processor):
        print(f"Initializing GroundingDataset...")
        print(f"  - Loading annotations from: {data_path}")
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)

        self.data_base_dir = data_base_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mask_transform = transforms.ToTensor()
        print(f"✅ GroundingDataset Initialized. Found {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # [新增] 打印正在处理的样本索引
        print(f"--- [DATASET_DEBUG] Getting item at index: {i} ---")
        item = self.data[i]
        
        try:
            # 1. 处理输入图像
            image_relative_path = item['image']
            image_path = os.path.join(self.data_base_dir, image_relative_path)
            input_image = Image.open(image_path).convert('RGB')
            images_vit, images_convnext = self.image_processor(input_image)

            # 2. 解析并加载真值掩码图像
            answer = item['conversations'][1]['value']
            mask_path_match = re.search(r'<mask>(.*?)<\/mask>', answer)
            if not mask_path_match: 
                print(f"❌ [DATASET_DEBUG] Index {i}: Regex failed to find <mask> tag.")
                return None
            
            mask_relative_path = mask_path_match.group(1)
            mask_path = os.path.join(self.data_base_dir, mask_relative_path)
            if not os.path.exists(mask_path):
                # 智能尝试添加.png
                if os.path.exists(mask_path + '.png'):
                    mask_path += '.png'
                else:
                    print(f"❌ [DATASET_DEBUG] Index {i}: Mask file not found at '{mask_path}' (and with .png).")
                    return None

            gt_mask_image = Image.open(mask_path).convert('L')
            gt_masks_tensor = self.mask_transform(gt_mask_image)
            gt_masks_tensor = (gt_masks_tensor > 0.5).float()

            # --- [核心修正] ---
            # 使用与MedPLIB原作一致的对话模板和图文分词函数
            instruction = item['conversations'][0]['value']
            
            # 使用Llava的对话模板来构建prompt
            conv = conversation_lib.conv_templates["llava_v1"].copy()
            # 将<image>和指令放在USER的回合
            conv.append_message(conv.roles[0], self.DEFAULT_IMAGE_TOKEN + '\n' + instruction)
            # 将<SEG>作为ASSISTANT的回答开头
            conv.append_message(conv.roles[1], "<SEG>")
            prompt = conv.get_prompt()

            # 使用专门的函数来处理包含<image> token的prompt
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.DEFAULT_IMAGE_TOKEN)
            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                image_token_index=image_token_id,
                return_tensors='pt'
            ).squeeze(0)
            
            # 我们只训练模型生成<SEG>的能力，所以将所有其他部分的标签都mask掉
            labels = torch.full_like(input_ids, self.IGNORE_INDEX)
            # 找到<SEG> token的位置，让模型只学习预测它
            seg_token_id = self.tokenizer.convert_tokens_to_ids("<SEG>")
            seg_token_indices = torch.where(input_ids == seg_token_id)[0]
            if len(seg_token_indices) > 0:
                labels[seg_token_indices[0] - 1] = input_ids[seg_token_indices[0] -1] # 学习预测ASSISTANT:之后的<SEG>

        except Exception as e:
            # print(f"Error processing sample {i}: {e}") # for debug
            return None

        return dict(
            images_vit=images_vit,
            images_convnext=images_convnext,
            input_ids=input_ids,
            labels=labels,
            gt_masks=gt_masks_tensor.squeeze(0),
        )