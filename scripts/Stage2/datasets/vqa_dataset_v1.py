# file: scripts/Stage2/datasets/vqa_dataset.py (DEBUG VERSION)

import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 假定 MedPLIB 的工具代码位于 Stage1 的目录中
from scripts.Stage1.model.medplib.mm_utils import tokenizer_image_token
from scripts.Stage1.utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from scripts.Stage1.model.medplib import conversation as conversation_lib


class VQADataset(Dataset):
    """Dataset for Stage 2: VQA Supervised Fine-Tuning."""
    
    def __init__(self, data_path, image_folder, tokenizer, image_processor, conv_mode="llava_v1"):
        with open(data_path, "r") as f:
            self.data = json.load(f)

        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        print(f"✅ VQADataset Initialized. Found {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # [DEBUG] 打印当前处理的样本索引
        print(f"\n--- [DEBUG] Processing sample index: {i} ---")
        item = self.data[i]
        
        try:
            image_file = os.path.basename(item['image']) # 使用 basename 确保安全
            image_path = os.path.join(self.image_folder, image_file)
            
            # [DEBUG] 打印正在尝试加载的图片路径
            print(f"[DEBUG] Attempting to load image: {image_path}")

            if not os.path.exists(image_path):
                print(f"❌ [DEBUG] FATAL: Image path does not exist, though it should in a clean dataset. Path: {image_path}")
                return None

            image = Image.open(image_path).convert('RGB')
            images_vit, images_convnext = self.image_processor(image)
            # [DEBUG] 打印图片加载成功信息
            print(f"✅ [DEBUG] Image loaded successfully.")

        except Exception as e:
            # [DEBUG] 打印详细的异常信息
            print(f"❌ [DEBUG] Exception during image loading for item {i}: {e}")
            return None

        conv = conversation_lib.conv_templates[self.conv_mode].copy()
        
        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        
        question_with_token = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], question_with_token)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        
        # [DEBUG] 打印构建的完整 prompt
        print(f"[DEBUG] Generated prompt: {prompt}")
        
        image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, image_token_id, return_tensors='pt').squeeze(0)
        labels = input_ids.clone()

        # Masking: 只对答案部分计算 Loss
        sep = conv.sep + conv.roles[1] + ":"
        parts = prompt.split(sep)
        
        # [DEBUG] 打印分割后的 prompt 部分
        print(f"[DEBUG] Prompt split by '{sep}'. Number of parts: {len(parts)}")
        for j, part in enumerate(parts):
            print(f"  [DEBUG] Part {j}: '{part[:100]}...'") # 只打印前100个字符

        if len(parts) != 2:
            print(f"❌ [DEBUG] Returning None because prompt format is incorrect. Expected 2 parts, got {len(parts)}.")
            return None # 格式不正确，过滤掉

        parts[0] += sep
        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer, image_token_id)) - 1

        labels[:instruction_len] = IGNORE_INDEX
        
        # [DEBUG] 打印最终返回的数据
        print(f"✅ [DEBUG] Successfully processed sample {i}. Returning data dictionary.")
        
        return dict(
            images_vit=images_vit,
            images_convnext=images_convnext,
            input_ids=input_ids,
            labels=labels
        )