# file: scripts/Stage2/datasets/vqa_dataset.py

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
        """
        Args:
            data_path (str): Path to the VQA JSON data file.
            image_folder (str): Path to the folder containing images.
            tokenizer: The tokenizer.
            image_processor (callable): A function/callable that takes a PIL image and returns two tensors (images_vit, images_convnext).
            conv_mode (str): The conversation template mode.
        """
        # VQA 数据格式示例 (JSON 列表):
        # [
        #   {
        #     "id": "...", "image": "image_name.jpg",
        #     "conversations": [
        #       {"from": "human", "value": "Question about the image?"},
        #       {"from": "gpt", "value": "Answer to the question."}
        #     ]
        #   }, ...
        # ]
        with open(data_path, "r") as f:
            self.data = json.load(f)

        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        
        try:
            image_file = item['image']
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            images_vit, images_convnext = self.image_processor(image)
        except Exception:
            return None # 由 collate_fn 过滤

        conv = conversation_lib.conv_templates[self.conv_mode].copy()
        
        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        
        image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, image_token_id, return_tensors='pt').squeeze(0)
        labels = input_ids.clone()

        # Masking: 只对答案部分计算 Loss
        sep = conv.sep + conv.roles[1] + ":"
        total_len = len(labels)
        
        # 找到 "ASSISTANT:" 出现的位置
        parts = prompt.split(sep)
        if len(parts) != 2:
            return None # 格式不正确，过滤掉

        # parts[0] 是 "USER: <image>\n{question}\n"
        # parts[1] 是 " {answer}</s>"
        parts[0] += sep
        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer, image_token_id)) - 1 # -1 to keep the first token of the answer

        labels[:instruction_len] = IGNORE_INDEX
        
        return dict(
            images_vit=images_vit,
            images_convnext=images_convnext,
            input_ids=input_ids,
            labels=labels
        )