# file: datasets/alignment_dataset.py
# THE FINAL VERSION - BUILT ON USER'S GROUND TRUTH
# This version implements the user's explicit instruction:
# It discards all directory info from the jsonl's `image` field using `os.path.basename()`
# and joins it directly with the provided `image_folder`.

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from model.medplib.mm_utils import tokenizer_image_token
from utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX

class AlignmentDataset(Dataset):
    def __init__(self, data_path, image_folder, tokenizer, **kwargs):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.data = self._load_data(data_path)
        self.vit_transform, self.convnext_transform = self._setup_transforms()
        print(f"✅ Dataset Initialized. Base image folder: '{self.image_folder}'. Found {len(self.data)} samples.")

    def _load_data(self, data_path):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            raise IOError(f"❌ Could not read or parse data file: {data_path}. Error: {e}")

    def _setup_transforms(self):
        return (
            transforms.Compose([
                transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ]),
            transforms.Compose([
                transforms.Resize((768, 768), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        try:
            item = self.data[i]
            image_filename_from_jsonl = item.get('image')
            prompt = item.get('caption', '').strip()

            if not image_filename_from_jsonl or not prompt or DEFAULT_IMAGE_TOKEN not in prompt:
                return None

            # 🔥 THE ONLY CORRECT PATH LOGIC BASED ON USER INSTRUCTION:
            base_filename = os.path.basename(image_filename_from_jsonl)
            image_path = os.path.join(self.image_folder, base_filename)

            # Now, we check if this simple, correct path exists.
            if not os.path.exists(image_path):
                # If it still doesn't exist, the file is truly missing from the folder.
                # This print will help find those few missing files.
                # print(f"Warning: Skipping. File not found at the definitive path: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            
            images_vit = self.vit_transform(image)
            images_convnext = self.convnext_transform(image)

            image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, image_token_id, return_tensors='pt').squeeze(0)
            
            labels = input_ids.clone()
            token_indices = torch.where(input_ids == image_token_id)[0]
            if len(token_indices) > 0:
                labels[:token_indices[0] + 1] = IGNORE_INDEX
            else:
                labels[:] = IGNORE_INDEX
            
            return dict(
                images_vit=images_vit,
                images_convnext=images_convnext,
                input_ids=input_ids,
                labels=labels
            )
        except Exception:
            return None