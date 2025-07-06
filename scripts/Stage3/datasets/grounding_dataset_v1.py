# file: scripts/Stage3/datasets/grounding_dataset.py (Enhanced Debug Version)

print("="*50)
print("===== EXECUTING THE CORRECTLY MODIFIED grounding_dataset_0627.py =====")
print(f"      File location: {__file__}")
print("="*50)

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
        # ======================================================================
        # --- 调试信息：阶段1 ---
        print(f"\n--- [DEBUG] Attempting to process sample at index: {i} ---")
        item = self.data[i]
        print(f"[DEBUG] Raw item data: {json.dumps(item, indent=2)}")
        # ======================================================================

        try:
            # --- 调试信息：阶段2 (加载图像) ---
            image_path_from_json = item['image']
            base_filename = os.path.basename(image_path_from_json)
            image_path = os.path.join(self.data_base_dir, "images", base_filename)
            print(f"[DEBUG] Constructed image path: {image_path}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file does not exist at path: {image_path}")
            
            input_image = Image.open(image_path).convert('RGB')
            images_vit, images_convnext = self.image_processor(input_image)
            print("✅ [DEBUG] Input image loaded and processed successfully.")
            # ------------------------------------

            # --- 调试信息：阶段3 (解析和加载Mask) ---
            answer = item['conversations'][1]['value']
            print(f"[DEBUG] Answer string to parse: '{answer}'")
            mask_path_match = re.search(r'<mask>(.*?)<\/mask>', answer)
            
            if not mask_path_match:
                print(f"⚠️  [DEBUG] SKIPPING sample {i}: No <mask> tag found. This is not a grounding sample.")
                return None

            mask_relative_path = mask_path_match.group(1)
            print(f"[DEBUG] Extracted mask relative path: '{mask_relative_path}'")
            mask_path = os.path.join(self.data_base_dir, mask_relative_path)
            print(f"[DEBUG] Constructed mask path: {mask_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file does not exist at path: {mask_path}")

            gt_mask_image = Image.open(mask_path).convert('L')
            gt_masks_tensor = self.mask_transform(gt_mask_image)
            gt_masks_tensor = (gt_masks_tensor > 0.5).float()
            print("✅ [DEBUG] Ground truth mask image loaded and processed successfully.")
            # ------------------------------------

            # --- 调试信息：阶段4 (处理文本和Tokenize) ---
            instruction = item['conversations'][0]['value'].replace("<image>\n", "")
            full_instruction = instruction + "\n<SEG>"

            # 使用 MedPLIB 的 conversation 模板
            conv = conversation_lib.conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + full_instruction)
            conv.append_message(conv.roles[1], None) # gpt的回合留空
            prompt = conv.get_prompt()
            print(f"[DEBUG] Final prompt for tokenizer: '{prompt[:200]}...'")
            
            # 使用 MedPLIB 的 tokenizer_image_token 函数
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                image_token_index=-200, # LLaVA标准实践
                return_tensors='pt'
            ).squeeze(0)
            
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            print(f"✅ [DEBUG] Prompt tokenized successfully. Shape: {input_ids.shape}")
            # ------------------------------------

        except Exception as e:
            # --- 调试信息：阶段5 (捕获所有未知异常) ---
            print(f"❌ [FATAL ERROR] An unexpected exception occurred while processing sample at index: {i}")
            print(f"   - Exception Type: {type(e).__name__}")
            print(f"   - Exception Details: {e}")
            # 重新抛出异常，让整个程序停止，并显示清晰的错误信息
            raise e
            # ------------------------------------

        # --- 调试信息：阶段6 (成功返回) ---
        print(f"✅ [SUCCESS] Sample {i} processed successfully. Returning data dictionary.")
        # ------------------------------------
        
        return dict(
            images_vit=images_vit, 
            images_convnext=images_convnext, 
            input_ids=input_ids, 
            labels=labels, 
            gt_masks=gt_masks_tensor.squeeze(0)
        )

    # def __getitem__(self, i):
    #     # --- [新增] 详细的调试信息 ---
    #     print(f"\n--- [DEBUG] Attempting to process sample at index: {i} ---")
    #     item = self.data[i]
    #     print(f"[DEBUG] Raw item data: {json.dumps(item, indent=2)}")

    #     try:
    #         # 1. 处理输入图像
    #         image_relative_path = item.get('image')
    #         if not image_relative_path:
    #             print("❌ [DEBUG] 'image' key not found in item. Returning None.")
    #             return None
            
    #         image_path = os.path.join(self.data_base_dir, image_relative_path)
    #         print(f"[DEBUG] Constructed image path: {image_path}")
    #         if not os.path.exists(image_path):
    #             print(f"❌ [DEBUG] Image file does not exist at path. Returning None.")
    #             return None
            
    #         input_image = Image.open(image_path).convert('RGB')
    #         images_vit, images_convnext = self.image_processor(input_image)
    #         print("✅ [DEBUG] Input image loaded and processed successfully.")

    #         # 2. 解析并加载真值掩码图像
    #         answer = item.get('conversations', [{}, {}])[1].get('value')
    #         if not answer:
    #             print("❌ [DEBUG] Answer string is missing or invalid. Returning None.")
    #             return None
    #         print(f"[DEBUG] Answer string to parse: '{answer}'")

    #         mask_path_match = re.search(r'<mask>(.*?)<\/mask>', answer)
    #         if not mask_path_match:
    #             print("❌ [DEBUG] Regex failed to find <mask> tag in answer. Returning None.")
    #             return None
            
    #         mask_relative_path = mask_path_match.group(1)
    #         print(f"[DEBUG] Extracted mask relative path: '{mask_relative_path}'")
            
    #         mask_path = os.path.join(self.data_base_dir, mask_relative_path)
    #         print(f"[DEBUG] Constructed mask path: {mask_path}")
    #         if not os.path.exists(mask_path):
    #             print(f"❌ [DEBUG] Mask file does not exist at path. Returning None.")
    #             return None

    #         gt_mask_image = Image.open(mask_path).convert('L')
    #         print("✅ [DEBUG] Ground truth mask image loaded successfully.")

    #         # 3. 处理掩码为标签
    #         gt_masks_tensor = self.mask_transform(gt_mask_image)
    #         gt_masks_tensor = (gt_masks_tensor > 0.5).float()
    #         print("✅ [DEBUG] Ground truth mask processed to tensor.")

    #         # 4. 处理文本指令
    #         # --- 文本处理部分 (最终修正) ---
    #         instruction = item['conversations'][0]['value']
    #         full_instruction = instruction + "\n<SEG>"
            
    #         conv = conversation_lib.conv_templates["llava_v1"].copy()
    #         conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + full_instruction)
    #         conv.append_message(conv.roles[1], None) 
    #         prompt = conv.get_prompt()
            
    #         # [核心修正1] 在这里获取 image_token_id
    #         image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            
    #         # [核心修正2] 将获取到的 image_token_id 传递给函数
    #         input_ids = tokenizer_image_token(
    #             prompt, 
    #             self.tokenizer, 
    #             image_token_index=image_token_id, # <-- 明确地传入ID
    #             return_tensors='pt'
    #         ).squeeze(0)
            
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     except Exception as e:
    #         # 打印出详细的错误信息和出错的样本信息
    #         print(f"❌ [FATAL ERROR] Failed to process data item at index: {i}")
    #         image_path = item.get('image', 'N/A')
    #         print(f"   - Associated image file: {os.path.join(self.data_base_dir, image_path)}")
    #         print(f"   - Original Exception: {e}")
            
    #         # 重新抛出异常，这将使整个训练程序停止，并显示完整的错误堆栈
    #         # 这是定位问题的关键步骤
    #         raise e

    #     return dict(
    #         images_vit=images_vit,
    #         images_convnext=images_convnext,
    #         input_ids=input_ids,
    #         labels=labels,
    #         masks_list=[gt_masks_tensor],
    #         seg_flag=torch.tensor([True])
    #     )