import base64
from io import BytesIO

import torch
from PIL import Image
from transformers import StoppingCriteria

# from .constants import IMAGE_TOKEN_INDEX
from utils.utils import IMAGE_TOKEN_INDEX,REGION_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors="pt")["pixel_values"]

def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
    image_token_id = tokenizer.convert_tokens_to_ids('<image>')

    print("\n" + "=" * 20)
    print("--- [DEBUG_MM_UTILS] Entering tokenizer_image_token ---")
    print(f"[DEBUG_MM_UTILS] Received prompt (first 150 chars): '{prompt[:150]}...'")
    print(f"[DEBUG_MM_UTILS] Image token id to be used: {image_token_id}")

    chunks = prompt.split("<image>")
    print(f"[DEBUG_MM_UTILS] Step 1: Prompt split by '<image>' into {len(chunks)} chunk(s).")

    input_ids = []
    for idx, chunk in enumerate(chunks):
        print(f"  [DEBUG_MM_UTILS]   - Tokenizing chunk {idx}: '{chunk[:100]}...'")
        tokenized_output = tokenizer(chunk)
        chunk_ids = tokenized_output.input_ids

        if idx == 0 and len(chunk_ids) > 0 and tokenized_output.input_ids[0] == tokenizer.bos_token_id:
            input_ids.append(chunk_ids[0])
            input_ids.extend(chunk_ids[1:])
            print(f"  [DEBUG_MM_UTILS]   - Handled BOS token in chunk {idx}.")
        else:
            input_ids.extend(chunk_ids)

        if idx < len(chunks) - 1:
            input_ids.append(image_token_id)
            print(f"  [DEBUG_MM_UTILS]   - Inserted image token at position {len(input_ids) - 1}.")

    print(f"[DEBUG_MM_UTILS] Final constructed input_ids: {input_ids}")
    print("--- [DEBUG_MM_UTILS] Exiting tokenizer_image_token ---")
    print("=" * 20 + "\n")

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor([input_ids], dtype=torch.long)  # 带 batch 维度
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids

# def tokenizer_image_token(
#     prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
# ):
#     # [新增] 健壮性检查：如果传入的image_token_index是None，则使用默认值-200
#     if image_token_index is None:
#         image_token_index = IMAGE_TOKEN_INDEX # 确保它有一个默认的整数值
#     # --- [新增的超详细调试代码] ---
    
#     print("\n" + "="*20)
#     print("--- [DEBUG_MM_UTILS] Entering tokenizer_image_token ---")
#     print(f"[DEBUG_MM_UTILS] Received prompt (first 150 chars): '{prompt[:150]}...'")
#     print(f"[DEBUG_MM_UTILS] Image token index to be used: {image_token_index}")
    
#     # 1. 切割prompt
#     try:
#         chunks = prompt.split("<image>")
#         print(f"[DEBUG_MM_UTILS] Step 1: Prompt split by '<image>' into {len(chunks)} chunk(s).")
#     except Exception as e:
#         print(f"❌ [DEBUG_MM_UTILS] FAILED at Step 1 (split). Error: {e}")
#         raise e

#     # 2. 对每个部分进行分词
#     prompt_chunks = []
#     print("[DEBUG_MM_UTILS] Step 2: Tokenizing each chunk...")
#     for i, chunk in enumerate(chunks):
#         # 打印将要被分词的文本块
#         print(f"  [DEBUG_MM_UTILS]   - Tokenizing chunk {i}: '{chunk[:100]}...'")
#         try:
#             # 检查tokenizer返回的类型
#             tokenized_output = tokenizer(chunk)
#             chunk_ids = tokenized_output.input_ids
#             print(f"  [DEBUG_MM_UTILS]     - Success. Tokenized ids: {chunk_ids}")
#             prompt_chunks.append(chunk_ids)
#         except Exception as e:
#             print(f"❌ [DEBUG_MM_UTILS] FAILED at Step 2 (tokenizing chunk {i}). Error: {e}")
#             # 即使出错，也添加一个标记，以便追踪
#             prompt_chunks.append(f"ERROR_TOKENIZING_CHUNK_{i}")
    
#     # 3. 拼接最终的input_ids列表
#     print("[DEBUG_MM_UTILS] Step 3: Constructing final input_ids list...")
#     input_ids = []
#     offset = 0
#     if (len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0] and prompt_chunks[0][0] == tokenizer.bos_token_id):
#         offset = 1
#         input_ids.append(prompt_chunks[0][0])
#         print(f"[DEBUG_MM_UTILS]   - Handled BOS token. Offset is now {offset}.")

#     # 这个函数的作用是在列表元素间插入分隔符
#     def insert_separator(X, sep):
#         return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]
    
#     # 因为我们的prompt中可能有多个特殊token，我们只按<image>分割
#     num_image_tokens = prompt.count("<image>")
    
#     # 准备插入的列表
#     chunks_to_insert = insert_separator(prompt_chunks, [image_token_index] * num_image_tokens)
#     print(f"[DEBUG_MM_UTILS]   - Prepared chunks to insert: {chunks_to_insert}")

#     for x in chunks_to_insert:
#         # 检查每个要被添加的块 x 是否为None或有问题
#         print(f"[DEBUG_MM_UTILS]   - Extending with chunk: {x}")
#         if x is None:
#             print("❌ [DEBUG_MM_UTILS] CRITICAL: Found a None chunk before extend. Aborting this item.")
#             return None # 如果有None就提前退出
#         input_ids.extend(x[offset:])
        
#     print(f"[DEBUG_MM_UTILS]   - Final constructed list: {input_ids}")
#     print("--- [DEBUG_MM_UTILS] Exiting tokenizer_image_token ---")
#     print("="*20 + "\n")
#     # --- 调试代码结束 ---

#     if return_tensors is not None:
#         if return_tensors == "pt":
#             return torch.tensor(input_ids, dtype=torch.long)
#         raise ValueError(f"Unsupported tensor type: {return_tensors}")
#     return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
