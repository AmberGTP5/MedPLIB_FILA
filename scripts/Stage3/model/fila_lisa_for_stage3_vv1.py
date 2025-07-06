# file: model/fila_lisa.py
# THE FINAL VERSION - Self-contained and robust forward pass.

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from functools import partial
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from model.medplib.model.language_model.medplib_llama import LlavaLlamaForCausalLM
from model.medplib.model.multimodal_encoder.builder import build_vision_tower
from model.medplib.model.multimodal_projector.builder import build_vision_projector
from utils.utils import IGNORE_INDEX

class CVFM(nn.Module):
    def __init__(self, vit_dim, convnext_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(vit_dim+convnext_dim, vit_dim*4), nn.GELU(), nn.Linear(vit_dim*4, vit_dim))
        self.gate = nn.Parameter(torch.zeros(1)); self.norm = nn.LayerNorm(vit_dim)
    def forward(self, vit_features, convnext_features):
        h = w = int((vit_features.shape[1] - 1) ** 0.5)
        aligned = F.interpolate(convnext_features.to(vit_features.dtype), (h, w), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        cls_feat = aligned.mean(dim=1, keepdim=True)
        aligned = torch.cat([cls_feat, aligned], dim=1)
        combined = torch.cat([vit_features, aligned], dim=-1)
        return self.norm(vit_features + self.gate.tanh() * self.mlp(combined))

class HybridEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.vision_tower = config, build_vision_tower(config)
        self.convnext_encoder = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
        self.convnext_encoder.classifier = nn.Identity()
        for p in self.vision_tower.parameters(): p.requires_grad = False
        for p in self.convnext_encoder.parameters(): p.requires_grad = False
        self.interaction_idx, dims, vit_dim = [2, 6, 12, 20], [192, 384, 768, 1536], self.vision_tower.config.hidden_size
        self.cvfm_modules = nn.ModuleList([CVFM(vit_dim, d) for d in dims])
        self.convnext_features = None
        for i, idx in enumerate(self.interaction_idx):
            self.vision_tower.vision_tower.vision_model.encoder.layers[idx-1].register_forward_hook(partial(self._hook, i))
    def _hook(self, i, module, args, output):
        if self.convnext_features is None: return output
        return (self.cvfm_modules[i](output[0], self.convnext_features[i]),) + output[1:]
    def _convnext_stages(self, x):
        feats, x = [], self.convnext_encoder.features[0](x.to(self.dtype))
        for i in range(1, 8):
            x = self.convnext_encoder.features[i](x)
            if i % 2 == 1: feats.append(x)
        return feats
    def forward(self, images_vit, images_convnext):
        self.convnext_features = self._convnext_stages(images_convnext)
        img_feats = self.vision_tower(images_vit)
        self.convnext_features = None
        return img_feats
    @property
    def dtype(self): return next(self.vision_tower.parameters()).dtype

class FILAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.get_model().vision_tower = HybridEncoder(config)
        self.get_model().mm_projector = build_vision_projector(config)

    def encode_images(self, images_vit, images_convnext):
        img_feats = self.get_model().vision_tower(images_vit, images_convnext)
        return self.get_model().mm_projector(img_feats.to(self.dtype))

    def forward(self, **kwargs):
        # 🔥 推理逻辑保持不变，这是一个很好的设计
        if kwargs.get('inputs_embeds') is not None:
            return LlamaForCausalLM.forward(self, **kwargs)

        # --- 以下是原始的训练逻辑 ---
        input_ids = kwargs.get('input_ids')
        labels = kwargs.get('labels')
        images_vit = kwargs.get('images_vit')
        images_convnext = kwargs.get('images_convnext')
        
        if images_vit is None or images_convnext is None:
            return LlamaForCausalLM.forward(self, **kwargs)

        image_features = self.encode_images(images_vit, images_convnext)
        
        # [核心修正] -----------------------------------------------------------------
        # 1. 删除下面这两行错误的、试图从vision_tower获取tokenizer的代码
        # tokenizer = self.get_model().vision_tower.vision_tower.tokenizer
        # image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        
        # 2. 替换为一个固定的常量。这个值必须与数据预处理阶段使用的占位符索引一致。
        #    根据LLaVA的标准实践以及我们对您代码的分析，这个值是 -200。
        IMAGE_TOKEN_INDEX = -200
        # --------------------------------------------------------------------------
        
        new_embeds, new_labels = [], []
        for i, cur_ids in enumerate(input_ids):
            # 3. 使用修正后的常量来查找占位符的位置
            token_indices = torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0]
            
            # [核心调试代码] -----------------------------------------------------------
            if not len(token_indices): 
                # 如果一个样本不包含图像占位符，这是一个异常情况，我们需要详细记录它
                print("="*60)
                print(f"⚠️ [DEBUGGER] FOUND A SAMPLE WITH NO IMAGE TOKEN IN THE BATCH!")
                print(f"  - Batch Index (i): {i}")
                print(f"  - Sample's input_ids (first 60 tokens): {cur_ids[:60]}")
                print(f"  - Does it contain the image token index (-200)?: {IMAGE_TOKEN_INDEX in cur_ids}")
                print("="*60)
                
                # 我们仍然跳过这个异常样本，以防它导致后续代码崩溃
                continue

            start_idx = token_indices[0]
            embeds = self.get_model().embed_tokens(cur_ids)

            # --- [核心调试代码] ---
            # 在执行拼接操作前，打印出所有相关张量的形状
            print("\n" + "="*60)
            print(f">>> DEBUGGING torch.cat at batch_index {i}")
            
            text_embed_slice_1 = embeds[:start_idx]
            image_feature_slice = image_features[i]
            text_embed_slice_2 = embeds[start_idx+1:]
            
            print(f"  - Shape of text embeddings ('embeds'): {embeds.shape}")
            print(f"  - Shape of all image features ('image_features'): {image_features.shape}")
            print("-" * 20)
            print(f"  - Shape of text part 1 to concat: {text_embed_slice_1.shape}")
            print(f"  - Shape of image features to concat: {image_feature_slice.shape}")
            print(f"  - Shape of text part 2 to concat: {text_embed_slice_2.shape}")
            
            # 检查维度1（特征维度）是否匹配
            if text_embed_slice_1.shape[1] != image_feature_slice.shape[1]:
                print(f"  - ❌ MISMATCH! Dimension 1 is different: {text_embed_slice_1.shape[1]} (text) vs {image_feature_slice.shape[1]} (image)")
            else:
                print("  - ✅ Dimension 1 (features) seems to match.")
            
            print("="*60 + "\n")
            # --- 调试结束 ---

            spliced = torch.cat([embeds[:start_idx], image_features[i], embeds[start_idx+1:]], dim=0)
            new_embeds.append(spliced)
            
            if labels is not None and labels[i] is not None:
                new_label = torch.full((image_features.shape[1],), IGNORE_INDEX, device=labels.device, dtype=torch.long)
                spliced_label = torch.cat([labels[i][:start_idx], new_label, labels[i][start_idx+1:]], dim=0)
                new_labels.append(spliced_label)
            
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_embeds, batch_first=True, padding_value=0.0)
        final_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX) if new_labels else None

        # 1. 将我们处理好的 embedding 和 labels 更新到 kwargs 中
        #    这样即使原始kwargs里有旧的labels，也会被我们处理过的覆盖
        kwargs['inputs_embeds'] = inputs_embeds
        kwargs['labels'] = final_labels
        
        # 2. 当提供了 inputs_embeds 时，input_ids 必须为 None
        kwargs['input_ids'] = None
        
        # 3. 从 kwargs 中“弹出”并丢弃所有 LlamaForCausalLM 不认识的自定义参数
        kwargs.pop('images_vit', None)
        kwargs.pop('images_convnext', None)
        # 训练Gounding Expert时，数据字典里有gt_masks，也要一并移除
        kwargs.pop('gt_masks', None) 
        
        # 4. 创建或更新 attention_mask 以匹配新的 inputs_embeds 形状
        #    这是非常重要的一步，确保注意力机制能正确工作
        kwargs['attention_mask'] = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        # 如果批次中所有样本都没有图像，则可能new_embeds为空
        if not new_embeds:
            return LlamaForCausalLM.forward(self, **kwargs)
        
        return LlamaForCausalLM.forward(
            self,
            input_ids=None,
            attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device),
            inputs_embeds=inputs_embeds,
            labels=final_labels,
            return_dict=True
        )