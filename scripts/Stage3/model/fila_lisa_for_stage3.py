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
    def __init__(self, config, tokenizer, **kwargs):
        super().__init__(config)
        self.tokenizer = tokenizer  # 保存 tokenizer 以便 forward 使用
        self.get_model().vision_tower = HybridEncoder(config)
        self.get_model().mm_projector = build_vision_projector(config)

    def encode_images(self, images_vit, images_convnext):
        img_feats = self.get_model().vision_tower(images_vit, images_convnext)
        return self.get_model().mm_projector(img_feats.to(self.dtype))

    def forward(self, **kwargs):
        if kwargs.get('inputs_embeds') is not None and kwargs.get('input_ids') is None:
            return LlamaForCausalLM.forward(self, **kwargs)

        input_ids = kwargs.pop('input_ids')
        labels = kwargs.pop('labels', None)
        images_vit = kwargs.pop('images_vit', None)
        images_convnext = kwargs.pop('images_convnext', None)

        if images_vit is None or images_convnext is None:
            kwargs['input_ids'] = input_ids
            kwargs['labels'] = labels
            return LlamaForCausalLM.forward(self, **kwargs)

        image_features = self.encode_images(images_vit, images_convnext)

        IMAGE_TOKEN_ID = self.tokenizer.convert_tokens_to_ids('<image>')
        IGNORE_INDEX = -100

        new_embeds, new_labels = [], []
        for i, cur_ids in enumerate(input_ids):
            token_indices = torch.where(cur_ids == IMAGE_TOKEN_ID)[0]

            if not len(token_indices):
                print("=" * 60)
                print(f"⚠️ [DEBUGGER] FOUND A SAMPLE WITH NO IMAGE TOKEN IN THE BATCH!")
                print(f"  - Batch Index (i): {i}")
                print(f"  - Sample's input_ids (first 60 tokens): {cur_ids[:60]}")
                print(f"  - Does it contain the image token index?: {IMAGE_TOKEN_ID in cur_ids}")
                print("=" * 60)
                continue

            start_idx = token_indices[0].item()
            embeds = self.get_model().embed_tokens(cur_ids)

            assert start_idx < embeds.shape[0], f"Start index {start_idx} exceeds embed size {embeds.shape[0]}"

            print("\n" + "=" * 60)
            print(f">>> DEBUGGING torch.cat at batch_index {i}")

            text_embed_slice_1 = embeds[:start_idx]
            image_feature_slice = image_features[i]
            text_embed_slice_2 = embeds[start_idx + 1:]

            print(f"  - Shape of text embeddings ('embeds'): {embeds.shape}")
            print(f"  - Shape of all image features ('image_features'): {image_features.shape}")
            print("-" * 20)
            print(f"  - Shape of text part 1 to concat: {text_embed_slice_1.shape}")
            print(f"  - Shape of image features to concat: {image_feature_slice.shape}")
            print(f"  - Shape of text part 2 to concat: {text_embed_slice_2.shape}")

            if text_embed_slice_1.shape[1] != image_feature_slice.shape[1]:
                print(f"  - ❌ MISMATCH! Feature dimension is different: {text_embed_slice_1.shape[1]} (text) vs {image_feature_slice.shape[1]} (image)")
            else:
                print("  - ✅ Dimension 1 (features) seems to match.")

            print("=" * 60 + "\n")

            spliced = torch.cat([text_embed_slice_1, image_feature_slice, text_embed_slice_2], dim=0)
            new_embeds.append(spliced)

            if labels is not None and labels[i] is not None:
                new_label = torch.full((image_feature_slice.shape[0],), IGNORE_INDEX, device=labels.device, dtype=torch.long)
                spliced_label = torch.cat([labels[i][:start_idx], new_label, labels[i][start_idx + 1:]], dim=0)
                new_labels.append(spliced_label)

        if not new_embeds:
            print("⚠️ [WARNING] A batch with no processable image tokens was encountered. Treating as text-only.")
            kwargs['input_ids'] = input_ids
            kwargs['labels'] = labels
            return LlamaForCausalLM.forward(self, **kwargs)

        inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_embeds, batch_first=True, padding_value=0.0)
        final_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX) if new_labels else None

        final_kwargs = kwargs.copy()
        final_kwargs.pop('gt_masks', None)
        final_kwargs['input_ids'] = None
        final_kwargs['inputs_embeds'] = inputs_embeds
        final_kwargs['labels'] = final_labels
        final_kwargs['attention_mask'] = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        return LlamaForCausalLM.forward(self, **final_kwargs)