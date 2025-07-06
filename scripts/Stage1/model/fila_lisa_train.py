# file: model/fila_lisa.py
#
# FINAL DEBUGGING VERSION with complete methods and extensive tensor checks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from functools import partial
import os

from .medplib.model.language_model.medplib_llama import LlavaLlamaForCausalLM
from .medplib.model.multimodal_encoder.builder import build_vision_tower
from .medplib.model.multimodal_projector.builder import build_vision_projector
from utils.utils import IMAGE_TOKEN_INDEX, IGNORE_INDEX

# --- 调试函数 ---
def debug_tensor(tensor, name: str):
    if int(os.environ.get("RANK", 0)) != 0: return
    if not isinstance(tensor, torch.Tensor):
        print(f"DEBUG: '{name}' is not a tensor.")
        return
    
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    print("-" * 50)
    print(f"DEBUG CHECKING: Tensor '{name}'")
    
    if has_nan or has_inf:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! ALERT: Tensor '{name}' CONTAINS nan or inf !!!")
        print(f"    - Has NaN: {has_nan.item()}")
        print(f"    - Has Inf: {has_inf.item()}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise ValueError(f"NaN or Inf detected in tensor: {name}")
    else:
        print(f"  - Status: OK")
        print(f"  - Shape: {tensor.shape}, Dtype: {tensor.dtype}")
        print(f"  - Stats: Mean={tensor.mean().item():.4f}, Std={tensor.std().item():.4f}, Max={tensor.max().item():.4f}, Min={tensor.min().item():.4f}")
    print("-" * 50)

# --- 模型架构 ---

class CVFM(nn.Module):
    def __init__(self, vit_dim: int, convnext_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(vit_dim + convnext_dim, vit_dim * 4), nn.GELU(), nn.Linear(vit_dim * 4, vit_dim))
        self.gate = nn.Parameter(torch.zeros(1))
    def forward(self, vit_features: torch.Tensor, convnext_features: torch.Tensor) -> torch.Tensor:
        cls_token, patch_tokens = vit_features[:, :1, :], vit_features[:, 1:, :]
        vit_h = vit_w = int(patch_tokens.shape[1] ** 0.5)
        aligned_conv_feats = F.interpolate(convnext_features.to(patch_tokens.dtype), size=(vit_h, vit_w), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        fused_patch_tokens = patch_tokens + self.gate.tanh() * self.mlp(torch.cat([patch_tokens, aligned_conv_feats], dim=-1))
        return torch.cat([cls_token, fused_patch_tokens], dim=1)

class HybridEncoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.vision_tower = build_vision_tower(model_args)
        self.convnext_encoder = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
        self.convnext_encoder.classifier = nn.Identity()
        for param in self.vision_tower.parameters(): param.requires_grad = False
        for param in self.convnext_encoder.parameters(): param.requires_grad = False
        self.interaction_layers_idx = [2, 6, 12, 20]
        convnext_dims = [192, 384, 768, 1536]
        vit_dim = self.vision_tower.config.hidden_size
        self.cvfm_modules = nn.ModuleList([CVFM(vit_dim, convnext_dims[i]) for i in range(len(self.interaction_layers_idx))])
        for i, idx in enumerate(self.interaction_layers_idx):
            hook_fn = partial(self._cvfm_hook, cvfm_module_index=i)
            self.vision_tower.vision_tower.vision_model.encoder.layers[idx-1].register_forward_hook(hook_fn)
        self.convnext_features_stages = None

    def _forward_convnext_stages(self, x: torch.Tensor):
        """
        Helper to get features from all 4 stages of ConvNeXt.
        This version correctly follows torchvision's ConvNeXt structure.
        """
        features = []
        # features[0] is the stem
        x = self.convnext_encoder.features[0](x.to(self.dtype))
        
        # Stage 1
        x = self.convnext_encoder.features[1](x)
        # debug_tensor(x, 'ConvNeXt Stage 1 Output')
        features.append(x)
        
        # Stage 2
        x = self.convnext_encoder.features[2](x) # Downsampling
        x = self.convnext_encoder.features[3](x)
        # debug_tensor(x, 'ConvNeXt Stage 2 Output')
        features.append(x)
        
        # Stage 3
        x = self.convnext_encoder.features[4](x) # Downsampling
        x = self.convnext_encoder.features[5](x)
        # debug_tensor(x, 'ConvNeXt Stage 3 Output')
        features.append(x)
        
        # Stage 4
        x = self.convnext_encoder.features[6](x) # Downsampling
        x = self.convnext_encoder.features[7](x)
        # debug_tensor(x, 'ConvNeXt Stage 4 Output')
        features.append(x)
        
        return features

    def _cvfm_hook(self, module, input, output, cvfm_module_index):
        vit_features = output[0]
        # debug_tensor(vit_features, f'CVFM Hook Input (ViT Layer {self.interaction_layers_idx[cvfm_module_index]} Out)')
        conv_feats = self.convnext_features_stages[cvfm_module_index]
        fused_output = self.cvfm_modules[cvfm_module_index](vit_features, conv_feats)
        # debug_tensor(fused_output, f'CVFM Hook Fused Output {cvfm_module_index+1}')
        return (fused_output,) + output[1:]

    def forward(self, images_vit: torch.Tensor, images_convnext: torch.Tensor):
        # debug_tensor(images_vit, 'HybridEncoder Input (ViT)')
        # debug_tensor(images_convnext, 'HybridEncoder Input (ConvNeXt)')
        self.convnext_features_stages = self._forward_convnext_stages(images_convnext)
        vision_outputs = self.vision_tower.vision_tower(pixel_values=images_vit, output_hidden_states=True, return_dict=True)
        self.convnext_features_stages = None
        selected_features = vision_outputs.hidden_states[self.vision_tower.select_layer]
        if self.vision_tower.select_feature == "patch": image_features = selected_features[:, 1:]
        else: image_features = selected_features
        # debug_tensor(image_features, 'HybridEncoder Final Output')
        return image_features

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

class FILAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        super(LlavaLlamaForCausalLM, self).__init__(config)
        self.model.vision_tower = HybridEncoder(config)
        self.model.mm_projector = build_vision_projector(config)
        
    def get_vision_tower(self):
        return self.model.vision_tower

    def encode_images(self, images_vit, images_convnext):
        image_features = self.model.vision_tower(images_vit, images_convnext)
        # debug_tensor(image_features, 'Features from HybridEncoder (before projector)')
        projected_features = self.model.mm_projector(image_features)
        # debug_tensor(projected_features, 'Features after mm_projector')
        return projected_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, 
        images_vit, images_convnext
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or (images_vit is None and images_convnext is None) or input_ids.shape[1] == 1:
            return input_ids, attention_mask, past_key_values, None, labels
        
        image_features = self.encode_images(images_vit, images_convnext)
        
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_attention_mask = []
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
            if len(image_token_indices) > 0:
                image_token_start = image_token_indices[0]
                
                text_embeds_before = self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                attention_mask_before = attention_mask[batch_idx, :image_token_start]
                
                image_embeds = image_features[batch_idx]
                attention_mask_image = torch.ones(image_embeds.shape[0], device=attention_mask.device, dtype=attention_mask.dtype)

                text_embeds_after = self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:])
                attention_mask_after = attention_mask[batch_idx, image_token_start + 1:]

                cur_new_input_embeds = torch.cat([text_embeds_before, image_embeds, text_embeds_after], dim=0)
                new_attention_mask.append(torch.cat([attention_mask_before, attention_mask_image, attention_mask_after], dim=0))
                
                if labels is not None:
                    cur_labels = labels[batch_idx]
                    labels_before = cur_labels[:image_token_start]
                    labels_image = torch.full((image_embeds.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
                    labels_after = cur_labels[image_token_start + 1:]
                    cur_new_labels = torch.cat([labels_before, labels_image, labels_after], dim=0)
                    new_labels.append(cur_new_labels)
            else:
                cur_new_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_attention_mask.append(attention_mask[batch_idx])
                if labels is not None: new_labels.append(labels[batch_idx])

            new_input_embeds.append(cur_new_input_embeds)

        # Pad the sequences in the batch
        if new_input_embeds:
            inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_input_embeds, batch_first=True, padding_value=0)
            attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
            if labels is not None:
                labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        # debug_tensor(inputs_embeds, 'Final inputs_embeds before LLM')
        return None, attention_mask, past_key_values, inputs_embeds, labels

    def forward(self, **kwargs):
        images_vit = kwargs.pop('images_vit', None)
        images_convnext = kwargs.pop('images_convnext', None)
        input_ids = kwargs.pop('input_ids')
        labels = kwargs.pop('labels', None)
        attention_mask = kwargs.pop('attention_mask', None)

        ( _, new_attention_mask, past_key_values, inputs_embeds, new_labels) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, None, labels, images_vit, images_convnext
        )
        
        outputs = super(LlavaLlamaForCausalLM, self).forward(
            attention_mask=new_attention_mask, inputs_embeds=inputs_embeds, labels=new_labels, **kwargs
        )
        
        # # debug_tensor(outputs.logits, 'Logits from LLM')
        # if outputs.loss is not None:
        #     debug_tensor(outputs.loss, 'Final Calculated Loss')
        
        return outputs