# file: scripts/Stage3/model/unimed_vlm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

# 导入我们自己的Stage1模型作为基底
from model.fila_lisa import FILAForCausalLM
# 导入MedPLIB中的SAM模型构造器
from model.segment_anything_med2d import build_sam_vit_b

def dice_loss(inputs, targets, eps=1e-6):
    """一个标准的Dice损失函数"""
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    intersection = (inputs * targets).sum(1)
    union = inputs.sum(1) + targets.sum(1)
    dice_score = (2. * intersection + eps) / (union + eps)
    return 1 - dice_score.mean()

class UniMedVLMForGrounding(FILAForCausalLM):
    """
    我们自己的、用于Grounding任务的最终模型。
    它以FILAForCausalLM为基底，并“组合”了SAM的分割能力。
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        # 1. 初始化并集成SAM-Med2D模型
        # 我们只保留它的掩码解码器和一部分提示编码器
        vision_pretrained = kwargs.get("vision_pretrained", None)
        sam_model = build_sam_vit_b(sam_checkpoint=vision_pretrained)
        self.visual_model = sam_model
        
        # 2. 初始化将语言特征投影到SAM解码器所能理解的维度的线性层
        llm_hidden_size = config.hidden_size
        sam_feature_dim = 256 # SAM解码器期望的特征维度
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(llm_hidden_size, sam_feature_dim)
        )
        self.seg_token_idx = kwargs.get("seg_token_idx")

    def forward(self, images_vit=None, images_convnext=None, input_ids=None, labels=None, gt_masks=None, **kwargs):
        # 首先，调用父类（FILAForCausalLM）的forward方法得到语言模型的输出
        # 我们需要修改父类的forward使其能返回hidden_states
        outputs = super().forward(
            images_vit=images_vit, images_convnext=images_convnext,
            input_ids=input_ids, labels=labels,
            output_hidden_states=True, return_dict=True
        )
        
        # 语言模型的loss (在Grounding任务中，labels通常被完全mask，所以此项为0)
        lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)
        
        # --- 开始计算分割损失 ---
        seg_loss = torch.tensor(0.0, device=input_ids.device)
        if gt_masks is not None:
            last_hidden_state = outputs.hidden_states[-1]
            
            # 找到<SEG> token的位置，并提取其对应的特征
            seg_token_mask = (input_ids == self.seg_token_idx)
            if torch.sum(seg_token_mask) > 0:
                seg_token_features = last_hidden_state[seg_token_mask]
                
                # 将语言特征投影为SAM的提示嵌入
                prompt_embeddings = self.text_hidden_fcs(seg_token_features)
                
                # 使用SAM解码器生成预测掩码
                with torch.no_grad():
                    # 这里我们只使用低分辨率的vit图像给SAM
                    image_embeddings = self.get_model().vision_tower.vision_tower.vision_tower.image_encoder(images_vit)

                low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=prompt_embeddings.unsqueeze(1),
                    dense_prompt_embeddings=self.visual_model.prompt_encoder.get_dense_pe(),
                    multimask_output=False,
                )
                
                predicted_mask_upsampled = F.interpolate(
                    low_res_masks, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False
                )
                
                # 计算Dice Loss + BCE Loss
                bce = F.binary_cross_entropy_with_logits(predicted_mask_upsampled.squeeze(1), gt_masks)
                dice = dice_loss(predicted_mask_upsampled.squeeze(1), gt_masks)
                seg_loss = bce + dice

        # 返回总损失
        total_loss = lm_loss + seg_loss 

        return CausalLMOutputWithPast(loss=total_loss, logits=outputs.logits)