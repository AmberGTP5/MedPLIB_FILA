# file: scripts/Stage3/model/grounding_model.py
import torch
import sys
from pathlib import Path

# --- 路径设置，确保能找到 Stage1 和 MedPLIB 的模块 ---
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)
STAGE1_ROOT = os.path.join(PROJECT_ROOT, "scripts", "Stage1")
sys.path.insert(0, STAGE1_ROOT)

# 导入 MedPLIB 中负责分割的核心模型 LISA
# 注意：这里我们假设您已将 MedPLIB-main 文件夹放置在项目根目录下
# 或者您需要将 MedPLIB-main 的路径也加入 sys.path
sys.path.insert(0, os.path.join(PROJECT_ROOT, "MedPLIB-main"))
from model.LISA import LISAForCausalLM, LISAConfig

# 导入我们第一阶段的模型，因为它包含了 HybridEncoder
from scripts.Stage1.model.fila_lisa import FILAForCausalLM as FILABaseModel


class UniMedVLMForGrounding(LISAForCausalLM):
    """
    我们的第三阶段模型。
    它继承了LISA的分割能力，并用我们自己的HybridEncoder替换了其原始的视觉编码器。
    """

    def __init__(self, config: LISAConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # 核心替换：用我们第一阶段的HybridEncoder替换掉LISA原有的vision_tower
        # 这一步将高分辨率能力注入到分割模型中
        self.get_model().vision_tower = FILABaseModel(config).get_model().vision_tower
        
    # 我们直接继承 LISAForCausalLM 的 forward 方法，它已经包含了计算分割损失的逻辑。
    # 它期望的输入中会包含 `masks_list` 和 `seg_flag=True` 等参数。
    # 我们的 grounding_dataset.py 脚本会负责准备这些特殊输入。