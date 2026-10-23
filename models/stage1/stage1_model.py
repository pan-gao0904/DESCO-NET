"""Stage 1 — Physics-Anchored Underwater Image Restoration.

Forward pipeline:
    I -> MobileTBNet -> T [B, 3, H, W], B_field [B, 3, H, W]

Physics formulation:
    B     = B_field * (1 - T)
    J_raw = (I - B) / T
    J     = clamp(J_raw, 0, 1)
"""

import torch
import torch.nn as nn

from models.stage1.tb_net import MobileTBNet


class Stage1Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tb_net = MobileTBNet(pretrained=True)

    def forward(self, x: torch.Tensor) -> dict:
        T, b_field = self.tb_net(x)

        # Backscatter estimation from physical model
        B = b_field * (1.0 - T)

        # Physics-based scene radiance restoration
        J_raw = (x - B) / T
        J_clamp = J_raw.clamp(0.0, 1.0)

        return {"J": J_clamp, "J_raw": J_raw, "T": T, "B": B, "B_field": b_field}
