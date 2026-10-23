import torch
import torch.nn as nn

from models.stage1.stage1_model import Stage1Model
from models.stage2.stage2_model import Stage2Model


class DUILFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stage1 = Stage1Model(cfg)
        self.stage2 = Stage2Model(cfg)

    def forward(self, x, run_stage1=True, run_stage2=True):
        outputs = {}
        if run_stage1:
            s1 = self.stage1(x)
            outputs.update(s1)
            j_in, t_in, b_in = s1["J"], s1["T"], s1["B"]
        else:
            j_in, t_in, b_in = x, None, None

        if run_stage2:
            s2 = self.stage2(j_in, t_in, b_in)
            outputs.update(s2)
        return outputs
