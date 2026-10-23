import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()
