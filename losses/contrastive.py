import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, enhanced, gt, raw):
        q = F.normalize(enhanced.flatten(1), dim=1)
        pos = F.normalize(gt.flatten(1), dim=1)
        neg = F.normalize(raw.flatten(1), dim=1)
        logits = torch.cat([torch.sum(q * pos, dim=1, keepdim=True), torch.sum(q * neg, dim=1, keepdim=True)], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
