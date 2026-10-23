import torch
import torch.nn as nn
import torchvision


class VGGPerceptualLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features
        mapping = {"relu2_2": 8, "relu3_3": 15, "relu4_3": 24}
        self.layer_ids = [mapping[n] for n in list(cfg.loss.vgg_layers)]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x):
        return (x - self.mean) / self.std

    def _features(self, x):
        feats = []
        h = self._norm(x)
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layer_ids:
                feats.append(h)
            if i >= max(self.layer_ids):
                break
        return feats

    def forward(self, pred, target):
        fp, ft = self._features(pred), self._features(target)
        return sum((a - b).abs().mean() for a, b in zip(fp, ft))
