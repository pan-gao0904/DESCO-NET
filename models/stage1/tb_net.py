import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""

    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class MobileTBNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.features = mobilenet.features

        # Background light branch
        self.b_field_branch = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Transmission map decoder branch
        self.up3 = UNetBlock(576 + 48, 128)
        self.up2 = UNetBlock(128 + 24, 64)
        self.up1 = UNetBlock(64 + 16, 32)
        self.up0 = UNetBlock(32, 16)
        self.up_fin = UNetBlock(16, 16)

        self.se = SEBlock(16)
        self.t_head = nn.Sequential(
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, _, H, W = x.shape

        f = x
        f1, f2, f3, f4 = None, None, None, None

        for i, layer in enumerate(self.features):
            f = layer(f)
            if i == 1:
                f1 = f
            elif i == 3:
                f2 = f
            elif i == 8:
                f3 = f
            elif i == 12:
                f4 = f

        # Background light prediction
        b_low = self.b_field_branch(f4)
        b_field = F.interpolate(b_low, size=(H, W), mode='bilinear', align_corners=False)

        # Transmission map prediction
        def upsample_concat(x, skip):
            x_up = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            return torch.cat([x_up, skip], dim=1)

        d3 = self.up3(upsample_concat(f4, f3))
        d2 = self.up2(upsample_concat(d3, f2))
        d1 = self.up1(upsample_concat(d2, f1))

        d0 = self.up0(F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False))
        d_fin = self.up_fin(F.interpolate(d0, size=(H, W), mode='bilinear', align_corners=False))

        d_fin = self.se(d_fin)

        t_raw = self.t_head(d_fin)
        T_map = torch.clamp(t_raw, min=0.05, max=1.0)

        return T_map, b_field
