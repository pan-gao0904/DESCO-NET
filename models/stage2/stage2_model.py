"""Stage 2 — Physics-Guided Affine Modulator (PGAM).

Architecture:
    Input: Concat(J, T, B) = 9 channels
    ├─ Global Branch:  GAP -> MLP -> scene-level color context
    ├─ Encoder:        NAFBlocks at full resolution
    ├─ Mid-1:          NAFBlocks at 1/2 resolution
    ├─ Mid-2:          NAFBlocks at 1/4 resolution
    ├─ Decoder:        Progressive upsampling + skip connections
    └─ Output Head:    Predict alpha (3ch), beta (3ch), r (3ch)

    Final output: O = clamp(alpha * J + beta + r, 0, 1)

Design rationale:
    After Stage 1 physics inversion, residual errors are primarily color
    bias and noise. PGAM predicts per-pixel affine parameters conditioned
    on the physical state (T, B) to perform color correction while
    preserving structural details from Stage 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies — replaces traditional activations."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Simplified NAF Block (ECCV 2022)."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1),
        )
        self.conv2 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm(x)
        y = self.conv1(y)
        y1, y2 = y.chunk(2, dim=1)
        y = y1 * y2
        y = self.dw(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        return x + y * self.beta


class Stage2Model(nn.Module):
    """Predicts per-pixel affine parameters alpha, beta and residual r for color correction."""

    def __init__(self, cfg):
        super().__init__()
        c = int(cfg.model.stage2.base_channels)
        n = int(cfg.model.stage2.num_blocks)

        # Input: J(3) + T(3) + B(3) = 9 channels
        self.in_proj = nn.Conv2d(9, c, 3, padding=1)

        # Global branch: scene-level color context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv2d(c, c, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 1),
        )

        # Encoder (full resolution)
        self.enc = nn.Sequential(*[NAFBlock(c) for _ in range(n)])

        # 1/2 scale
        self.down1 = nn.Conv2d(c, c * 2, 2, stride=2)
        self.mid1 = nn.Sequential(*[NAFBlock(c * 2) for _ in range(n)])

        # 1/4 scale
        self.down2 = nn.Conv2d(c * 2, c * 4, 2, stride=2)
        self.mid2 = nn.Sequential(*[NAFBlock(c * 4) for _ in range(n)])

        # Upsampling projections
        self.up2 = nn.Conv2d(c * 4, c * 2, 1)
        self.up1 = nn.Conv2d(c * 2, c, 1)

        # Decoder (full resolution with skip connection)
        self.dec = nn.Sequential(*[NAFBlock(c) for _ in range(n)])

        # Decoupled heads
        head_c = c // 2

        # Alpha and Beta predict low-frequency color context from 1/4 scale features
        self.head_alpha = nn.Sequential(
            nn.Conv2d(c * 4, head_c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_c, 3, 3, padding=1),
        )

        self.head_beta = nn.Sequential(
            nn.Conv2d(c * 4, head_c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_c, 3, 3, padding=1),
        )

        # Residual r is predicted at full resolution
        self.head_r = nn.Sequential(
            nn.Conv2d(c, head_c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_c, 3, 3, padding=1),
        )

    def forward(self, j, t, b):
        if t is None:
            t = torch.ones_like(j)
        if b is None:
            b = torch.zeros_like(j)

        inp = torch.cat([j, t, b], dim=1)
        x = self.in_proj(inp)

        # Global context injection
        g = self.global_mlp(self.global_pool(x))
        x = x + g

        # Multi-scale feature extraction
        enc = self.enc(x)
        m1 = self.mid1(self.down1(enc))
        m2 = self.mid2(self.down2(m1))

        # Progressive upsampling with skip connections
        up2 = self.up2(F.interpolate(m2, size=m1.shape[-2:], mode='bilinear', align_corners=False))
        m1_out = m1 + up2

        up1 = self.up1(F.interpolate(m1_out, size=enc.shape[-2:], mode='bilinear', align_corners=False))
        dec = self.dec(up1 + enc)

        # Frequency-decoupled parameter heads
        alpha_raw = self.head_alpha(m2)
        beta_raw = self.head_beta(m2)

        alpha_raw = F.interpolate(alpha_raw, size=j.shape[-2:], mode='bilinear', align_corners=False)
        beta_raw = F.interpolate(beta_raw, size=j.shape[-2:], mode='bilinear', align_corners=False)

        r_raw = self.head_r(dec)

        alpha = torch.sigmoid(alpha_raw) * 2.0
        beta = torch.tanh(beta_raw) * 0.5
        r = torch.tanh(r_raw) * 0.25

        # Physics-uncertainty gated residual
        u_map = 1.0 - t.mean(dim=1, keepdim=True)
        r_gated = r * u_map

        out = (alpha * j + beta + r_gated).clamp(0.0, 1.0)
        return {"O": out, "alpha": alpha, "beta": beta, "r": r_gated}
