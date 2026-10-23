import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class FullRefMetrics:
    def __init__(self, cfg, device):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(
            kernel_size=int(cfg.eval.ssim_window_size), data_range=1.0
        ).to(device)

    @staticmethod
    def rgb_to_y(x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def __call__(self, pred, target, use_y=True):
        if use_y:
            pred, target = self.rgb_to_y(pred), self.rgb_to_y(target)
        return self.psnr(pred, target).item(), self.ssim(pred, target).item()


def uciqe(img: np.ndarray) -> float:
    """UCIQE — Yang & Sowmya (2015).
    img: H×W×3 float32 in [0, 1], RGB order.
    Uses CIELab chroma std, luminance contrast, and mean chroma/L saturation.
    """
    # Manual sRGB → CIELab (D65, 2°)
    rgb = img.astype(np.float64)

    def linearize(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    lin = linearize(rgb)
    X = lin[..., 0] * 0.4124564 + lin[..., 1] * 0.3575761 + lin[..., 2] * 0.1804375
    Y = lin[..., 0] * 0.2126729 + lin[..., 1] * 0.7151522 + lin[..., 2] * 0.0721750
    Z = lin[..., 0] * 0.0193339 + lin[..., 1] * 0.1191920 + lin[..., 2] * 0.9503041

    # Normalize by D65 white point
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = float(np.std(chroma))
    con_l = float(np.percentile(L, 99) - np.percentile(L, 1))
    mu_s = float(np.mean(chroma / (L + 1e-5)))

    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


def uiqm(img: np.ndarray) -> float:
    """UIQM — Panetta, Gao & Agaian (2015).
    img: H×W×3 float32 in [0, 1], RGB order.
    UIQM = c1*UICM + c2*UISM + c3*UIConM
    """
    R, G, B = img[..., 0].astype(np.float64), img[..., 1].astype(np.float64), img[..., 2].astype(np.float64)

    # --- UICM: Underwater Image Colorfulness Measure ---
    RG = R - G
    YB = 0.5 * (R + G) - B
    mu_rg, sig_rg = np.mean(RG), np.std(RG)
    mu_yb, sig_yb = np.mean(YB), np.std(YB)
    uicm = -0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * np.sqrt(sig_rg ** 2 + sig_yb ** 2)

    # --- UISM: Underwater Image Sharpness Measure ---
    # EME-based: use gradient magnitude of each channel, average across channels
    def _eme_channel(c):
        gy, gx = np.gradient(c)
        return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    uism = (_eme_channel(R) + _eme_channel(G) + _eme_channel(B)) / 3.0

    # --- UIConM: Underwater Image Contrast Measure (log-AME) ---
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    eps = 1e-5
    uiconm = float(np.log(np.mean(gray) + eps) - np.mean(np.log(gray + eps)))

    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return float(c1 * uicm + c2 * uism + c3 * uiconm)
