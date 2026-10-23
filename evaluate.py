import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from utils.metrics import FullRefMetrics, uciqe, uiqm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, required=True)
    return p.parse_args()


def load_img(path):
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = FullRefMetrics(cfg, device)

    rows = []
    rr = Path(args.results_dir)
    gg = Path(args.gt_dir)

    # Evaluate directory directly if it contains images without subdirectories
    method_dirs = [x for x in rr.iterdir() if x.is_dir()]
    if not method_dirs:
        method_dirs = [rr]

    for method in sorted(method_dirs):
        psnrs, ssims, uiqms, uciqes = [], [], [], []
        # Match input images
        input_images = list(method.glob("*.png")) + list(method.glob("*.jpg")) + list(method.glob("*.jpeg"))

        for ip in sorted(input_images):
            # Extract base image stem to match ground truth
            base_name = ip.name.replace("_enhanced.png", "").replace("_enhanced.jpg", "").replace(".png", "").replace(".jpg", "").replace(".jpeg", "")

            gt_p = gg / (base_name + ".png")
            if not gt_p.exists():
                gt_p = gg / (base_name + ".jpg")
            if not gt_p.exists():
                gt_p = gg / (base_name + ".jpeg")

            if not gt_p.exists():
                print(f"Warning: 没有给真实的 Ground Truth 找到匹配的图 {base_name}")
                continue
            pred = load_img(ip).to(device)
            gt = load_img(gt_p).to(device)

            if pred.shape != gt.shape:
                gt = torch.nn.functional.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)

            p, s = metric(pred, gt, use_y=bool(cfg.eval.y_channel_psnr))
            psnrs.append(p)
            ssims.append(s)
            n = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            uiqms.append(uiqm(n))
            uciqes.append(uciqe(n))
        if psnrs:
            rows.append((method.name, np.mean(psnrs), np.std(psnrs), np.mean(ssims), np.std(ssims), np.mean(uiqms), np.mean(uciqes)))

    print(r"\begin{tabular}{lcccccc}")
    print(r"Method & PSNR$\uparrow$ & PSNR std & SSIM$\uparrow$ & SSIM std & UIQM$\uparrow$ & UCIQE$\uparrow$ \\")
    print(r"\hline")
    for row in rows:
        print(f"{row[0]} & {row[1]:.3f} & {row[2]:.3f} & {row[3]:.4f} & {row[4]:.4f} & {row[5]:.3f} & {row[6]:.3f} \\\\")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
