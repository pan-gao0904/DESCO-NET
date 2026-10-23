import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from omegaconf import OmegaConf
from PIL import Image

from models.duilf_net import DUILFNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--mode", type=str, required=True, choices=["full", "stage1", "stage2"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--input_restored", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_stage1", action="store_true")
    p.add_argument("--save_transmission", action="store_true")
    p.add_argument("--save_backscatter", action="store_true")
    p.add_argument("--resize", type=int, default=None, help="Resize input resolution for inference.")
    return p.parse_args()


def load_img(path):
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_img(t, path):
    arr = (t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_gray(t, path):
    """Save single-channel map as grayscale image."""
    arr = (t.squeeze(0).squeeze(0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def save_heatmap(t, path):
    """Save tensor as a JET heatmap."""
    if t.shape[1] > 1:
        t = t.mean(dim=1, keepdim=True)

    arr = t.squeeze().detach().cpu().numpy().clip(0, 1)
    arr_uint8 = (arr * 255.0).astype(np.uint8)

    heatmap = cv2.applyColorMap(arr_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    Image.fromarray(heatmap_rgb).save(path)


def pad_to_multiple(x, k):
    h, w = x.shape[-2:]
    ph = (k - h % k) % k
    pw = (k - w % k) % k
    y = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return y, (h, w)


def unpad(x, shape):
    h, w = shape
    return x[..., :h, :w]


def iter_inputs(inp):
    p = Path(inp)
    if p.is_dir():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            for f in p.glob(ext):
                yield f
    else:
        yield p


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = DUILFNet(cfg).to(device).eval()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pad_mul = int(cfg.inference.pad_multiple)

    for p in iter_inputs(args.input):
        x = load_img(p).to(device)

        if args.resize is not None:
            x = torch.nn.functional.interpolate(x, size=(args.resize, args.resize), mode="bilinear", align_corners=False)

        x, hw = pad_to_multiple(x, pad_mul)
        with torch.no_grad():
            if args.mode == "full":
                out = model(x, run_stage1=True, run_stage2=True)
                y = out["O"]
            elif args.mode == "stage1":
                out = model(x, run_stage1=True, run_stage2=False)
                y = out["J"]
            else:
                xr = load_img(args.input_restored).to(device) if args.input_restored else x
                xr, _ = pad_to_multiple(xr, pad_mul)
                s2 = model.stage2(xr, None, None)
                out = {"O": s2["O"]}
                y = out["O"]

        y = unpad(y, hw)
        save_img(y, out_dir / f"{p.stem}.png")
        if args.save_stage1 and "J" in out:
            save_img(unpad(out["J"], hw), out_dir / f"{p.stem}_stage1.png")
        if args.save_transmission and "T" in out:
            save_heatmap(unpad(out["T"], hw), out_dir / f"{p.stem}_transmission.png")
        if args.save_backscatter and "B" in out:
            save_img(unpad(out["B"], hw), out_dir / f"{p.stem}_backscatter.png")


if __name__ == "__main__":
    main()
