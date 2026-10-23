"""DUILF-Net training script.

Usage:
    python train.py --phase 1                                                # Stage 1: T/B physics inversion
    python train.py --phase 2  --resume outputs/checkpoints/phase1_best.pth  # Stage 2: PGAM color correction
    python train.py --phase 3  --resume outputs/checkpoints/phase2_best.pth  # End-to-end fine-tuning
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

from data.dataset import UnderwaterDataset
from losses.charbonnier import CharbonnierLoss
from losses.edge import tv_loss
from models.duilf_net import DUILFNet
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import TrainLogger
from utils.metrics import FullRefMetrics


def parse_args():
    p = argparse.ArgumentParser(description="DUILF-Net training")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--phase", type=str, required=True, choices=["1", "2", "3"])
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device, cfg, phase, metric_fn, amp_enabled):
    model.eval()
    psnrs, ssims = [], []
    sample_batch = None

    for batch in loader:
        raw = batch["raw"].to(device)
        gt = batch["gt"].to(device)

        with autocast(enabled=amp_enabled):
            if phase == "1":
                s1 = model.stage1(raw)
                pred = s1["J"]
            elif phase == "2":
                s1 = model.stage1(raw)
                s2 = model.stage2(s1["J"], s1["T"], s1["B"])
                pred = s2["O"]
            else:  # phase 3
                out = model(raw)
                pred = out["O"]

        p, s = metric_fn(pred.float(), gt.float(), use_y=bool(cfg.eval.y_channel_psnr))
        psnrs.append(p)
        ssims.append(s)

        if sample_batch is None:
            with autocast(enabled=amp_enabled):
                s1_vis = model.stage1(raw[:4])
            sample_batch = {
                "raw": raw[:4].cpu(),
                "stage1": s1_vis["J"][:4].cpu(),
                "pred": pred[:4].cpu(),
                "gt": gt[:4].cpu(),
            }

    model.train()
    val_psnr = sum(psnrs) / len(psnrs) if psnrs else 0.0
    val_ssim = sum(ssims) / len(ssims) if ssims else 0.0
    return val_psnr, val_ssim, sample_batch


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    seed = int(cfg.project.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.train.mixed_precision) and device.type == "cuda"

    model = DUILFNet(cfg).to(device)

    phase_key = f"phase{args.phase}"
    ph = cfg.train[phase_key]
    lr = float(ph.lr)

    # DataLoaders
    loader_kwargs = dict(
        batch_size=int(ph.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=bool(cfg.data.persistent_workers) and int(cfg.data.num_workers) > 0,
    )
    train_dataset = UnderwaterDataset(cfg, split="train")
    val_dataset = UnderwaterDataset(cfg, split="val")
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Optimizer
    betas = tuple(cfg.train.optimizer.betas)
    wd = float(cfg.train.optimizer.weight_decay)

    if args.phase == "3":
        optimizer = AdamW(
            [
                {"params": model.stage1.parameters(), "lr": lr * float(ph.stage1_lr_ratio)},
                {"params": model.stage2.parameters(), "lr": lr},
            ],
            betas=betas, weight_decay=wd,
        )
    elif args.phase == "1":
        optimizer = AdamW(model.stage1.parameters(), lr=lr, betas=betas, weight_decay=wd)
    else:  # phase 2 — freeze stage1
        for p in model.stage1.parameters():
            p.requires_grad = False
        optimizer = AdamW(model.stage2.parameters(), lr=lr, betas=betas, weight_decay=wd)

    scheduler = CosineAnnealingLR(optimizer, T_max=int(ph.epochs))
    scaler = GradScaler(enabled=amp_enabled)

    # Loss functions
    charbonnier = CharbonnierLoss(cfg.loss.charbonnier_eps).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    lam_ssim = float(cfg.loss.lambda_ssim)
    lam_ssim_out = float(cfg.loss.lambda_ssim_out)
    lam_tv = float(cfg.loss.lambda_tv)
    clip_norm = float(cfg.train.grad_clip_norm)

    def _y(x):
        """RGB to Y (luminance) channel."""
        return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

    def sobel_edge_loss(pred_y, gt_y):
        """Sobel edge loss for high-frequency detail preservation."""
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3) / 4.0
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3) / 4.0
        pred_dx = torch.nn.functional.conv2d(pred_y, kx, padding=0)
        pred_dy = torch.nn.functional.conv2d(pred_y, ky, padding=0)
        gt_dx = torch.nn.functional.conv2d(gt_y, kx, padding=0)
        gt_dy = torch.nn.functional.conv2d(gt_y, ky, padding=0)
        return torch.abs(pred_dx - gt_dx).mean() + torch.abs(pred_dy - gt_dy).mean()

    def _uv(x):
        """RGB to UV chrominance channels."""
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return torch.cat([u, v], dim=1)

    def get_gradient(img):
        """Compute spatial gradients."""
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        return dx, dy

    # Checkpoint resume
    start_epoch = 0
    best = {"psnr": 0.0, "ssim": 0.0}
    if args.resume:
        is_same_phase = os.path.basename(args.resume).startswith(f"phase{args.phase}")
        opt_to_load = optimizer if is_same_phase else None
        sch_to_load = scheduler if is_same_phase else None

        ckpt = load_checkpoint(args.resume, model, opt_to_load, sch_to_load, map_location=device)
        if is_same_phase:
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best = ckpt.get("best_metrics", best)
            print(f"Resumed Phase {args.phase} from epoch {start_epoch - 1}.")
        else:
            print(f"Loaded pretrained weights from {args.resume} for Phase {args.phase}.")

    # Logger & Metrics
    os.makedirs(str(cfg.project.save_dir), exist_ok=True)
    logger = TrainLogger(
        log_dir=os.path.join(str(cfg.project.log_dir), f"phase{args.phase}"),
        csv_path=os.path.join(str(cfg.project.log_dir), f"phase{args.phase}_metrics.csv"),
    )
    val_metrics = FullRefMetrics(cfg, device)

    epochs = int(ph.epochs)
    global_step = start_epoch * len(train_loader)
    ckpt_every = int(cfg.train.checkpoint_every_epochs)
    img_every = int(cfg.train.image_log_every_epochs)

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_valid = 0

        for batch in train_loader:
            raw = batch["raw"].to(device)
            gt = batch["gt"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                if args.phase == "1":
                    s1 = model.stage1(raw)
                elif args.phase == "2":
                    with torch.no_grad():
                        s1 = model.stage1(raw)
                    s2 = model.stage2(s1["J"], s1["T"], s1["B"])
                else:  # phase 3
                    out = model(raw)

            # Loss computation
            if args.phase == "1":
                jp_raw = s1["J_raw"].float()
                jp_clamp = s1["J"].float()
                tp = s1["T"].float()
                b_field_pred = s1["B_field"].float()
                gg = gt.float()
                rr = raw.float()

                j_y = _y(jp_clamp)
                g_y = _y(gg)

                # Structure and luminance constraints
                loss_ssim = 1.0 - ssim_fn(j_y, g_y)
                loss_luma = charbonnier(j_y, g_y)

                # Gradient matching loss
                dx_j, dy_j = get_gradient(jp_clamp)
                dx_g, dy_g = get_gradient(gg)
                loss_grad = torch.abs(dx_j - dx_g).mean() + torch.abs(dy_j - dy_g).mean()

                # Edge-aware transmission smoothness
                dx_t, dy_t = get_gradient(tp)
                rr_y = _y(rr)
                dx_r, dy_r = get_gradient(rr_y)
                weight_x = torch.exp(-10.0 * torch.abs(dx_r))
                weight_y = torch.exp(-10.0 * torch.abs(dy_r))

                loss_t_smooth = (torch.abs(dx_t) * weight_x).mean() + (torch.abs(dy_t) * weight_y).mean()

                loss = lam_ssim * loss_ssim + 1.0 * loss_luma + 1.0 * loss_grad + 0.1 * loss_t_smooth

            elif args.phase == "2":
                pf = s2["O"].float()
                gf = gt.float()
                jp = s1["J"].float()

                # Gradient conservation loss to preserve Stage 1 physical structure
                dx_o, dy_o = get_gradient(_y(pf))
                dx_j, dy_j = get_gradient(_y(jp))
                loss_cons = torch.abs(dx_o - dx_j.detach()).mean() + torch.abs(dy_o - dy_j.detach()).mean()

                loss = (
                    charbonnier(_uv(pf), _uv(gf))
                    + lam_ssim_out * (1.0 - ssim_fn(_y(pf), _y(gf)))
                    + lam_tv * (tv_loss(s2["alpha"].float()) + tv_loss(s2["beta"].float()))
                    + 0.5 * loss_cons
                )

            else:  # phase 3
                jp = out["J"].float()
                op = out["O"].float()
                gf = gt.float()

                y_j = _y(jp)
                y_o = _y(op)
                y_g = _y(gf)

                # Chrominance and luminance alignment
                loss_color = charbonnier(_uv(op), _uv(gf))
                loss_luma_l1 = charbonnier(y_o, y_g)
                loss_luma_ssim = lam_ssim_out * (1.0 - ssim_fn(y_o, y_g))

                # Edge sharpness alignment
                loss_edge = 0.5 * sobel_edge_loss(y_o, y_g)

                # Physical consistency regularization
                loss_anchor = 0.1 * (charbonnier(y_j, y_g) + lam_ssim * (1.0 - ssim_fn(y_j, y_g)))

                # Affine parameter smoothness regularization
                loss_tv_flow = lam_tv * (tv_loss(out["alpha"].float()) + tv_loss(out["beta"].float()))

                loss = loss_color + loss_luma_l1 + loss_luma_ssim + loss_edge + loss_anchor + loss_tv_flow

            # NaN protection: skip corrupted batch
            if torch.isnan(loss) or torch.isinf(loss):
                del loss
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_valid += 1
            logger.log_batch_loss(loss.item(), global_step)
            global_step += 1

        scheduler.step()
        epoch_loss /= max(n_valid, 1)

        # Free fragmented GPU memory between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validation
        val_psnr, val_ssim, samples = validate(
            model, val_loader, device, cfg, args.phase, val_metrics, amp_enabled
        )

        is_best = val_psnr > best["psnr"]
        if is_best:
            best["psnr"] = val_psnr
            best["ssim"] = val_ssim
            save_checkpoint(
                os.path.join(str(cfg.project.save_dir), f"phase{args.phase}_best.pth"),
                model, optimizer, scheduler, epoch, best,
                OmegaConf.to_container(cfg, resolve=True),
            )

        logger.log_epoch(epoch, args.phase, epoch_loss, val_psnr, val_ssim)

        if (epoch + 1) % img_every == 0 and samples is not None:
            logger.log_images(samples["raw"], samples["stage1"], samples["pred"], samples["gt"], epoch)

        if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == epochs:
            save_checkpoint(
                os.path.join(str(cfg.project.save_dir), f"phase{args.phase}_epoch{epoch + 1}.pth"),
                model, optimizer, scheduler, epoch, best,
                OmegaConf.to_container(cfg, resolve=True),
            )

        print(
            f"[Phase {args.phase}] Epoch {epoch + 1}/{epochs} | "
            f"loss={epoch_loss:.4f} | val_psnr={val_psnr:.3f} | val_ssim={val_ssim:.4f}"
        )


if __name__ == "__main__":
    main()
