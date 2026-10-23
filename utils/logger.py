import csv
import os
from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    def __init__(self, log_dir, csv_path):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.tb = SummaryWriter(log_dir=log_dir)
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["epoch", "phase", "train_loss", "val_psnr", "val_ssim"])

    def log_batch_loss(self, loss, step):
        self.tb.add_scalar("train/loss_batch", float(loss), step)

    def log_epoch(self, epoch, phase, train_loss, val_psnr, val_ssim):
        self.tb.add_scalar("val/psnr", float(val_psnr), epoch)
        self.tb.add_scalar("val/ssim", float(val_ssim), epoch)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, phase, float(train_loss), float(val_psnr), float(val_ssim)])

    def log_images(self, raw, stage1, stage2, gt, epoch):
        self.tb.add_images("samples/raw", raw, epoch)
        self.tb.add_images("samples/stage1", stage1, epoch)
        self.tb.add_images("samples/stage2", stage2, epoch)
        self.tb.add_images("samples/gt", gt, epoch)
