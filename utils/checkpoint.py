import os
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metrics, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_metrics": best_metrics,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    model_state = model.state_dict()
    pretrained_state = ckpt.get("model", ckpt)
    filtered_state = {}
    for k, v in pretrained_state.items():
        if k in model_state and v.shape != model_state[k].shape:
            continue
        filtered_state[k] = v

    model.load_state_dict(filtered_state, strict=False)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
