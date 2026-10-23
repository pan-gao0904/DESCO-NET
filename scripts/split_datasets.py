import os
import random
import shutil
from pathlib import Path


def split_and_copy_dataset(dataset_name, src_dir, dst_train_dir, dst_val_dir, num_train, num_val, seed=42):
    """Split a dataset into train and validation sets, and copy files to target directories.

    Args:
        dataset_name: Name of the dataset for logging.
        src_dir: Root directory of the source dataset (must contain 'raw' and 'gt' subfolders).
        dst_train_dir: Target directory for training data.
        dst_val_dir: Target directory for validation data.
        num_train: Number of images for training.
        num_val: Number of images for validation.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)

    src_raw_dir = Path(src_dir) / 'raw'
    src_gt_dir = Path(src_dir) / 'gt'

    if not src_raw_dir.exists() or not src_gt_dir.exists():
        print(f"Warning: Source directories for {dataset_name} do not exist. Skipping...")
        return

    raw_files = {f.name for f in src_raw_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']}
    gt_files = {f.name for f in src_gt_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']}

    matched_names = list(raw_files.intersection(gt_files))
    matched_names.sort()
    random.shuffle(matched_names)

    total_found = len(matched_names)
    print(f"[{dataset_name}] Found {total_found} matched image pairs.")

    expected_total = num_train + num_val
    if total_found < expected_total:
        print(f"[{dataset_name}] ERROR: Not enough images! Needed {expected_total}, but only found {total_found}. Proceeding with maximum available.")
        num_train = min(num_train, total_found)
        num_val = min(num_val, total_found - num_train)

    train_names = matched_names[:num_train]
    val_names = matched_names[num_train : num_train + num_val]

    for dst_dir in [dst_train_dir, dst_val_dir]:
        (Path(dst_dir) / 'raw').mkdir(parents=True, exist_ok=True)
        (Path(dst_dir) / 'gt').mkdir(parents=True, exist_ok=True)

    def copy_files(names, dst_root):
        for name in names:
            src_raw_path = src_raw_dir / name
            src_gt_path = src_gt_dir / name

            dst_raw_path = Path(dst_root) / 'raw' / name
            dst_gt_path = Path(dst_root) / 'gt' / name

            shutil.copy2(src_raw_path, dst_raw_path)
            shutil.copy2(src_gt_path, dst_gt_path)

    copy_files(train_names, dst_train_dir)
    copy_files(val_names, dst_val_dir)

    print(f"[{dataset_name}] Success: Copied {len(train_names)} pairs to {dst_train_dir}")
    print(f"[{dataset_name}] Success: Copied {len(val_names)} pairs to {dst_val_dir}\n")


if __name__ == '__main__':
    # Configuration
    SEED = 2025

    DATA_ROOT = "data"
    UIEB_SRC = f"{DATA_ROOT}/UIEB"
    LSUI_SRC = f"{DATA_ROOT}/LSUI"

    TRAIN_MIXED_DIR = f"{DATA_ROOT}/dataset_mixed/train_mixed"
    VAL_UIEB_DIR = f"{DATA_ROOT}/dataset_mixed/val_uieb"
    VAL_LSUI_DIR = f"{DATA_ROOT}/dataset_mixed/val_lsui"

    print("=== 开始处理数据集划分 ===")

    # 1. Split UIEB (800 train, 90 val)
    split_and_copy_dataset(
        dataset_name="UIEB",
        src_dir=UIEB_SRC,
        dst_train_dir=TRAIN_MIXED_DIR,
        dst_val_dir=VAL_UIEB_DIR,
        num_train=800,
        num_val=90,
        seed=SEED,
    )

    # 2. Split LSUI (3879 train, 400 val)
    split_and_copy_dataset(
        dataset_name="LSUI",
        src_dir=LSUI_SRC,
        dst_train_dir=TRAIN_MIXED_DIR,
        dst_val_dir=VAL_LSUI_DIR,
        num_train=3879,
        num_val=400,
        seed=SEED,
    )

    print("=== 划分处理完成！ ===")
    print(f"请检查输出目录: {DATA_ROOT}/dataset_mixed/")
    print("你现在可以让你的 Dataloader 的训练路径指向 train_mixed 啦！")
