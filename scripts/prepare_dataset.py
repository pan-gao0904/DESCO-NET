import argparse
import csv
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare underwater dataset splits from merged paired folders."
    )
    parser.add_argument("--source_raw", type=str, default="data/source_merged/raw")
    parser.add_argument("--source_gt", type=str, default="data/source_merged/gt")
    parser.add_argument(
        "--source_transmission",
        type=str,
        default="data/source_merged/transmission",
        help="Optional. If folder missing or file missing, transmission is skipped per sample.",
    )
    parser.add_argument("--output_root", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="hardlink saves disk but needs filesystem support.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".png,.jpg,.jpeg,.tif,.tiff",
        help="Comma separated image extensions.",
    )
    return parser.parse_args()


def collect_images(folder, exts):
    if not folder.exists():
        return []
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def safe_link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def split_indices(n, train_ratio, val_ratio):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def main():
    args = parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    exts = set(x.strip().lower() for x in args.exts.split(",") if x.strip())
    source_raw = Path(args.source_raw)
    source_gt = Path(args.source_gt)
    source_tr = Path(args.source_transmission)
    output_root = Path(args.output_root)

    raw_files = collect_images(source_raw, exts)
    if not raw_files:
        raise RuntimeError(f"No source raw images found in: {source_raw}")

    samples = []
    missing_gt = []
    for raw_path in raw_files:
        gt_path = source_gt / raw_path.name
        tr_path = source_tr / (raw_path.stem + ".png")
        if not tr_path.exists():
            tr_path = source_tr / raw_path.name
            
        if not gt_path.exists():
            missing_gt.append(raw_path.name)
            continue
        samples.append(
            {
                "name": raw_path.name,
                "raw": raw_path,
                "gt": gt_path,
                "tr": tr_path if tr_path.exists() else None,
            }
        )

    if not samples:
        raise RuntimeError("No paired samples found (raw+gt).")

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n_train, n_val, n_test = split_indices(len(samples), args.train_ratio, args.val_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    split_map = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    for split in ["train", "val", "test"]:
        for sub in ["raw", "gt", "transmission"]:
            (output_root / split / sub).mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "filename", "has_transmission"])

        for split, items in split_map.items():
            for item in items:
                safe_link_or_copy(
                    item["raw"], output_root / split / "raw" / item["name"], args.copy_mode
                )
                safe_link_or_copy(
                    item["gt"], output_root / split / "gt" / item["name"], args.copy_mode
                )
                if item["tr"] is not None:
                    safe_link_or_copy(
                        item["tr"],
                        output_root / split / "transmission" / item["tr"].name,
                        args.copy_mode,
                    )
                writer.writerow([split, item["name"], int(item["tr"] is not None)])

    print("Dataset preparation complete.")
    print(f"Total paired samples: {len(samples)}")
    print(f"Train / Val / Test: {len(train_samples)} / {len(val_samples)} / {len(test_samples)}")
    print(f"Manifest: {manifest_path}")
    if missing_gt:
        print(f"Skipped {len(missing_gt)} raw files without GT.")


if __name__ == "__main__":
    main()
