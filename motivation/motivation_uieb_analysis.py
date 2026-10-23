# -*- coding: utf-8 -*-
"""
Motivation analysis for underwater image enhancement:
Quantitatively verify that structure is more stable than color
across multiple enhancement candidates.
"""

import os
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wilcoxon


# -----------------------------
# Utility
# -----------------------------
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_file_map(folder):
    folder = Path(folder)
    file_map = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            file_map[p.stem] = str(p)
    return file_map


def intersect_keys(dicts):
    keys = None
    for d in dicts:
        ks = set(d.keys())
        keys = ks if keys is None else (keys & ks)
    return sorted(list(keys))


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def rgb_to_lab_cached(rgb):
    return rgb2lab(rgb)


def grad_mag(L):
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag


def edge_map_from_L(L, q=0.85):
    mag = grad_mag(L)
    thr = np.quantile(mag, q)
    edge = (mag >= thr).astype(np.uint8)
    return edge


def edge_f1_score(e1, e2, eps=1e-8):
    e1 = e1.astype(bool)
    e2 = e2.astype(bool)
    tp = np.logical_and(e1, e2).sum()
    fp = np.logical_and(~e1, e2).sum()
    fn = np.logical_and(e1, ~e2).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1


def safe_corrcoef(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 1.0
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        c = 1.0
    return float(c)


def circular_hue_diff_deg(h1, h2):
    # h1,h2 in radians
    d = np.angle(np.exp(1j * (h1 - h2)))  # [-pi, pi]
    d = np.abs(d)
    return np.degrees(d)


def pair_metrics_from_lab(lab1, lab2):
    # Structure metrics on L
    L1 = lab1[..., 0] / 100.0
    L2 = lab2[..., 0] / 100.0

    l_ssim = ssim(L1, L2, data_range=1.0)

    g1 = grad_mag(L1)
    g2 = grad_mag(L2)
    grad_corr = safe_corrcoef(g1, g2)

    e1 = edge_map_from_L(L1)
    e2 = edge_map_from_L(L2)
    edge_f1 = edge_f1_score(e1, e2)

    # Color metrics
    de00 = float(deltaE_ciede2000(lab1, lab2).mean())

    a1, b1 = lab1[..., 1], lab1[..., 2]
    a2, b2 = lab2[..., 1], lab2[..., 2]
    ab_dist = float(np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2).mean())

    h1 = np.arctan2(b1, a1)
    h2 = np.arctan2(b2, a2)
    hue_diff_deg = float(circular_hue_diff_deg(h1, h2).mean())

    return {
        "l_ssim": float(l_ssim),
        "grad_corr": float(grad_corr),
        "edge_f1": float(edge_f1),
        "de00": de00,
        "ab_dist": ab_dist,
        "hue_diff_deg": hue_diff_deg
    }


def minmax_norm(series):
    arr = np.asarray(series, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def zscore(series):
    arr = np.asarray(series, dtype=np.float64)
    mu, sd = arr.mean(), arr.std()
    if sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


# -----------------------------
# Visualization
# -----------------------------
def save_metric_figure(image_df, out_png, out_pdf):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- boxplot: structure ----
    ax = axes[0, 0]
    structure_data = [
        image_df["l_ssim_mean"].values,
        image_df["grad_corr_mean"].values,
        image_df["edge_f1_mean"].values,
    ]
    ax.boxplot(structure_data, labels=["L-SSIM", "Grad-Corr", "Edge-F1"], showfliers=False)
    ax.set_title("Structure consistency across enhancement candidates")
    ax.set_ylabel("Higher is better")
    ax.grid(alpha=0.25)

    # ---- boxplot: color ----
    ax = axes[0, 1]
    color_data = [
        image_df["de00_mean"].values,
        image_df["ab_dist_mean"].values,
        image_df["hue_diff_deg_mean"].values,
    ]
    ax.boxplot(color_data, labels=[r"$\Delta E_{00}$", "ab-dist", "Hue-diff"], showfliers=False)
    ax.set_title("Color disagreement across enhancement candidates")
    ax.set_ylabel("Higher means larger disagreement")
    ax.grid(alpha=0.25)

    # ---- scatter: structure vs color ----
    ax = axes[1, 0]
    ax.scatter(
        image_df["structure_consistency_score"].values,
        image_df["color_disagreement_score"].values,
        s=12, alpha=0.55
    )
    ax.set_xlabel("Structure consistency score (higher better)")
    ax.set_ylabel("Color disagreement score (higher larger)")
    ax.set_title("Per-image structure stability vs color disagreement")
    ax.grid(alpha=0.25)

    # ---- normalized disagreement comparison ----
    ax = axes[1, 1]
    comp_data = [
        image_df["structure_disagreement_norm"].values,
        image_df["color_disagreement_norm"].values
    ]
    ax.boxplot(comp_data, labels=["Structure disagreement", "Color disagreement"], showfliers=False)
    ax.set_title("Normalized disagreement comparison")
    ax.set_ylabel("0-1 normalized disagreement")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_visual_panel(selected_df, file_maps, out_png, out_pdf, panel_type="color"):
    """
    panel_type:
        - color : raw + ref + 4 methods
        - edge  : edge maps of ref + 4 methods
    """
    methods_order = ["Reference", "CLUIE-Net", "DeepWaveNet", "FUnIE-GAN", "Shallow-UWNet"]

    if panel_type == "color":
        titles = ["Raw", "Reference", "CLUIE-Net", "DeepWaveNet", "FUnIE-GAN", "Shallow-UWNet"]
        ncols = 6
    else:
        titles = ["Reference-edge", "CLUIE-edge", "DeepWave-edge", "FUnIE-edge", "Shallow-edge"]
        ncols = 5

    nrows = len(selected_df)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows))

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (_, row) in enumerate(selected_df.iterrows()):
        stem = row["image_id"]

        if panel_type == "color":
            show_list = [
                ("Raw", file_maps["Raw"][stem]),
                ("Reference", file_maps["Reference"][stem]),
                ("CLUIE-Net", file_maps["CLUIE-Net"][stem]),
                ("DeepWaveNet", file_maps["DeepWaveNet"][stem]),
                ("FUnIE-GAN", file_maps["FUnIE-GAN"][stem]),
                ("Shallow-UWNet", file_maps["Shallow-UWNet"][stem]),
            ]
            for c, (name, path) in enumerate(show_list):
                ax = axes[r, c]
                rgb = read_rgb(path)
                ax.imshow(rgb)
                ax.axis("off")
                if r == 0:
                    ax.set_title(titles[c], fontsize=10)
                if c == 0:
                    text = (
                        f"{stem}\n"
                        f"S={row['structure_consistency_score']:.3f}\n"
                        f"C={row['color_disagreement_score']:.3f}"
                    )
                    ax.set_ylabel(text, fontsize=9)
        else:
            show_list = [
                ("Reference", file_maps["Reference"][stem]),
                ("CLUIE-Net", file_maps["CLUIE-Net"][stem]),
                ("DeepWaveNet", file_maps["DeepWaveNet"][stem]),
                ("FUnIE-GAN", file_maps["FUnIE-GAN"][stem]),
                ("Shallow-UWNet", file_maps["Shallow-UWNet"][stem]),
            ]
            for c, (name, path) in enumerate(show_list):
                ax = axes[r, c]
                rgb = read_rgb(path)
                lab = rgb_to_lab_cached(rgb)
                L = lab[..., 0] / 100.0
                edge = edge_map_from_L(L)
                ax.imshow(edge, cmap="gray")
                ax.axis("off")
                if r == 0:
                    ax.set_title(titles[c], fontsize=10)
                if c == 0:
                    text = (
                        f"{stem}\n"
                        f"S={row['structure_consistency_score']:.3f}\n"
                        f"C={row['color_disagreement_score']:.3f}"
                    )
                    ax.set_ylabel(text, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main(args):
    ensure_dir(args.out_dir)

    # Build file maps
    file_maps = {
        "Raw": build_file_map(args.raw_dir),
        "Reference": build_file_map(args.ref_dir),
        "CLUIE-Net": build_file_map(args.cluie_dir),
        "DeepWaveNet": build_file_map(args.deepwave_dir),
        "FUnIE-GAN": build_file_map(args.funie_dir),
        "Shallow-UWNet": build_file_map(args.shallow_dir),
    }

    common_ids = intersect_keys(file_maps.values())
    if len(common_ids) == 0:
        raise RuntimeError("No common image ids found across all folders.")

    print(f"[INFO] Number of matched images: {len(common_ids)}")

    # Enhancement candidates used for pairwise quantitative analysis
    candidate_names = ["Reference", "CLUIE-Net", "DeepWaveNet", "FUnIE-GAN", "Shallow-UWNet"]

    pairwise_rows = []
    image_rows = []

    for stem in tqdm(common_ids, desc="Analyzing images"):
        labs = {}
        for name in candidate_names:
            rgb = read_rgb(file_maps[name][stem])
            labs[name] = rgb_to_lab_cached(rgb)

        # all pairwise metrics among candidates
        pairs = list(combinations(candidate_names, 2))
        img_pair_metrics = []

        for a, b in pairs:
            metrics = pair_metrics_from_lab(labs[a], labs[b])
            row = {
                "image_id": stem,
                "method_a": a,
                "method_b": b,
                **metrics
            }
            pairwise_rows.append(row)
            img_pair_metrics.append(metrics)

        # aggregate image-level metrics
        img_df = pd.DataFrame(img_pair_metrics)
        l_ssim_mean = img_df["l_ssim"].mean()
        grad_corr_mean = img_df["grad_corr"].mean()
        edge_f1_mean = img_df["edge_f1"].mean()
        de00_mean = img_df["de00"].mean()
        ab_dist_mean = img_df["ab_dist"].mean()
        hue_diff_mean = img_df["hue_diff_deg"].mean()

        image_rows.append({
            "image_id": stem,
            "l_ssim_mean": l_ssim_mean,
            "grad_corr_mean": grad_corr_mean,
            "edge_f1_mean": edge_f1_mean,
            "de00_mean": de00_mean,
            "ab_dist_mean": ab_dist_mean,
            "hue_diff_deg_mean": hue_diff_mean,
        })

    pairwise_df = pd.DataFrame(pairwise_rows)
    image_df = pd.DataFrame(image_rows)

    # Add summary scores
    image_df["grad_corr_01"] = (image_df["grad_corr_mean"] + 1.0) / 2.0
    image_df["structure_consistency_score"] = (
        image_df["l_ssim_mean"] +
        image_df["grad_corr_01"] +
        image_df["edge_f1_mean"]
    ) / 3.0

    # normalized disagreement scores
    structure_dis_lssim = minmax_norm(1.0 - image_df["l_ssim_mean"])
    structure_dis_grad = minmax_norm(1.0 - image_df["grad_corr_01"])
    structure_dis_edge = minmax_norm(1.0 - image_df["edge_f1_mean"])

    color_dis_de00 = minmax_norm(image_df["de00_mean"])
    color_dis_ab = minmax_norm(image_df["ab_dist_mean"])
    color_dis_hue = minmax_norm(image_df["hue_diff_deg_mean"])

    image_df["structure_disagreement_norm"] = (
        structure_dis_lssim + structure_dis_grad + structure_dis_edge
    ) / 3.0

    image_df["color_disagreement_norm"] = (
        color_dis_de00 + color_dis_ab + color_dis_hue
    ) / 3.0

    image_df["color_disagreement_score"] = image_df["color_disagreement_norm"]

    # select representative images:
    # high structure consistency + high color disagreement
    image_df["selection_score"] = (
        zscore(image_df["structure_consistency_score"]) +
        zscore(image_df["color_disagreement_score"])
    )
    selected_df = image_df.sort_values("selection_score", ascending=False).head(args.num_visuals).copy()

    # dataset summary
    summary_items = []
    metric_cols = [
        "l_ssim_mean", "grad_corr_mean", "edge_f1_mean",
        "de00_mean", "ab_dist_mean", "hue_diff_deg_mean",
        "structure_consistency_score",
        "structure_disagreement_norm",
        "color_disagreement_norm"
    ]

    for col in metric_cols:
        vals = image_df[col].values
        summary_items.append({
            "metric": col,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    summary_df = pd.DataFrame(summary_items)

    # Statistical test
    stat, pval = wilcoxon(
        image_df["color_disagreement_norm"].values,
        image_df["structure_disagreement_norm"].values,
        alternative="greater"
    )

    # Save tables
    pairwise_csv = os.path.join(args.out_dir, "pairwise_metrics.csv")
    image_csv = os.path.join(args.out_dir, "image_level_metrics.csv")
    summary_csv = os.path.join(args.out_dir, "dataset_summary.csv")
    selected_csv = os.path.join(args.out_dir, "selected_examples.csv")

    pairwise_df.to_csv(pairwise_csv, index=False, encoding="utf-8-sig")
    image_df.to_csv(image_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    # Save figures
    save_metric_figure(
        image_df,
        out_png=os.path.join(args.out_dir, "motivation_stats.png"),
        out_pdf=os.path.join(args.out_dir, "motivation_stats.pdf")
    )

    make_visual_panel(
        selected_df,
        file_maps,
        out_png=os.path.join(args.out_dir, "motivation_visual_panel.png"),
        out_pdf=os.path.join(args.out_dir, "motivation_visual_panel.pdf"),
        panel_type="color"
    )

    make_visual_panel(
        selected_df,
        file_maps,
        out_png=os.path.join(args.out_dir, "motivation_edge_panel.png"),
        out_pdf=os.path.join(args.out_dir, "motivation_edge_panel.pdf"),
        panel_type="edge"
    )

    # Save summary report
    report_path = os.path.join(args.out_dir, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Motivation Analysis Report ===\n\n")
        f.write(f"Matched images: {len(common_ids)}\n")
        f.write(f"Pairwise candidates: {candidate_names}\n")
        f.write(f"Total pairwise comparisons per image: {len(list(combinations(candidate_names, 2)))}\n\n")

        f.write("Image-level summary:\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"{row['metric']}: "
                f"mean={row['mean']:.4f}, std={row['std']:.4f}, "
                f"median={row['median']:.4f}, q25={row['q25']:.4f}, q75={row['q75']:.4f}\n"
            )

        f.write("\n")
        f.write("Wilcoxon signed-rank test:\n")
        f.write("H1: normalized color disagreement > normalized structure disagreement\n")
        f.write(f"statistic = {stat:.4f}\n")
        f.write(f"p-value   = {pval:.6e}\n\n")

        f.write("Core quantities:\n")
        f.write(
            f"Mean structure consistency score = {image_df['structure_consistency_score'].mean():.4f}\n"
        )
        f.write(
            f"Mean normalized structure disagreement = {image_df['structure_disagreement_norm'].mean():.4f}\n"
        )
        f.write(
            f"Mean normalized color disagreement     = {image_df['color_disagreement_norm'].mean():.4f}\n"
        )
        f.write(
            f"Mean disagreement gap (color - structure) = "
            f"{(image_df['color_disagreement_norm'] - image_df['structure_disagreement_norm']).mean():.4f}\n"
        )

        f.write("\nInterpretation suggestion:\n")
        f.write(
            "If color_disagreement_norm is consistently higher than structure_disagreement_norm,\n"
            "and the Wilcoxon test is significant, then the results support the claim that\n"
            "different enhancement candidates preserve scene structure more consistently\n"
            "than they agree on chromatic restoration.\n"
        )

    print(f"[DONE] Results saved to: {args.out_dir}")
    print(f"[DONE] Key files:")
    print(f"  - {pairwise_csv}")
    print(f"  - {image_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {os.path.join(args.out_dir, 'motivation_stats.png')}")
    print(f"  - {os.path.join(args.out_dir, 'motivation_visual_panel.png')}")
    print(f"  - {os.path.join(args.out_dir, 'motivation_edge_panel.png')}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_dir",
        type=str,
        default=r"F:\A001\02research\0002code\02_datasets\UIEB\test\raw_256"
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        default=r"F:\A001\02research\0002code\02_datasets\UIEB\test\gt_256"
    )
    parser.add_argument(
        "--cluie_dir",
        type=str,
        default=r"F:\A001\02research\0001artical\DUILF-Net\reslut\CLUIE-Net\UIEB_256"
    )
    parser.add_argument(
        "--deepwave_dir",
        type=str,
        default=r"F:\A001\02research\0001artical\DUILF-Net\reslut\DeepWaveNet\UIEB_256"
    )
    parser.add_argument(
        "--funie_dir",
        type=str,
        default=r"F:\A001\02research\0001artical\DUILF-Net\reslut\FUnIE-GAN\UIEB_256"
    )
    parser.add_argument(
        "--shallow_dir",
        type=str,
        default=r"F:\A001\02research\0001artical\DUILF-Net\reslut\Shallow-UWNet\UIEB_256"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"F:\A001\02research\0001artical\DUILF-Net\motivation_analysis_uieb"
    )
    parser.add_argument(
        "--num_visuals",
        type=int,
        default=8,
        help="Number of representative images for visual panel"
    )

    args = parser.parse_args()
    main(args)