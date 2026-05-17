import os
import cv2
import json
import numpy as np
import tifffile
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mobilnet.gmm_thresholds import fit_gmm_4class


def standardize_for_ndvi(rgb_img, nir_img):
    """Matches dataset.py NDVI computation EXACTLY."""
    rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_norm = rgb.astype(np.float32) / 255.0

    nir_norm = nir_img.astype(np.float32) / (
        65535.0 if nir_img.dtype == np.uint16 else 255.0
    )
    if nir_norm.ndim == 3:
        nir_norm = nir_norm[:, :, 0]

    if rgb_norm.shape[:2] != nir_norm.shape:
        nir_norm = cv2.resize(nir_norm,
                              (rgb_norm.shape[1], rgb_norm.shape[0]))

    ndvi = (nir_norm - rgb_norm[:, :, 0]) / (
        nir_norm + rgb_norm[:, :, 0] + 1e-6
    )
    return ndvi


def _collect_ndvi_values(root_dir, subsample_step=15):
    """Collect NDVI values from ALL RGB/NIR pairs in a directory."""
    all_files = sorted(os.listdir(root_dir))
    rgb_files = [f for f in all_files if "_RGB" in f.upper()]

    vals = []
    print(f"Collecting NDVI from ALL {len(rgb_files)} RGB images in {root_dir}...")
    for f in tqdm(rgb_files, desc="NDVI collection"):
        base = f.upper().split("_RGB")[0]
        nir_f = next(
            (n for n in all_files if n.upper().startswith(base + "_NIR")),
            None,
        )
        if not nir_f:
            continue

        rgb = cv2.imread(os.path.join(root_dir, f))
        nir = cv2.imread(os.path.join(root_dir, nir_f), cv2.IMREAD_UNCHANGED)
        if nir is None:
            nir = tifffile.imread(os.path.join(root_dir, nir_f))

        ndvi = standardize_for_ndvi(rgb, nir)
        flat = ndvi.flatten()[::subsample_step]
        flat = flat[np.isfinite(flat)]
        vals.append(flat)

    if not vals:
        raise RuntimeError(f"No valid RGB/NIR pairs found in {root_dir}")

    data = np.concatenate(vals)
    print(f"   Collected {len(data):,} NDVI samples from {len(vals)} image pairs")
    return data


def calculate_gmm_thresholds_4class(root_dir, plot_dir=None):
    """Fit 4-component GMM on NDVI from ALL training images, return MAP thresholds.

    k=4 is fixed by the agronomic four-tier vegetation taxonomy, not by model
    selection. No BIC/AIC sweep — the choice of k is a domain constraint.

    Args:
        root_dir:  Training image directory.
        plot_dir:  Directory to save diagnostic plots (optional).

    Returns:
        thresholds: list of 3 floats [t1, t2, t3], MAP-optimal class boundaries.
    """
    data = _collect_ndvi_values(root_dir)

    # MAP-optimal thresholds via shared utility
    thresholds, gmm, idx = fit_gmm_4class(data)

    sorted_means = gmm.means_.flatten()[idx]
    sorted_stds = np.sqrt(gmm.covariances_.flatten()[idx])
    sorted_weights = gmm.weights_[idx]

    midpoints = [
        (sorted_means[i] + sorted_means[i + 1]) / 2.0 for i in range(3)
    ]

    print(f"\nGMM 4-component fit:")
    for i, name in enumerate(["dead/soil", "severe_stress",
                              "moderate_stress", "healthy"]):
        print(f"   {name:>16s}: mean={sorted_means[i]:+.4f}  "
              f"std={sorted_stds[i]:.4f}  weight={sorted_weights[i]:.3f}")
    print(f"   MAP thresholds:      {[round(t, 4) for t in thresholds]}")
    print(f"   (legacy midpoints):  {[round(t, 4) for t in midpoints]}")
    shifts = [thresholds[i] - midpoints[i] for i in range(3)]
    print(f"   shift (MAP - mid):   {[round(s, 4) for s in shifts]}")

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        _plot_ndvi_histogram(data, thresholds, gmm, idx, plot_dir, midpoints)
        _plot_gmm_components(data, gmm, idx, thresholds, plot_dir)

    return thresholds


def _plot_ndvi_histogram(data, thresholds, gmm, idx, plot_dir, midpoints=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(data, bins=300, density=True, alpha=0.7, color="#4a90d9",
            edgecolor="none", label="NDVI distribution")

    colors_t = ["#e74c3c", "#f39c12", "#2ecc71"]
    for t, c, lab in zip(thresholds, colors_t,
                         [f"t1 = {thresholds[0]:.4f}",
                          f"t2 = {thresholds[1]:.4f}",
                          f"t3 = {thresholds[2]:.4f}"]):
        ax.axvline(t, color=c, linewidth=2.5, linestyle="--",
                   label=f"{lab} (MAP)")

    if midpoints is not None:
        for m in midpoints:
            ax.axvline(m, color="grey", linewidth=1.2, linestyle=":", alpha=0.6)
        ax.plot([], [], color="grey", linestyle=":",
                label="Midpoint (legacy)")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    regions = [
        (xlim[0], thresholds[0], "#e74c3c", "Dead/Soil"),
        (thresholds[0], thresholds[1], "#f39c12", "Severe Stress"),
        (thresholds[1], thresholds[2], "#f1c40f", "Moderate Stress"),
        (thresholds[2], xlim[1], "#2ecc71", "Healthy"),
    ]
    for lo, hi, c, name in regions:
        ax.axvspan(lo, hi, alpha=0.08, color=c)
        ax.text((lo + hi) / 2, ylim[1] * 0.92, name, ha="center", fontsize=9,
                fontweight="bold", color=c, alpha=0.9)

    ax.set_xlabel("NDVI", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("NDVI Distribution with MAP-Optimal GMM Thresholds",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    stats_text = (
        f"N pixels = {len(data):,.0f}\n"
        f"median = {np.median(data):.4f}\n"
        f"p5 = {np.percentile(data, 5):.4f}\n"
        f"p95 = {np.percentile(data, 95):.4f}"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(plot_dir, "ndvi_histogram_thresholds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


def _plot_gmm_components(data, gmm, idx, thresholds, plot_dir):
    """GMM components overlaid on NDVI histogram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data, bins=300, density=True, alpha=0.4, color="#bdc3c7",
            edgecolor="none", label="NDVI data")
    x_range = np.linspace(data.min(), data.max(), 1000)
    colors = ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"]
    names = ["Dead/Soil", "Severe Stress", "Moderate Stress", "Healthy"]
    means = gmm.means_.flatten()[idx]
    stds = np.sqrt(gmm.covariances_.flatten()[idx])
    weights = gmm.weights_[idx]
    for i in range(4):
        pdf = (weights[i] / (stds[i] * np.sqrt(2 * np.pi))
               * np.exp(-0.5 * ((x_range - means[i]) / stds[i]) ** 2))
        ax.plot(x_range, pdf, color=colors[i], linewidth=2.5,
                label=f"{names[i]} (mu={means[i]:.3f}, sigma={stds[i]:.3f})")
        ax.fill_between(x_range, pdf, alpha=0.15, color=colors[i])
    for t in thresholds:
        ax.axvline(t, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("NDVI", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("GMM Components on NDVI Distribution", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, "gmm_components.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


def calculate_and_save_stats(root_dir, output_path, split_name, plot_dir=None):
    """Calculate 4-class NDVI statistics and save to JSON.

    Train split: fits GMM, derives MAP thresholds, saves plots.
    Val/Test:    loads thresholds from train, computes class percentages.
    """
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        return

    all_files = sorted(os.listdir(root_dir))
    rgb_files = [f for f in all_files if "_RGB" in f.upper()]
    counts = np.zeros(4, dtype=np.int64)

    gmm_diagnostics = None 

    if split_name == "train":
        thresholds = calculate_gmm_thresholds_4class(root_dir, plot_dir=plot_dir)
        
        data = _collect_ndvi_values(root_dir)
        _, gmm, idx = fit_gmm_4class(data)
        sorted_means = gmm.means_.flatten()[idx]
        sorted_stds = np.sqrt(gmm.covariances_.flatten()[idx])
        sorted_weights = gmm.weights_[idx]
        midpoints = [
            float((sorted_means[i] + sorted_means[i + 1]) / 2.0)
            for i in range(3)
        ]
        gmm_diagnostics = {
            "threshold_method": "map_crossing",
            "gmm_components": [
                {"mean": float(sorted_means[i]),
                 "std": float(sorted_stds[i]),
                 "weight": float(sorted_weights[i])}
                for i in range(4)
            ],
            "midpoint_thresholds_legacy": [round(m, 6) for m in midpoints],
        }
    else:
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"Stats file '{output_path}' not found. "
                "Run calculate_and_save_stats on the training split first."
            )
        with open(output_path, 'r') as f:
            saved = json.load(f)
        thresholds = saved.get("train", {}).get("thresholds")
        if thresholds is None:
            raise KeyError(
                f"stats file '{output_path}' has no 'train.thresholds' key."
            )

    print(f"--- Calculating 4-Class Counts in {root_dir} ({split_name}) ---")
    for f in tqdm(rgb_files):
        base = f.upper().split("_RGB")[0]
        nir_file = next(
            (n for n in all_files if n.upper().startswith(base + "_NIR")),
            None,
        )
        if nir_file:
            rgb = cv2.imread(os.path.join(root_dir, f))
            nir = cv2.imread(os.path.join(root_dir, nir_file),
                             cv2.IMREAD_UNCHANGED)
            if nir is None:
                nir = tifffile.imread(os.path.join(root_dir, nir_file))

            ndvi = standardize_for_ndvi(rgb, nir)
            mask = np.zeros_like(ndvi, dtype=np.int64)
            mask[ndvi >= thresholds[0]] = 1
            mask[ndvi >= thresholds[1]] = 2
            mask[ndvi >= thresholds[2]] = 3

            for c in range(4):
                counts[c] += np.sum(mask == c)

    total_pixels = np.sum(counts)
    if total_pixels == 0:
        return

    percentages = (counts / total_pixels) * 100
    raw_weights = total_pixels / (4 * counts.astype(np.float32) + 1e-6)
    normalized_weights = raw_weights / raw_weights[0]

    full_stats = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                full_stats = json.load(f)
        except Exception:
            full_stats = {}

    split_record = {
        "thresholds": [round(t, 4) for t in thresholds],
        "class_percentages": {
            "dead_soil": round(percentages[0], 2),
            "severe_stress": round(percentages[1], 2),
            "moderate_stress": round(percentages[2], 2),
            "healthy": round(percentages[3], 2),
        },
        "calculated_weights": [
            round(w, 4) for w in normalized_weights.tolist()
        ],
    }
    if gmm_diagnostics is not None:
        split_record.update(gmm_diagnostics)

    full_stats[split_name] = split_record

    with open(output_path, 'w') as f_out:
        json.dump(full_stats, f_out, indent=4)
    print(
        f"{split_name.upper()} Stats: "
        f"Dead:{percentages[0]:.1f}% | Sev:{percentages[1]:.1f}% | "
        f"Mod:{percentages[2]:.1f}% | Heal:{percentages[3]:.1f}%"
    )