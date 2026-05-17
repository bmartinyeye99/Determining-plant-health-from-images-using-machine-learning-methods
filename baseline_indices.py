

import os
import cv2
import json
import numpy as np
import tifffile
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from mobilnet.gmm_thresholds import gaussian_crossing


# ---------------------------------------------------------------
# Index computation  (single source of truth — imported by other scripts)
# ---------------------------------------------------------------
def compute_indices(rgb_norm):

    R = rgb_norm[:, :, 0]
    G = rgb_norm[:, :, 1]
    B = rgb_norm[:, :, 2]

    exg   = np.clip(2.0 * G - R - B,               -1.0, 1.0)
    vari  = np.clip((G - R) / (G + R - B + 1e-6),  -1.0, 1.0)
    ngrdi = np.clip((G - R) / (G + R + 1e-6),       -1.0, 1.0)

    return {"ExG": exg, "VARI": vari, "NGRDI": ngrdi}


# ---------------------------------------------------------------
# NDVI ground truth  (matches dataset.py exactly)
# ---------------------------------------------------------------
def compute_ndvi_mask(rgb_bgr, nir_raw, thresholds):

    rgb      = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb_norm = rgb.astype(np.float32) / 255.0

    nir_norm = nir_raw.astype(np.float32) / (
        65535.0 if nir_raw.dtype == np.uint16 else 255.0
    )
    if nir_norm.ndim == 3:
        nir_norm = nir_norm[:, :, 0]
    if rgb_norm.shape[:2] != nir_norm.shape:
        nir_norm = cv2.resize(nir_norm,
                              (rgb_norm.shape[1], rgb_norm.shape[0]))

    red  = rgb_norm[:, :, 0]
    ndvi = (nir_norm - red) / (nir_norm + red + 1e-6)

    mask = np.zeros_like(ndvi, dtype=np.int64)
    mask[ndvi >= thresholds[0]] = 1
    mask[ndvi >= thresholds[1]] = 2
    mask[ndvi >= thresholds[2]] = 3
    return mask, rgb_norm



def _collect_index_values(root_dir, index_name, subsample_step=15):

    all_files = sorted(os.listdir(root_dir))
    rgb_files = [f for f in all_files if "_RGB" in f.upper()]

    vals = []
    for f in tqdm(rgb_files, desc=f"Collecting {index_name} (all images)"):
        rgb_bgr = cv2.imread(os.path.join(root_dir, f))
        if rgb_bgr is None:
            continue
        rgb_norm = (cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                    .astype(np.float32) / 255.0)
        indices = compute_indices(rgb_norm)
        flat = indices[index_name].flatten()[::subsample_step]
        flat = flat[np.isfinite(flat)]
        vals.append(flat)

    if not vals:
        raise RuntimeError(f"No valid images found in {root_dir}")

    data = np.concatenate(vals)
    print(f"   {index_name}: {len(data):,} samples from {len(vals)} images")
    return data


# ---------------------------------------------------------------
# GMM threshold fitting  (called only when cache miss)
# ---------------------------------------------------------------
def fit_index_thresholds(root_dir, index_name, plot_dir=None):
    """Fit a 4-component GMM on an RGB index with MAP-optimal crossing points.

    BIC/AIC sweep is disabled for performance. The model is fixed to k=4 
    to match the standard 4-class agronomic schema.
    """
    data    = _collect_index_values(root_dir, index_name)
    data_2d = data.reshape(-1, 1)

    # --- BIC / AIC sweep DISABLED ---
    # print(f"   Running BIC/AIC for {index_name} (k=2..6)...")
    # bic_scores, aic_scores, fitted_gmms = {}, {}, {}
    # for k in range(2, 7):
    #     gmm_k = GaussianMixture(
    #         n_components=k, random_state=42, max_iter=300, n_init=3,
    #     ).fit(data_2d)
    #     bic_scores[k]  = gmm_k.bic(data_2d)
    #     aic_scores[k]  = gmm_k.aic(data_2d)
    #     fitted_gmms[k] = gmm_k
    
    print(f"   Fitting 4-component GMM for {index_name}...")
    gmm = GaussianMixture(
        n_components=4, random_state=42, max_iter=300, n_init=3,
    ).fit(data_2d)

    means = gmm.means_.flatten()
    idx   = np.argsort(means)

    sorted_means   = means[idx]
    sorted_stds    = np.sqrt(gmm.covariances_.flatten()[idx])
    sorted_weights = gmm.weights_[idx]

    # MAP-optimal thresholds: solve for x where adjacent weighted Gaussian PDFs intersect.
    # This remains the single source of truth for statistically sound boundaries.
    thresholds = [
        float(gaussian_crossing(
            sorted_means[i],     sorted_stds[i],     sorted_weights[i],
            sorted_means[i + 1], sorted_stds[i + 1], sorted_weights[i + 1],
        ))
        for i in range(3)
    ]

    print(f"\n   {index_name} GMM 4-component fit (Fast Mode):")
    for i, name in enumerate(["dead/soil", "severe_stress", "moderate_stress", "healthy"]):
        print(f"      {name:>16s}: mean={sorted_means[i]:+.4f}  "
              f"std={sorted_stds[i]:.4f}  weight={sorted_weights[i]:.3f}")
    
    print(f"      MAP thresholds:      {[round(t, 4) for t in thresholds]}")

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        _plot_histogram(data, thresholds, gmm, idx, index_name, plot_dir)
        # _plot_bic_aic(bic_scores, aic_scores, index_name, plot_dir) # Disabled as scores aren't generated
        _plot_components(data, gmm, idx, thresholds, index_name, plot_dir)

    return thresholds


# ---------------------------------------------------------------
# Load-or-fit  (the single entry point used by run_baseline)
# ---------------------------------------------------------------
def _load_or_fit_thresholds(stats_path, idx_name, train_dir, plot_dir):
    """Return thresholds from JSON cache if present, otherwise fit and save.

    Key stored: stats['train']['{idx_name.lower()}_thresholds']
    e.g.  stats['train']['vari_thresholds']

    Args:
        stats_path: Path to class_stats.json.
        idx_name:   Index name ("ExG", "VARI", "NGRDI").
        train_dir:  Training image directory (used only when fitting).
        plot_dir:   Plot directory (used only when fitting).

    Returns:
        List of 3 threshold floats.
    """
    threshold_key = idx_name.lower() + "_thresholds"

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    if threshold_key in stats.get('train', {}):
        cached = stats['train'][threshold_key]
        print(f"   {idx_name} thresholds loaded from JSON: "
              f"{[round(t, 4) for t in cached]}")
        return cached

    # Cache miss → fit on training data
    print(f"   {idx_name} thresholds not in JSON — fitting on training set...")
    thresholds = fit_index_thresholds(train_dir, idx_name, plot_dir=plot_dir)

    # Persist so subsequent runs skip fitting
    stats['train'][threshold_key] = [round(t, 6) for t in thresholds]
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   {idx_name} thresholds saved to {stats_path} "
          f"under train.{threshold_key}")

    return thresholds


# ---------------------------------------------------------------
# Evaluate one index on a test split
# ---------------------------------------------------------------
def evaluate_index(root_dir, index_name, index_thresholds, ndvi_thresholds,
                   image_size=256):
    """Classify pixels with RGB index thresholds, evaluate against NDVI GT.

    Both RGB and NIR images are resized to image_size × image_size before
    any computation, matching the resolution used by the CNN.

    Args:
        root_dir:         Test image directory.
        index_name:       "ExG", "VARI", or "NGRDI".
        index_thresholds: 3 ascending floats from _load_or_fit_thresholds.
        ndvi_thresholds:  3 ascending floats from stats['train']['thresholds'].
        image_size:       Resize target (must match VegetationImageDataset).

    Returns:
        dict with keys: macro_f1, per_class_f1, classification_report,
        confusion_matrix.
    """
    all_files = sorted(os.listdir(root_dir))
    rgb_files = [f for f in all_files if "_RGB" in f.upper()]

    all_preds   = []
    all_targets = []

    for f in tqdm(rgb_files, desc=f"Eval {index_name}"):
        base  = f.upper().split("_RGB")[0]
        nir_f = next((n for n in all_files
                      if n.upper().startswith(base + "_NIR")), None)
        if not nir_f:
            continue

        rgb_bgr = cv2.imread(os.path.join(root_dir, f))
        nir_raw = cv2.imread(os.path.join(root_dir, nir_f),
                             cv2.IMREAD_UNCHANGED)
        if nir_raw is None:
            nir_raw = tifffile.imread(os.path.join(root_dir, nir_f))
        if nir_raw.ndim == 3:
            nir_raw = nir_raw[:, :, 0]

        # Resize to CNN evaluation resolution
        rgb_bgr = cv2.resize(rgb_bgr, (image_size, image_size))
        nir_raw = cv2.resize(
            nir_raw.astype(np.float32), (image_size, image_size),
        ).astype(nir_raw.dtype)

        # Ground truth from NDVI
        gt_mask, rgb_norm = compute_ndvi_mask(rgb_bgr, nir_raw, ndvi_thresholds)

        # Prediction from RGB index
        index_vals = compute_indices(rgb_norm)[index_name]
        pred_mask  = np.zeros_like(index_vals, dtype=np.int64)
        pred_mask[index_vals >= index_thresholds[0]] = 1
        pred_mask[index_vals >= index_thresholds[1]] = 2
        pred_mask[index_vals >= index_thresholds[2]] = 3

        # Subsample (same step as threshold collection)
        all_targets.append(gt_mask.flatten()[::15])
        all_preds.append(pred_mask.flatten()[::15])

    all_targets = np.concatenate(all_targets)
    all_preds   = np.concatenate(all_preds)

    class_names  = ["dead", "severe", "moderate", "healthy"]
    macro_f1     = f1_score(all_targets, all_preds, average='macro',
                            zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None,
                            zero_division=0)

    return {
        "macro_f1":              macro_f1,
        "per_class_f1":          {n: float(v)
                                  for n, v in zip(class_names, per_class_f1)},
        "classification_report": classification_report(
            all_targets, all_preds, target_names=class_names,
            digits=4, zero_division=0,
        ),
        "confusion_matrix":      confusion_matrix(all_targets, all_preds),
    }


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------
def run_baseline(train_dir, test_dir, stats_path, plot_dir=None):
    """Run all RGB index baselines with GMM thresholds (load or fit).

    On the first run thresholds are fitted on the training set and saved
    to stats_path.  On subsequent runs they are loaded from JSON — no
    re-fitting occurs.

    Args:
        train_dir:  Training image directory.
        test_dir:   Test image directory.
        stats_path: Path to class_stats.json.
        plot_dir:   Directory to save diagnostic plots (optional).

    Returns:
        dict mapping index name → evaluation result dict.
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    ndvi_thresholds = stats['train']['thresholds']
    print(f"NDVI ground-truth thresholds: {ndvi_thresholds}")

    index_names = ["ExG", "VARI", "NGRDI"]
    results     = {}

    for idx_name in index_names:
        print(f"\n{'='*60}")
        print(f"  Baseline: {idx_name}")
        print(f"{'='*60}")

        idx_thresholds = _load_or_fit_thresholds(
            stats_path, idx_name, train_dir, plot_dir
        )

        print(f"   Evaluating on test set...")
        res = evaluate_index(test_dir, idx_name, idx_thresholds, ndvi_thresholds)
        results[idx_name] = res

        print(f"\n{res['classification_report']}")
        print(f"   Macro F1: {res['macro_f1']:.4f}")

    # Summary table
    print(f"\n{'='*78}")
    print(f"  SUMMARY: RGB Index Baselines")
    print(f"{'='*78}")
    print(f"{'Method':<20} | {'Dead':<8} | {'Severe':<8} | "
          f"{'Moderate':<8} | {'Healthy':<8} | {'Macro F1':<8}")
    print(f"-" * 78)
    for idx_name in index_names:
        r  = results[idx_name]
        pf = r['per_class_f1']
        print(f"{idx_name:<20} | {pf['dead']:.4f}  | {pf['severe']:.4f}  | "
              f"{pf['moderate']:.4f}  | {pf['healthy']:.4f}  | "
              f"{r['macro_f1']:.4f}")
    print(f"{'='*78}")
    print(f"\nCompare with CNN results from trainer.test() output.")

    return results


# ---------------------------------------------------------------
# Diagnostic plots (unchanged from original)
# ---------------------------------------------------------------
def _plot_histogram(data, thresholds, gmm, idx, index_name, plot_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data, bins=300, density=True, alpha=0.7, color="#4a90d9",
            edgecolor="none", label=f"{index_name} distribution")

    colors_t = ["#e74c3c", "#f39c12", "#2ecc71"]
    for t, c, lab in zip(thresholds, colors_t,
                         [f"t1={thresholds[0]:.4f}",
                          f"t2={thresholds[1]:.4f}",
                          f"t3={thresholds[2]:.4f}"]):
        ax.axvline(t, color=c, linewidth=2.5, linestyle="--", label=lab)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    regions = [
        (xlim[0],       thresholds[0], "#e74c3c", "Dead/Soil"),
        (thresholds[0], thresholds[1], "#f39c12", "Severe Stress"),
        (thresholds[1], thresholds[2], "#f1c40f", "Moderate Stress"),
        (thresholds[2], xlim[1],       "#2ecc71", "Healthy"),
    ]
    for lo, hi, c, name in regions:
        ax.axvspan(lo, hi, alpha=0.08, color=c)
        ax.text((lo + hi) / 2, ylim[1] * 0.92, name,
                ha="center", fontsize=9, fontweight="bold", color=c)

    ax.set_xlabel(index_name, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{index_name} Distribution with GMM Thresholds",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    stats_text = (f"N = {len(data):,.0f}\n"
                  f"median = {np.median(data):.4f}\n"
                  f"p5 = {np.percentile(data, 5):.4f}\n"
                  f"p95 = {np.percentile(data, 95):.4f}")
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    plt.tight_layout()
    path = os.path.join(plot_dir, f"{index_name.lower()}_histogram_thresholds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Saved: {path}")


def _plot_bic_aic(bic_scores, aic_scores, index_name, plot_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(bic_scores)
    ax.plot(ks, [bic_scores[k] for k in ks], "o-", color="#e74c3c",
            linewidth=2, markersize=8, label="BIC")
    ax.plot(ks, [aic_scores[k] for k in ks], "s-", color="#3498db",
            linewidth=2, markersize=8, label="AIC")
    best = min(bic_scores, key=bic_scores.get)
    ax.axvline(best, color="#e74c3c", linestyle=":", alpha=0.5,
               label=f"BIC-optimal k={best}")
    ax.axvline(4, color="#2ecc71", linestyle="--", alpha=0.7, label="k=4 (used)")
    ax.set_xlabel("Number of GMM Components (k)", fontsize=12)
    ax.set_ylabel("Score (lower = better)", fontsize=12)
    ax.set_title(f"{index_name} — GMM Model Selection: BIC & AIC",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(ks)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, f"{index_name.lower()}_bic_aic.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Saved: {path}")


def _plot_components(data, gmm, idx, thresholds, index_name, plot_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data, bins=300, density=True, alpha=0.4, color="#bdc3c7",
            edgecolor="none", label=f"{index_name} data")
    x_range = np.linspace(data.min(), data.max(), 1000)
    colors = ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"]
    names  = ["Dead/Soil", "Severe Stress", "Moderate Stress", "Healthy"]
    means   = gmm.means_.flatten()[idx]
    stds    = np.sqrt(gmm.covariances_.flatten()[idx])
    weights = gmm.weights_[idx]
    for i in range(4):
        pdf = (weights[i] / (stds[i] * np.sqrt(2 * np.pi))
               * np.exp(-0.5 * ((x_range - means[i]) / stds[i]) ** 2))
        ax.plot(x_range, pdf, color=colors[i], linewidth=2.5,
                label=f"{names[i]} (μ={means[i]:.3f}, σ={stds[i]:.3f})")
        ax.fill_between(x_range, pdf, alpha=0.15, color=colors[i])
    for t in thresholds:
        ax.axvline(t, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_xlabel(index_name, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{index_name} — GMM Components on Distribution",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, f"{index_name.lower()}_gmm_components.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Saved: {path}")