

import os
import re
import cv2
import json
import torch
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, classification_report

from mobilnet.model import EfficientUNet4Class
# Single source of truth for index formulas and clipping
from mobilnet.baseline_indices import compute_indices


def find_pairs_offset5(data_dir):
    """Pair RGB .png with NIR .tif at index number +5."""
    all_files = sorted(os.listdir(data_dir))
    png_files = [f for f in all_files if f.lower().endswith('.png')]
    tif_set   = {f for f in all_files if f.lower().endswith('.tif')}

    pairs = []
    for png in png_files:
        m = re.search(r'(\d+)\.png$', png, re.IGNORECASE)
        if not m:
            continue
        num    = int(m.group(1))
        prefix = png[:m.start(1)]
        for fmt in (f"{prefix}{num+5:04d}.tif", f"{prefix}{num+5}.tif"):
            if fmt in tif_set:
                pairs.append((png, fmt))
                break
    return pairs


def find_pairs_suffix(data_dir):
    all_files = sorted(os.listdir(data_dir))
    rgb_files = [f for f in all_files if "_RGB" in f.upper()]

    pairs = []
    for f in rgb_files:
        base = f.upper().split("_RGB")[0]
        nir  = next((n for n in all_files
                     if n.upper().startswith(base + "_NIR")), None)
        if nir:
            pairs.append((f, nir))
    return pairs


PAIR_MODES = {"offset5": find_pairs_offset5, "suffix": find_pairs_suffix}



def preprocess_rgb(rgb_np):
    """Gray World colour constancy + CLAHE on LAB L-channel."""
    rgb_f = rgb_np.astype(np.float32)
    if rgb_f.max() <= 1.0:
        rgb_f = rgb_f * 255.0

    avg_all = rgb_f.mean()
    for c in range(3):
        avg_c = rgb_f[:, :, c].mean()
        if avg_c > 1e-6:
            rgb_f[:, :, c] *= avg_all / avg_c
    rgb_f = np.clip(rgb_f, 0, 255)

    u8  = rgb_f.astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0



def compute_training_stats(train_dir, pair_mode="suffix", image_size=256, max_images=200):

    pairs = PAIR_MODES[pair_mode](train_dir)
    if not pairs:
        raise RuntimeError(f"No pairs found in {train_dir} with mode {pair_mode}")
    
    # Limit to max_images for speed
    if len(pairs) > max_images:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(pairs), max_images, replace=False)
        pairs = [pairs[i] for i in indices]
    
    rgb_sums = np.zeros(3, dtype=np.float64)
    rgb_sq_sums = np.zeros(3, dtype=np.float64)
    nir_sum = 0.0
    nir_sq_sum = 0.0
    n_pixels = 0
    
    hist_rgb = np.zeros((3, 256), dtype=np.float64)
    hist_nir = np.zeros(256, dtype=np.float64)
    
    for rgb_name, nir_name in tqdm(pairs, desc="Computing training stats"):
        rgb_bgr = cv2.imread(os.path.join(train_dir, rgb_name))
        if rgb_bgr is None:
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.resize(rgb, (image_size, image_size))
        rgb_norm = rgb_r.astype(np.float32) / 255.0
        
        nir_raw = cv2.imread(os.path.join(train_dir, nir_name), cv2.IMREAD_UNCHANGED)
        if nir_raw is None:
            nir_raw = tifffile.imread(os.path.join(train_dir, nir_name))
        if nir_raw.ndim == 3:
            nir_raw = nir_raw[:, :, 0]
        nir_f = nir_raw.astype(np.float32) / (65535.0 if nir_raw.dtype == np.uint16 else 255.0)
        nir_r = cv2.resize(nir_f, (image_size, image_size))
        
        npx = image_size * image_size
        n_pixels += npx
        
        for c in range(3):
            rgb_sums[c] += rgb_norm[:, :, c].sum()
            rgb_sq_sums[c] += (rgb_norm[:, :, c] ** 2).sum()
            hist_rgb[c] += cv2.calcHist([rgb_r], [c], None, [256], [0, 256]).flatten()
        
        nir_sum += nir_r.sum()
        nir_sq_sum += (nir_r ** 2).sum()
        nir_u8 = np.clip(nir_r * 255, 0, 255).astype(np.uint8)
        hist_nir += cv2.calcHist([nir_u8], [0], None, [256], [0, 256]).flatten()
    
    mean_rgb = rgb_sums / n_pixels
    std_rgb = np.sqrt(rgb_sq_sums / n_pixels - mean_rgb ** 2)
    mean_nir = nir_sum / n_pixels
    std_nir = np.sqrt(nir_sq_sum / n_pixels - mean_nir ** 2)
    
    for c in range(3):
        hist_rgb[c] = np.cumsum(hist_rgb[c])
        hist_rgb[c] /= hist_rgb[c][-1]
    hist_nir = np.cumsum(hist_nir)
    hist_nir /= hist_nir[-1]
    
    return {
        'mean_rgb': mean_rgb,
        'std_rgb': std_rgb,
        'mean_nir': mean_nir,
        'std_nir': std_nir,
        'cdf_rgb': hist_rgb,
        'cdf_nir': hist_nir,
    }


def apply_zscore(rgb_norm, nir_r, train_stats):

    rgb_z = rgb_norm.copy()
    for c in range(3):
        rgb_z[:, :, c] = (rgb_z[:, :, c] - train_stats['mean_rgb'][c]) / (train_stats['std_rgb'][c] + 1e-8)
    # Rescale from z-scores back to approx [0, 1] using 3-sigma clipping
    rgb_z = np.clip(rgb_z / 6.0 + 0.5, 0, 1)
    
    nir_z = (nir_r - train_stats['mean_nir']) / (train_stats['std_nir'] + 1e-8)
    nir_z = np.clip(nir_z / 6.0 + 0.5, 0, 1)
    
    return rgb_z, nir_z


def apply_histogram_matching(rgb_u8, nir_r, train_stats):

    rgb_matched = np.zeros_like(rgb_u8)
    
    for c in range(3):
        # Compute CDF of source (target-domain) channel
        src_hist = cv2.calcHist([rgb_u8], [c], None, [256], [0, 256]).flatten()
        src_cdf = np.cumsum(src_hist)
        src_cdf = src_cdf / src_cdf[-1]
        
        ref_cdf = train_stats['cdf_rgb'][c]
        
        lut = np.zeros(256, dtype=np.uint8)
        for src_val in range(256):
            diff = np.abs(ref_cdf - src_cdf[src_val])
            lut[src_val] = np.argmin(diff)
        
        rgb_matched[:, :, c] = lut[rgb_u8[:, :, c]]
    
    nir_u8 = np.clip(nir_r * 255, 0, 255).astype(np.uint8)
    src_hist_nir = cv2.calcHist([nir_u8], [0], None, [256], [0, 256]).flatten()
    src_cdf_nir = np.cumsum(src_hist_nir)
    src_cdf_nir = src_cdf_nir / (src_cdf_nir[-1] + 1e-8)
    ref_cdf_nir = train_stats['cdf_nir']
    
    lut_nir = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        diff = np.abs(ref_cdf_nir - src_cdf_nir[src_val])
        lut_nir[src_val] = np.argmin(diff)
    
    nir_matched = lut_nir[nir_u8].astype(np.float32) / 255.0
    
    return rgb_matched, nir_matched


def _fit_target_thresholds(data_dir, pairs, index_name, image_size, subsample_step=15):
    """Fit a 4-component GMM on index_name over all pairs in data_dir."""
    vals = []
    for rgb_name, nir_name in tqdm(pairs, desc=f"Fitting {index_name} (target domain)", leave=False):
        rgb_bgr = cv2.imread(os.path.join(data_dir, rgb_name))
        if rgb_bgr is None: continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_norm = cv2.resize(rgb, (image_size, image_size)).astype(np.float32) / 255.0
        
        if index_name == "NDVI":
            nir_raw = cv2.imread(os.path.join(data_dir, nir_name), cv2.IMREAD_UNCHANGED)
            if nir_raw is None: nir_raw = tifffile.imread(os.path.join(data_dir, nir_name))
            if nir_raw.ndim == 3: nir_raw = nir_raw[:, :, 0]
            nir_f = cv2.resize(nir_raw.astype(np.float32) / (65535.0 if nir_raw.dtype == np.uint16 else 255.0), (image_size, image_size))
            flat = ((nir_f - rgb_norm[:, :, 0]) / (nir_f + rgb_norm[:, :, 0] + 1e-6)).flatten()[::subsample_step]
        else:
            flat = compute_indices(rgb_norm)[index_name].flatten()[::subsample_step]
            
        vals.append(flat[np.isfinite(flat)])

    if not vals:
        raise RuntimeError(f"No valid images for {index_name} fitting in {data_dir}")

    data = np.concatenate(vals).reshape(-1, 1)
    gmm  = GaussianMixture(n_components=4, random_state=42, max_iter=300, n_init=3).fit(data)
    means      = np.sort(gmm.means_.flatten())
    thresholds = [float((means[i] + means[i+1]) / 2.0) for i in range(len(means) - 1)]

    print(f"  {index_name} fitted thresholds: {[round(t, 4) for t in thresholds]}")
    return thresholds


def _get_or_fit_index_thresholds(stats_path, dataset_key, data_dir, pairs, index_name, image_size):
    """Return thresholds from JSON cache; fit on target domain if absent."""
    threshold_key = index_name.lower() + "_thresholds"
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    cached = (stats.get('cross_domain', {}).get(dataset_key, {}).get(threshold_key))
    if cached is not None:
        print(f"  {index_name} thresholds loaded from JSON: {[round(t, 4) for t in cached]}")
        return cached

    thresholds = _fit_target_thresholds(data_dir, pairs, index_name, image_size)

    stats.setdefault('cross_domain', {}).setdefault(dataset_key, {})[threshold_key] = [round(t, 6) for t in thresholds]
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved to cross_domain.{dataset_key}.{threshold_key}")

    return thresholds



def _classify_index(index_map, thresholds):
    mask = np.zeros_like(index_map, dtype=np.int64)
    mask[index_map >= thresholds[0]] = 1
    mask[index_map >= thresholds[1]] = 2
    mask[index_map >= thresholds[2]] = 3
    return mask


# ---------------------------------------------------------------
def _run_cnn_inference(model, rgb_prep, nir_r, device):
    """Run CNN in both full and RGB-only modes.
    
        model:    EfficientUNet4Class in eval mode
        rgb_prep: (H, W, 3) float32 [0, 1] — preprocessed RGB
        nir_r:    (H, W) float32 [0, 1] — NIR channel
        device:   torch device
    
    Returns:
        pred_full:    (H, W) int64 — argmax predictions (RGB+NIR)
        pred_rgb:     (H, W) int64 — argmax predictions (RGB-only)
        eov_full:     (H, W) float32 — expected ordinal value (RGB+NIR)
        eov_rgb:      (H, W) float32 — expected ordinal value (RGB-only)
    """
    rgb_t = torch.from_numpy(rgb_prep.transpose(2, 0, 1))
    nir_t = torch.from_numpy(nir_r).unsqueeze(0)
    
    full_input = torch.cat([rgb_t, nir_t], 0).unsqueeze(0).to(device)
    rgb_input  = torch.cat([rgb_t, torch.zeros_like(nir_t)], 0).unsqueeze(0).to(device)
    
    # Ordinal weights for expected value: [0, 1, 2, 3]
    ordinal_weights = torch.arange(4, dtype=torch.float32, device=device).view(1, 4, 1, 1)
    
    logits_full = model(full_input)
    probs_full  = torch.softmax(logits_full, dim=1)
    pred_full   = logits_full.argmax(1).cpu().numpy()[0]
    eov_full    = (probs_full * ordinal_weights).sum(dim=1).cpu().numpy()[0]
    
    logits_rgb = model(rgb_input)
    probs_rgb  = torch.softmax(logits_rgb, dim=1)
    pred_rgb   = logits_rgb.argmax(1).cpu().numpy()[0]
    eov_rgb    = (probs_rgb * ordinal_weights).sum(dim=1).cpu().numpy()[0]
    
    return pred_full, pred_rgb, eov_full, eov_rgb



def run_cross_domain_eval(
    model_path,
    data_dir,
    stats_path,
    pair_mode="offset5",
    output_dir="/content/cross_domain_results",
    image_size=256,
    num_plots=10,
    train_dir=None,
):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Load model
    model = EfficientUNet4Class(
        in_channels=4, num_classes=4,
        use_domain_head=False, moddrop_nir_prob=0.5,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model loaded")

    # Find pairs
    pairs = PAIR_MODES[pair_mode](data_dir)
    if not pairs:
        print("No pairs found. Check pair_mode and file naming.")
        return
    print(f"Found {len(pairs)} RGB/NIR pairs in {data_dir}")

    train_stats = None
    if train_dir is not None:
        print(f"\nComputing training set statistics from {train_dir}...")
        train_pair_mode = "suffix"  # Kaggle uses _RGB/_NIR naming
        train_stats = compute_training_stats(train_dir, pair_mode=train_pair_mode, image_size=image_size)
        print(f"  RGB mean: {train_stats['mean_rgb'].round(4)}")
        print(f"  RGB std:  {train_stats['std_rgb'].round(4)}")
        print(f"  NIR mean: {train_stats['mean_nir']:.4f}")
        print(f"  NIR std:  {train_stats['std_nir']:.4f}")

    # --- DUAL THRESHOLD SETUP ---
    dataset_key = os.path.basename(os.path.normpath(data_dir))
    
    with open(stats_path, 'r') as f:
        stats_check = json.load(f)
    train_ndvi_thresholds = stats_check['train']['thresholds']
    print(f"\n[TRAINING] NDVI Ground-Truth Thresholds: {train_ndvi_thresholds}")

    # 2. Fit/Load Target (New) NDVI Thresholds
    print(f"\nLoading/fitting TARGET index thresholds for '{dataset_key}'...")
    target_ndvi_thresholds = _get_or_fit_index_thresholds(
        stats_path, dataset_key, data_dir, pairs, "NDVI", image_size
    )
    
    drift = np.array(target_ndvi_thresholds) - np.array(train_ndvi_thresholds)
    print(f"--> NET NDVI DRIFT (Target - Train): {[round(d, 4) for d in drift]}")

    # Load Baselines
    index_names    = ["VARI", "ExG", "NGRDI"]
    idx_thresholds = {}
    for idx_name in index_names:
        idx_thresholds[idx_name] = _get_or_fit_index_thresholds(
            stats_path, dataset_key, data_dir, pairs, idx_name, image_size
        )

    color_map = {
        0: [255,   0,   0],
        1: [255, 255,   0],
        2: [  0, 255,   0],
        3: [  0, 100,   0],
    }

    def create_heatmap(mask):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls, color in color_map.items():
            rgb[mask == cls] = color
        return rgb

    # Accumulators — baseline (no normalization)
    all_gt_train   = []
    all_gt_target  = []
    all_pred_full  = []
    all_pred_rgb   = []
    all_idx_preds  = {n: [] for n in index_names}
    
    all_ndvi_cont     = []
    all_idx_cont      = {n: [] for n in index_names}
    all_eov_full_cont = []  
    all_eov_rgb_cont  = []

    # Accumulators — z-score normalization
    all_eov_full_zscore = []
    all_eov_rgb_zscore  = []
    all_pred_full_zscore = []
    all_pred_rgb_zscore  = []
    
    # Accumulators — histogram matching
    all_eov_full_histmatch = []
    all_eov_rgb_histmatch  = []
    all_pred_full_histmatch = []
    all_pred_rgb_histmatch  = []

    SUBSAMPLE = 15

    print(f"\nRunning inference on {len(pairs)} pairs...")
    with torch.no_grad():
        for plot_count, (rgb_name, nir_name) in enumerate(
            tqdm(pairs, desc="Cross-domain eval")
        ):
            try:
                # Load
                rgb_bgr = cv2.imread(os.path.join(data_dir, rgb_name))
                if rgb_bgr is None: continue
                rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

                nir_raw = cv2.imread(os.path.join(data_dir, nir_name), cv2.IMREAD_UNCHANGED)
                if nir_raw is None: nir_raw = tifffile.imread(os.path.join(data_dir, nir_name))
                if nir_raw.ndim == 3: nir_raw = nir_raw[:, :, 0]
                nir_f = nir_raw.astype(np.float32) / (65535.0 if nir_raw.dtype == np.uint16 else 255.0)

                # Resize
                rgb_r    = cv2.resize(rgb,   (image_size, image_size))
                nir_r    = cv2.resize(nir_f, (image_size, image_size))
                rgb_norm = rgb_r.astype(np.float32) / 255.0

                # DUAL NDVI GROUND TRUTHS
                red    = rgb_norm[:, :, 0]
                ndvi   = (nir_r - red) / (nir_r + red + 1e-6)
                gt_mask_train  = _classify_index(ndvi, train_ndvi_thresholds)
                gt_mask_target = _classify_index(ndvi, target_ndvi_thresholds)

                indices  = compute_indices(rgb_norm)
                idx_masks = {n: _classify_index(indices[n], idx_thresholds[n]) for n in index_names}

                rgb_prep = preprocess_rgb(rgb_r)
                pred_full, pred_rgb, eov_full, eov_rgb = _run_cnn_inference(
                    model, rgb_prep, nir_r, device
                )

                all_gt_train.append(gt_mask_train.flatten()[::SUBSAMPLE])
                all_gt_target.append(gt_mask_target.flatten()[::SUBSAMPLE])
                all_pred_full.append(pred_full.flatten()[::SUBSAMPLE])
                all_pred_rgb.append(pred_rgb.flatten()[::SUBSAMPLE])
                for n in index_names:
                    all_idx_preds[n].append(idx_masks[n].flatten()[::SUBSAMPLE])

                # Collect Continuous (now using Expected Ordinal Value)
                ndvi_flat = ndvi.flatten()[::SUBSAMPLE]
                valid = np.isfinite(ndvi_flat)
                all_ndvi_cont.append(ndvi_flat[valid])
                for n in index_names:
                    all_idx_cont[n].append(indices[n].flatten()[::SUBSAMPLE][valid])
                all_eov_full_cont.append(eov_full.flatten()[::SUBSAMPLE][valid])
                all_eov_rgb_cont.append(eov_rgb.flatten()[::SUBSAMPLE][valid])

                if train_stats is not None:
                    rgb_z, nir_z = apply_zscore(rgb_norm, nir_r, train_stats)
                    rgb_z_prep = preprocess_rgb((rgb_z * 255).astype(np.uint8))
                    pf_z, pr_z, eov_f_z, eov_r_z = _run_cnn_inference(
                        model, rgb_z_prep, nir_z, device
                    )
                    all_pred_full_zscore.append(pf_z.flatten()[::SUBSAMPLE])
                    all_pred_rgb_zscore.append(pr_z.flatten()[::SUBSAMPLE])
                    all_eov_full_zscore.append(eov_f_z.flatten()[::SUBSAMPLE][valid])
                    all_eov_rgb_zscore.append(eov_r_z.flatten()[::SUBSAMPLE][valid])

                if train_stats is not None:
                    rgb_hm, nir_hm = apply_histogram_matching(rgb_r, nir_r, train_stats)
                    rgb_hm_prep = preprocess_rgb(rgb_hm)
                    pf_hm, pr_hm, eov_f_hm, eov_r_hm = _run_cnn_inference(
                        model, rgb_hm_prep, nir_hm, device
                    )
                    all_pred_full_histmatch.append(pf_hm.flatten()[::SUBSAMPLE])
                    all_pred_rgb_histmatch.append(pr_hm.flatten()[::SUBSAMPLE])
                    all_eov_full_histmatch.append(eov_f_hm.flatten()[::SUBSAMPLE][valid])
                    all_eov_rgb_histmatch.append(eov_r_hm.flatten()[::SUBSAMPLE][valid])

                if plot_count < num_plots:
                    fig, axes = plt.subplots(1, 7, figsize=(32, 5))
                    fig.suptitle(f"{rgb_name} | Drift: {drift.round(2)}", fontsize=14)
                    
                    axes[0].imshow(rgb_r);                     axes[0].set_title("Raw RGB")
                    axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-0.3, vmax=0.8); axes[1].set_title("NDVI")
                    axes[2].imshow(create_heatmap(gt_mask_train)); axes[2].set_title("GT (Train Thresh)")
                    axes[3].imshow(create_heatmap(gt_mask_target));axes[3].set_title("GT (Target Thresh)")
                    axes[4].imshow(create_heatmap(pred_full)); axes[4].set_title("CNN (RGB+NIR)")
                    axes[5].imshow(create_heatmap(pred_rgb));  axes[5].set_title("CNN (RGB-only)")
                    axes[6].imshow(create_heatmap(idx_masks["VARI"])); axes[6].set_title("VARI baseline")
                    
                    for ax in axes: ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"sample_{plot_count:03d}.png"), dpi=150, bbox_inches='tight')
                    plt.close()

            except Exception as e:
                print(f"Error processing {rgb_name}: {e}")
                continue

    if not all_gt_train:
        print("No samples were successfully processed.")
        return

    all_gt_train   = np.concatenate(all_gt_train)
    all_gt_target  = np.concatenate(all_gt_target)
    all_pred_full  = np.concatenate(all_pred_full)
    all_pred_rgb   = np.concatenate(all_pred_rgb)
    for n in index_names: all_idx_preds[n] = np.concatenate(all_idx_preds[n])
        
    all_ndvi_cont = np.concatenate(all_ndvi_cont)
    for n in index_names: all_idx_cont[n] = np.concatenate(all_idx_cont[n])
    all_eov_full_cont = np.concatenate(all_eov_full_cont)
    all_eov_rgb_cont  = np.concatenate(all_eov_rgb_cont)

    class_names = ["dead", "severe", "moderate", "healthy"]

    # ================================================================
    # SECTION A: DUAL F1 REPORTING
    # ================================================================
    print(f"\n{'='*85}")
    print(f"  SECTION A: DUAL-THRESHOLD CLASSIFICATION METRICS")
    print(f"  Dataset: {dataset_key}  |  {len(pairs)} pairs")
    print(f"{'='*85}")

    def print_f1_block(title, gt_array, methods_dict):
        print(f"\n--- {title} ---")
        f1_res = {}
        for label, preds in methods_dict:
            per = f1_score(gt_array, preds, average=None, zero_division=0)
            mac = f1_score(gt_array, preds, average='macro', zero_division=0)
            f1_res[label] = {"per_class": per.tolist(), "macro": mac}
            
        print(f"{'Method':<22} | {'Dead':>6} | {'Severe':>6} | {'Moderate':>8} | {'Healthy':>7} | {'Macro F1':>8}")
        print("-" * 70)
        for method, res in f1_res.items():
            p = res['per_class']
            print(f"{method:<22} | {p[0]:>6.4f} | {p[1]:>6.4f} | {p[2]:>8.4f} | {p[3]:>7.4f} | {res['macro']:>8.4f}")
        return f1_res

    # 1. Blind Test
    f1_results_train = print_f1_block(
        f"BLIND TEST (Scored vs. Training Thresholds: {train_ndvi_thresholds})", 
        all_gt_train,
        [("CNN (RGB+NIR)", all_pred_full), ("CNN (RGB-only)", all_pred_rgb)],
    )

    # 2. Calibrated Test (Including Baselines)
    calibrated_methods = [("CNN (RGB+NIR)", all_pred_full), ("CNN (RGB-only)", all_pred_rgb)]
    for n in index_names:
        calibrated_methods.append((n, all_idx_preds[n]))
    
    f1_results_target = print_f1_block(
        f"CALIBRATED TEST (Scored vs. Local Target Thresholds: {[round(t, 4) for t in target_ndvi_thresholds]})",
        all_gt_target,
        calibrated_methods,
    )

    # ================================================================
    # SECTION B: Spearman correlation (using Expected Ordinal Value)
    # ================================================================
    print(f"\n{'='*85}")
    print(f"  SECTION B: SPEARMAN RANK CORRELATION WITH CONTINUOUS NDVI")
    print(f"  CNN score = Expected Ordinal Value: Σ i·P(class=i), i∈{{0,1,2,3}}")
    print(f"  Threshold-free ordinal agreement between methods and NDVI.")
    print(f"{'='*85}")

    spearman_results = {}
    
    for label, cont_preds, dict_key in [
        ("CNN (RGB+NIR)", all_eov_full_cont, "CNN_RGB_NIR_EOV"),
        ("CNN (RGB-only)", all_eov_rgb_cont, "CNN_RGB_only_EOV"),
    ]:
        rho, pval = spearmanr(cont_preds, all_ndvi_cont)
        spearman_results[dict_key] = {"rho": float(rho), "p": float(pval)}
        print(f"  {label:<18} vs NDVI:  ρ = {rho:.4f}  (p = {pval:.2e})")

    print("-" * 70)
    for n in index_names:
        rho, pval = spearmanr(all_idx_cont[n], all_ndvi_cont)
        spearman_results[n] = {"rho": float(rho), "p": float(pval)}
        print(f"  {n:<18} vs NDVI:  ρ = {rho:.4f}  (p = {pval:.2e})")

    # ================================================================
    # SECTION C: Domain Adaptation Experiments
    # ================================================================
    da_results = {}
    
    if train_stats is not None:
        print(f"\n{'='*85}")
        print(f"  SECTION C: INPUT-LEVEL DOMAIN ADAPTATION")
        print(f"  Testing z-score normalization and histogram matching separately.")
        print(f"  Same model, same weights — only input preprocessing changes.")
        print(f"{'='*85}")
        
        # Concatenate DA accumulators
        all_pred_full_zscore = np.concatenate(all_pred_full_zscore)
        all_pred_rgb_zscore  = np.concatenate(all_pred_rgb_zscore)
        all_eov_full_zscore  = np.concatenate(all_eov_full_zscore)
        all_eov_rgb_zscore   = np.concatenate(all_eov_rgb_zscore)
        
        all_pred_full_histmatch = np.concatenate(all_pred_full_histmatch)
        all_pred_rgb_histmatch  = np.concatenate(all_pred_rgb_histmatch)
        all_eov_full_histmatch  = np.concatenate(all_eov_full_histmatch)
        all_eov_rgb_histmatch   = np.concatenate(all_eov_rgb_histmatch)
        
        # --- C1: Spearman comparison across normalization strategies ---
        print(f"\n--- C1: Spearman ρ (EOV vs NDVI) by Normalization Strategy ---")
        print(f"{'Method':<30} | {'ρ (RGB+NIR)':>12} | {'ρ (RGB-only)':>12}")
        print("-" * 60)
        
        # Baseline (no DA)
        rho_base_full, _ = spearmanr(all_eov_full_cont, all_ndvi_cont)
        rho_base_rgb, _  = spearmanr(all_eov_rgb_cont, all_ndvi_cont)
        print(f"{'No adaptation (baseline)':<30} | {rho_base_full:>12.4f} | {rho_base_rgb:>12.4f}")
        da_results["baseline"] = {"full": float(rho_base_full), "rgb": float(rho_base_rgb)}
        
        # Z-score
        rho_z_full, _ = spearmanr(all_eov_full_zscore, all_ndvi_cont)
        rho_z_rgb, _  = spearmanr(all_eov_rgb_zscore, all_ndvi_cont)
        delta_z_full = rho_z_full - rho_base_full
        delta_z_rgb  = rho_z_rgb - rho_base_rgb
        print(f"{'Z-score normalization':<30} | {rho_z_full:>12.4f} | {rho_z_rgb:>12.4f}  (Δ: {delta_z_full:+.4f} / {delta_z_rgb:+.4f})")
        da_results["zscore"] = {"full": float(rho_z_full), "rgb": float(rho_z_rgb)}
        
        # Histogram matching
        rho_hm_full, _ = spearmanr(all_eov_full_histmatch, all_ndvi_cont)
        rho_hm_rgb, _  = spearmanr(all_eov_rgb_histmatch, all_ndvi_cont)
        delta_hm_full = rho_hm_full - rho_base_full
        delta_hm_rgb  = rho_hm_rgb - rho_base_rgb
        print(f"{'Histogram matching':<30} | {rho_hm_full:>12.4f} | {rho_hm_rgb:>12.4f}  (Δ: {delta_hm_full:+.4f} / {delta_hm_rgb:+.4f})")
        da_results["histmatch"] = {"full": float(rho_hm_full), "rgb": float(rho_hm_rgb)}
        
        # --- C2: Blind-test F1 with DA ---
        print(f"\n--- C2: Blind-Test Macro F1 by Normalization Strategy ---")
        print(f"{'Method':<30} | {'F1 (RGB+NIR)':>12} | {'F1 (RGB-only)':>12}")
        print("-" * 60)
        
        f1_base_full = f1_score(all_gt_train, all_pred_full, average='macro', zero_division=0)
        f1_base_rgb  = f1_score(all_gt_train, all_pred_rgb, average='macro', zero_division=0)
        print(f"{'No adaptation (baseline)':<30} | {f1_base_full:>12.4f} | {f1_base_rgb:>12.4f}")
        
        f1_z_full = f1_score(all_gt_train, all_pred_full_zscore, average='macro', zero_division=0)
        f1_z_rgb  = f1_score(all_gt_train, all_pred_rgb_zscore, average='macro', zero_division=0)
        delta_f1_z_full = f1_z_full - f1_base_full
        delta_f1_z_rgb  = f1_z_rgb - f1_base_rgb
        print(f"{'Z-score normalization':<30} | {f1_z_full:>12.4f} | {f1_z_rgb:>12.4f}  (Δ: {delta_f1_z_full:+.4f} / {delta_f1_z_rgb:+.4f})")
        
        f1_hm_full = f1_score(all_gt_train, all_pred_full_histmatch, average='macro', zero_division=0)
        f1_hm_rgb  = f1_score(all_gt_train, all_pred_rgb_histmatch, average='macro', zero_division=0)
        delta_f1_hm_full = f1_hm_full - f1_base_full
        delta_f1_hm_rgb  = f1_hm_rgb - f1_base_rgb
        print(f"{'Histogram matching':<30} | {f1_hm_full:>12.4f} | {f1_hm_rgb:>12.4f}  (Δ: {delta_f1_hm_full:+.4f} / {delta_f1_hm_rgb:+.4f})")
        
        da_results["f1_blind"] = {
            "baseline": {"full": float(f1_base_full), "rgb": float(f1_base_rgb)},
            "zscore":   {"full": float(f1_z_full),    "rgb": float(f1_z_rgb)},
            "histmatch":{"full": float(f1_hm_full),   "rgb": float(f1_hm_rgb)},
        }

    # ================================================================
    # Save results
    # ================================================================
    output_data = {
        "dataset":                  dataset_key,
        "num_pairs":                len(pairs),
        "ndvi_thresholds_training": train_ndvi_thresholds,
        "ndvi_thresholds_target":   [round(t, 4) for t in target_ndvi_thresholds],
        "index_thresholds_target":  {n: idx_thresholds[n] for n in index_names},
        "f1_results_blind_test": {
            method: { "per_class_f1": res["per_class"], "macro_f1": res["macro"] }
            for method, res in f1_results_train.items()
        },
        "f1_results_calibrated_test": {
            method: { "per_class_f1": res["per_class"], "macro_f1": res["macro"] }
            for method, res in f1_results_target.items()
        },
        "spearman_correlations_eov": spearman_results,
        "domain_adaptation_results": da_results,
    }

    json_path = os.path.join(output_dir, "cross_domain_results.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nVisual comparisons saved to: {output_dir}")
    print(f"Results JSON:                {json_path}")

    return output_data