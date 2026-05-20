
import os
import cv2
import json
import torch
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score

from mobilnet.model import EfficientUNet4Class
from mobilnet.baseline_indices import compute_indices, fit_index_thresholds

NOISE_SEED = 42


def apply_gamma(rgb, gamma):
    return np.clip(rgb ** gamma, 0.0, 1.0)


def apply_wb_shift(rgb, red_gain):
    out = rgb.copy()
    out[:, :, 0] = np.clip(out[:, :, 0] * red_gain, 0.0, 1.0)
    return out


def apply_noise(rgb, sigma, rng):
    noise = rng.standard_normal(rgb.shape).astype(np.float32) * sigma
    return np.clip(rgb + noise, 0.0, 1.0)


def apply_jpeg(rgb, quality):
    u8  = (rgb * 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode('.jpg', bgr,
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def apply_downsample(rgb, factor):
    if factor <= 1:
        return rgb
    h, w   = rgb.shape[:2]
    small  = cv2.resize(rgb, (w // factor, h // factor),
                        interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)



def camera_robust_preprocess(rgb_f):
    rgb_np = (rgb_f * 255).astype(np.float32)
    avg_all = rgb_np.mean()
    for c in range(3):
        avg_c = rgb_np[:, :, c].mean()
        if avg_c > 1e-6:
            rgb_np[:, :, c] *= avg_all / avg_c
    rgb_np = np.clip(rgb_np, 0, 255)

    u8  = rgb_np.astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0



def _load_or_fit_vari_thresholds(stats_path, train_dir):
    """
        stats_path: Path to class_stats.json.
        train_dir:  Training image directory.  Required only on cache miss.
                    Pass None to raise instead of fitting.

    Returns:
        List of 3 ascending threshold floats.
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    cached = stats.get('train', {}).get('vari_thresholds')
    if cached is not None:
        print(f"VARI thresholds loaded from JSON: "
              f"{[round(t, 4) for t in cached]}")
        return cached

    if train_dir is None:
        raise ValueError(
            "stats_path does not contain 'train.vari_thresholds' and "
            "train_dir was not provided.  Either run "
            "baseline_indices.run_baseline() first, or pass train_dir= "
            "so thresholds can be fitted automatically."
        )

    print("VARI thresholds not in JSON — fitting on training set...")
    thresholds = fit_index_thresholds(train_dir, "VARI", plot_dir=None)

    # Persist for all future runs
    stats['train']['vari_thresholds'] = [round(t, 6) for t in thresholds]
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"VARI thresholds saved to {stats_path} under train.vari_thresholds")

    return thresholds


def load_test_samples(test_dir, stats_path, image_size=256, max_samples=None):
    """
    Ground truth uses training-domain NDVI thresholds (stats['train']['thresholds']).

    Returns:
        samples:         list of dicts with keys 'rgb', 'nir', 'mask'.
        ndvi_thresholds: the 3-float list used to compute masks.
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    ndvi_thresholds = stats['train']['thresholds']

    all_files = sorted(os.listdir(test_dir))
    rgb_files = [f for f in all_files
                 if "_RGB" in f.upper()
                 and f.lower().endswith(('.png', '.jpg', '.tif'))]

    if max_samples:
        rgb_files = rgb_files[:max_samples]

    samples = []
    for f in tqdm(rgb_files, desc="Loading test samples"):
        base     = f.upper().split("_RGB")[0]
        nir_name = next((n for n in all_files
                         if n.upper().startswith(base + "_NIR")), None)
        if not nir_name:
            continue

        rgb = cv2.cvtColor(cv2.imread(os.path.join(test_dir, f)),
                           cv2.COLOR_BGR2RGB)
        nir_raw = cv2.imread(os.path.join(test_dir, nir_name),
                             cv2.IMREAD_UNCHANGED)
        if nir_raw is None:
            nir_raw = tifffile.imread(os.path.join(test_dir, nir_name))
        if nir_raw.ndim == 3:
            nir_raw = nir_raw[:, :, 0]

        rgb_f = rgb.astype(np.float32) / 255.0
        nir_f = nir_raw.astype(np.float32) / (
            65535.0 if nir_raw.dtype == np.uint16 else 255.0
        )

        rgb_f = cv2.resize(rgb_f, (image_size, image_size))
        nir_f = cv2.resize(nir_f, (image_size, image_size))

        red  = rgb_f[:, :, 0]
        ndvi = (nir_f - red) / (nir_f + red + 1e-6)
        mask = np.zeros_like(ndvi, dtype=np.int64)
        mask[ndvi >= ndvi_thresholds[0]] = 1
        mask[ndvi >= ndvi_thresholds[1]] = 2
        mask[ndvi >= ndvi_thresholds[2]] = 3

        samples.append({'rgb': rgb_f, 'nir': nir_f, 'mask': mask})

    print(f"Loaded {len(samples)} test samples")
    return samples, ndvi_thresholds



def _classify_vari(rgb_f, vari_thresholds):
    """Classify pixels via VARI using compute_indices (clipped, canonical)."""
    vari = compute_indices(rgb_f)["VARI"]
    mask = np.zeros(vari.shape, dtype=np.int64)
    mask[vari >= vari_thresholds[0]] = 1
    mask[vari >= vari_thresholds[1]] = 2
    mask[vari >= vari_thresholds[2]] = 3
    return mask


def evaluate_distortion(model, device, samples, distort_fn, distort_param,
                        vari_thresholds, noise_rng=None):

    all_gt          = []
    all_cnn_pred    = []
    all_vari_raw    = []
    all_vari_prep   = []
    SUBSAMPLE = 15  

    for s in samples:
        if noise_rng is not None and distort_fn is apply_noise:
            rgb_dist = distort_fn(s['rgb'], distort_param, noise_rng)
        else:
            rgb_dist = distort_fn(s['rgb'], distort_param)

        rgb_prep = rgb_dist
        rgb_t    = torch.from_numpy(rgb_prep.transpose(2, 0, 1))
        nir_zero = torch.zeros(1, rgb_t.shape[1], rgb_t.shape[2])
        x        = torch.cat([rgb_t, nir_zero], dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1).cpu().numpy()[0]

        vari_raw_pred  = _classify_vari(rgb_dist, vari_thresholds)
        vari_prep_pred = _classify_vari(rgb_prep,  vari_thresholds)

        all_gt.append(s['mask'].flatten()[::SUBSAMPLE])
        all_cnn_pred.append(pred.flatten()[::SUBSAMPLE])
        all_vari_raw.append(vari_raw_pred.flatten()[::SUBSAMPLE])
        all_vari_prep.append(vari_prep_pred.flatten()[::SUBSAMPLE])

    all_gt        = np.concatenate(all_gt)
    all_cnn_pred  = np.concatenate(all_cnn_pred)
    all_vari_raw  = np.concatenate(all_vari_raw)
    all_vari_prep = np.concatenate(all_vari_prep)

    kw = dict(average='macro', zero_division=0)
    return (
        f1_score(all_gt, all_cnn_pred,  **kw),
        f1_score(all_gt, all_vari_raw,  **kw),
        f1_score(all_gt, all_vari_prep, **kw),
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run_robustness_test(model_path, test_dir, stats_path,
                        train_dir=None,
                        output_dir="/content/robustness_results",
                        max_samples=1612):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = EfficientUNet4Class(in_channels=4, num_classes=4,
                                use_domain_head=False, moddrop_nir_prob=0.5)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")

    samples, ndvi_thresholds = load_test_samples(
        test_dir, stats_path, max_samples=max_samples
    )

    vari_thresholds = _load_or_fit_vari_thresholds(stats_path, train_dir)

    noise_rng = np.random.default_rng(NOISE_SEED)

    print("\nComputing undistorted baseline...")
    base_cnn, base_vari_raw, base_vari_prep = evaluate_distortion(
        model, device, samples,
        distort_fn=lambda rgb, _: rgb, distort_param=None,
        vari_thresholds=vari_thresholds,
    )
    print(f"Baseline — CNN: {base_cnn:.4f}, "
          f"VARI (raw): {base_vari_raw:.4f}, "
          f"VARI (preprocessed): {base_vari_prep:.4f}")

    experiments = {
        "Gamma Correction": {
            "params":         [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0],
            "fn":             apply_gamma,
            "xlabel":         "Gamma value (γ)",
        },
        "White Balance Shift": {
            "params":         [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            "fn":             apply_wb_shift,
            "xlabel":         "Red channel gain",
        },
        "Gaussian Noise": {
            "params":         [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15],
            "fn":             apply_noise,
            "xlabel":         f"Noise σ  (seed={NOISE_SEED})",
        },
        "JPEG Compression": {
            "params":         [100, 90, 70, 50, 30, 20, 10],
            "fn":             apply_jpeg,
            "xlabel":         "JPEG quality",
        },
        "Resolution Downsampling": {
            "params":         [1, 2, 3, 4, 6, 8],
            "fn":             apply_downsample,
            "xlabel":         "Downsample factor",
        },
    }

    all_results = {}

    for exp_name, cfg in experiments.items():
        print(f"\n{'='*60}")
        print(f"  {exp_name}")
        print(f"{'='*60}")

        cnn_scores, vari_raw_scores, vari_prep_scores = [], [], []
        exp_rng = np.random.default_rng(NOISE_SEED)

        for param in cfg["params"]:
            print(f"  {cfg['xlabel']} = {param} ...", end=" ", flush=True)
            cnn_f1, vari_raw_f1, vari_prep_f1 = evaluate_distortion(
                model, device, samples,
                distort_fn=cfg["fn"], distort_param=param,
                vari_thresholds=vari_thresholds,
                noise_rng=exp_rng if cfg["fn"] is apply_noise else None,
            )
            cnn_scores.append(cnn_f1)
            vari_raw_scores.append(vari_raw_f1)
            vari_prep_scores.append(vari_prep_f1)
            print(f"CNN: {cnn_f1:.4f}  VARI(raw): {vari_raw_f1:.4f}  "
                  f"VARI(prep): {vari_prep_f1:.4f}")

        all_results[exp_name] = {
            "params":               cfg["params"],
            "cnn_f1":               cnn_scores,
            "vari_raw_f1":          vari_raw_scores,
            "vari_preprocessed_f1": vari_prep_scores,
        }

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = range(len(cfg["params"]))
        ax.plot(xs, cnn_scores,       'b-o',  lw=2, ms=6, label='CNN (RGB-only)')
        ax.plot(xs, vari_raw_scores,  'r--s', lw=2, ms=6, label='VARI (raw)')
        ax.plot(xs, vari_prep_scores, 'g-.^', lw=2, ms=6, label='VARI (preprocessed)')
        ax.set_xticks(xs)
        ax.set_xticklabels([str(p) for p in cfg["params"]])
        ax.set_xlabel(cfg["xlabel"], fontsize=12)
        ax.set_ylabel("Macro F1", fontsize=12)
        ax.set_title(f"Robustness: {exp_name}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        all_f1 = cnn_scores + vari_raw_scores + vari_prep_scores
        ax.set_ylim([0, max(all_f1) * 1.15])
        plt.tight_layout()
        plot_path = os.path.join(
            output_dir,
            f"robustness_{exp_name.lower().replace(' ', '_')}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Summary table
    print(f"\n{'='*100}")
    print(f"  ROBUSTNESS SUMMARY  (Δ relative to undistorted baseline)")
    print(f"{'='*100}")
    for exp_name, res in all_results.items():
        print(f"\n{exp_name}:")
        print(f"  {'Param':<12} | {'CNN F1':<8} | {'VARI raw':<8} | "
              f"{'VARI prep':<9} | {'CNN Δ':<8} | "
              f"{'VARI raw Δ':<10} | {'VARI prep Δ':<11}")
        print(f"  {'-'*86}")
        for i, param in enumerate(res['params']):
            print(f"  {str(param):<12} | "
                  f"{res['cnn_f1'][i]:.4f}   | "
                  f"{res['vari_raw_f1'][i]:.4f}   | "
                  f"{res['vari_preprocessed_f1'][i]:.4f}    | "
                  f"{res['cnn_f1'][i] - base_cnn:+.4f}  | "
                  f"{res['vari_raw_f1'][i] - base_vari_raw:+.4f}     | "
                  f"{res['vari_preprocessed_f1'][i] - base_vari_prep:+.4f}")

    json_path = os.path.join(output_dir, "robustness_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n📊 Plots saved to:  {output_dir}")
    print(f"📊 Results JSON:    {json_path}")

    return all_results