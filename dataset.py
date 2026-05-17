

import os
import cv2
import torch
import numpy as np
import tifffile
import random
import json
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class VegetationImageDataset(Dataset):
    def __init__(self, root_dir, stats_path, image_size=256, augment=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.samples = self._load_samples()

        self.thresholds = None
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                if 'train' in stats and 'thresholds' in stats['train']:
                    self.thresholds = stats['train']['thresholds']

        if self.thresholds is None:
            raise FileNotFoundError(
                f"NDVI thresholds not found in '{stats_path}'. "
                
            )

    def _load_samples(self):
        all_files = sorted(os.listdir(self.root_dir))
        rgb_files = [f for f in all_files
                     if "_RGB" in f.upper()
                     and f.lower().endswith(('.png', '.jpg', '.tif'))]

        samples = []
        for f in rgb_files:
            base = f.upper().split("_RGB")[0]
            nir = next((n for n in all_files
                        if n.upper().startswith(base + "_NIR")), None)
            if nir:
                samples.append({
                    "rgb": os.path.join(self.root_dir, f),
                    "nir": os.path.join(self.root_dir, nir),
                })
            else:
                print(f"⚠️ Warning: Found RGB {f} but no matching NIR "
                      f"starting with {base}_NIR")

        if len(samples) == 0:
            print(f"❌ Critical: No valid RGB/NIR pairs in {self.root_dir}")
            print(f"Directory contains: {all_files[:5]}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        rgb = cv2.cvtColor(cv2.imread(s["rgb"]), cv2.COLOR_BGR2RGB)
        nir_raw = cv2.imread(s["nir"], cv2.IMREAD_UNCHANGED)
        if nir_raw is None:
            nir_raw = tifffile.imread(s["nir"])
        if nir_raw.ndim == 3:
            nir_raw = nir_raw[:, :, 0]

        rgb_f = rgb.astype(np.float32) / 255.0
        nir_max = 65535.0 if nir_raw.dtype == np.uint16 else 255.0
        nir_f = nir_raw.astype(np.float32) / nir_max

        rgb_t = torch.from_numpy(rgb_f.transpose(2, 0, 1))
        nir_t = torch.from_numpy(nir_f).unsqueeze(0)

        rgb_t = TF.resize(rgb_t, [self.image_size, self.image_size],
                          InterpolationMode.BILINEAR, antialias=True)
        nir_t = TF.resize(nir_t, [self.image_size, self.image_size],
                          InterpolationMode.BILINEAR, antialias=True)

        red = rgb_t[0]
        nir_ch = nir_t[0]
        ndvi = (nir_ch - red) / (nir_ch + red + 1e-6)

        mask = torch.zeros_like(ndvi, dtype=torch.long)
        mask[ndvi >= self.thresholds[0]] = 1
        mask[ndvi >= self.thresholds[1]] = 2
        mask[ndvi >= self.thresholds[2]] = 3

        if self.augment:
            if random.random() > 0.5:
                rgb_t = TF.hflip(rgb_t)
                nir_t = TF.hflip(nir_t)
                mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

            if random.random() > 0.5:
                rgb_t = TF.vflip(rgb_t)
                nir_t = TF.vflip(nir_t)
                mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

            if random.random() > 0.5:
                angle = random.uniform(-180, 180)
                rgb_t = TF.rotate(rgb_t, angle,
                                  interpolation=InterpolationMode.BILINEAR)
                nir_t = TF.rotate(nir_t, angle,
                                  interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask.unsqueeze(0).float(), angle,
                                 interpolation=InterpolationMode.NEAREST
                                 ).squeeze(0).long()

        if self.augment:
            rgb_t, nir_t = self._photometric_augment(rgb_t, nir_t)

        rgb_t = self._camera_robust_preprocess(rgb_t)

        features = torch.cat([rgb_t, nir_t], dim=0)

        return features, mask

    def _camera_robust_preprocess(self, rgb_t):

        rgb_np = (rgb_t.permute(1, 2, 0).numpy() * 255).astype(np.float32)


        avg_all = rgb_np.mean()
        for c in range(3):
            avg_c = rgb_np[:, :, c].mean()
            if avg_c > 1e-6:
                rgb_np[:, :, c] *= (avg_all / avg_c)
        rgb_np = np.clip(rgb_np, 0, 255)

        # --- CLAHE on L channel in LAB space ---
        u8 = rgb_np.astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        rgb_out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return torch.from_numpy(rgb_out.astype(np.float32) / 255.0).permute(2, 0, 1)

    # -----------------------------------------------------------------
    # Photometric augmentation
    # -----------------------------------------------------------------
    def _photometric_augment(self, rgb_t, nir_t):

        if random.random() > 0.5:
            bright = random.uniform(0.8, 1.2)
            rgb_t = torch.clamp(rgb_t * bright, 0, 1)
            nir_t = torch.clamp(nir_t * bright, 0, 1)

        if random.random() > 0.4:
            gains = [random.uniform(0.85, 1.15) for _ in range(3)]
            for c in range(3):
                rgb_t[c] = torch.clamp(rgb_t[c] * gains[c], 0, 1)

        
        if random.random() > 0.5:
            gamma = random.uniform(0.7, 1.4)
            rgb_t = torch.clamp(rgb_t.pow(gamma), 0, 1)

        if random.random() > 0.5:
            contrast = random.uniform(0.85, 1.15)
            mean_rgb = rgb_t.mean(dim=(1, 2), keepdim=True)
            rgb_t = torch.clamp((rgb_t - mean_rgb) * contrast + mean_rgb, 0, 1)


        if random.random() > 0.7:
            rgb_t = self._simulate_jpeg_artifacts(rgb_t)

        if random.random() > 0.5:
            rgb_noise = random.uniform(0.005, 0.025)
            nir_noise = random.uniform(0.005, 0.02)
            rgb_t = torch.clamp(rgb_t + torch.randn_like(rgb_t) * rgb_noise, 0, 1)
            nir_t = torch.clamp(nir_t + torch.randn_like(nir_t) * nir_noise, 0, 1)

        if random.random() > 0.7:
            rgb_t, nir_t = self._simulate_shadow(rgb_t, nir_t)

        return rgb_t, nir_t

    def _simulate_jpeg_artifacts(self, rgb_t):
        rgb_np = (rgb_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        quality = random.randint(30, 80)
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(decoded.astype(np.float32) / 255.0).permute(2, 0, 1)

    def _simulate_shadow(self, rgb_t, nir_t):
        _, H, W = rgb_t.shape
        sh = random.randint(int(H * 0.1), int(H * 0.4))
        sw = random.randint(int(W * 0.1), int(W * 0.4))
        y0 = random.randint(0, H - sh)
        x0 = random.randint(0, W - sw)

        shadow_rgb = random.uniform(0.4, 0.7)
        shadow_nir = random.uniform(0.6, 0.85)

        rgb_t[:, y0:y0 + sh, x0:x0 + sw] *= shadow_rgb
        nir_t[:, y0:y0 + sh, x0:x0 + sw] *= shadow_nir
        return rgb_t, nir_t