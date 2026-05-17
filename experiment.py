import json
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from mobilnet.dataset import VegetationImageDataset


# ---------------------------------------------------------------------------
# Standard NDVI colormap
# Red → Yellow → Green palette matching remote sensing conventions.
# ---------------------------------------------------------------------------
_NDVI_STOPS = [
    (-1.00, "#8B0000"),   # dark red    — bare/dead
    (-0.20, "#DC143C"),   # crimson     — very stressed
    ( 0.00, "#FFD700"),   # gold/yellow — sparse vegetation
    ( 0.20, "#ADFF2F"),   # yellow-green — moderate stress
    ( 0.50, "#228B22"),   # forest green — moderate healthy
    ( 1.00, "#006400"),   # dark green   — dense healthy
]

def _build_ndvi_cmap():
    positions = [(v + 1.0) / 2.0 for v, _ in _NDVI_STOPS]
    colors    = [c for _, c in _NDVI_STOPS]
    return mcolors.LinearSegmentedColormap.from_list(
        "ndvi_standard", list(zip(positions, colors)), N=512
    )

NDVI_CMAP = _build_ndvi_cmap()
NDVI_VMIN, NDVI_VMAX = -0.5, 0.8


class DataModule:
    def __init__(self, train_dir, val_dir, test_dir, batch_size, stats_path):
        self.bs         = batch_size
        self.stats_path = stats_path

        self.train_ds = VegetationImageDataset(train_dir, stats_path, augment=True)
        self.val_ds   = VegetationImageDataset(val_dir,   stats_path, augment=False)
        self.test_ds  = VegetationImageDataset(test_dir,  stats_path, augment=False)

        with open(stats_path, "r") as f:
            self._stats = json.load(f)

        self.weights = self._stats["train"]["calculated_weights"]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.bs, shuffle=True,
            drop_last=False, num_workers=2, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.bs, num_workers=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.bs, num_workers=2, pin_memory=True)

    def get_class_weights(self):
        return self.weights


def _to_rgb(tensor_chw: torch.Tensor) -> np.ndarray:
    """Convert first 3 channels to uint8 HxWx3 for display."""
    img = tensor_chw[:3].permute(1, 2, 0).numpy()
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def _compute_ndvi(tensor_chw: torch.Tensor) -> np.ndarray:
    """Compute NDVI from Red (ch 0) and NIR (ch 3). Returns (H, W) in [-1, 1]."""
    red = tensor_chw[0].numpy()
    nir = tensor_chw[3].numpy()
    return (nir - red) / (nir + red + 1e-8)


def _eov_to_ndvi_scale(eov: np.ndarray) -> np.ndarray:
    """Rescale Expected Ordinal Value [0, 3] → NDVI display range [-0.5, 0.8]."""
    return (eov / 3.0) * (NDVI_VMAX - NDVI_VMIN) + NDVI_VMIN


def _add_colorbar(fig, im, ax):
    """Add a colorbar with black tick labels on white background."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="black")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black", fontsize=8)


def log_predictions_to_wandb(model, loader, device, stats_path: str, num_plots: int = 15):
    """
    Log 4-panel comparison plots to WandB.

    Health score uses Expected Ordinal Value (EOV):
        EOV = Σ i·P(class=i),  i ∈ {0,1,2,3}
    rescaled to the NDVI display range for visual comparison with GT NDVI.
    This is the same score used in the Spearman ρ evaluation — making
    figures and quantitative metrics consistent.
    """
    model.eval()
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

    class_indices = torch.tensor(
        [0., 1., 2., 3.], dtype=torch.float32, device=device
    ).view(1, 4, 1, 1)

    plot_count = 0

    with torch.no_grad():
        for x, y in loader:
            if plot_count >= num_plots:
                break

            x_dev = x.to(device)

            with torch.amp.autocast(autocast_device):
                probs_full = torch.softmax(model(x_dev), dim=1)
                eov_full   = (probs_full * class_indices).sum(dim=1).cpu().numpy()

            x_rgb = x_dev.clone()
            x_rgb[:, 3:, :, :] = 0.0
            with torch.amp.autocast(autocast_device):
                probs_rgb = torch.softmax(model(x_rgb), dim=1)
                eov_rgb   = (probs_rgb * class_indices).sum(dim=1).cpu().numpy()

            for b_idx in range(x.shape[0]):
                if plot_count >= num_plots:
                    break

                fig, axes = plt.subplots(1, 4, figsize=(24, 5))
                fig.patch.set_facecolor("white")

                titles = [
                    "RGB Input",
                    "Ground-Truth NDVI",
                    "Predicted Health\n(RGB + NIR)",
                    "Predicted Health\n(RGB only)",
                ]

                axes[0].imshow(_to_rgb(x[b_idx]))

                ndvi = _compute_ndvi(x[b_idx])
                im_ndvi = axes[1].imshow(
                    ndvi, cmap=NDVI_CMAP, vmin=NDVI_VMIN, vmax=NDVI_VMAX
                )
                _add_colorbar(fig, im_ndvi, axes[1])

                health_full = _eov_to_ndvi_scale(eov_full[b_idx])
                im_full = axes[2].imshow(
                    health_full, cmap=NDVI_CMAP, vmin=NDVI_VMIN, vmax=NDVI_VMAX
                )
                _add_colorbar(fig, im_full, axes[2])

                health_rgb = _eov_to_ndvi_scale(eov_rgb[b_idx])
                im_rgb = axes[3].imshow(
                    health_rgb, cmap=NDVI_CMAP, vmin=NDVI_VMIN, vmax=NDVI_VMAX
                )
                _add_colorbar(fig, im_rgb, axes[3])

                for ax, title in zip(axes, titles):
                    ax.set_title(title, fontsize=11, fontweight="bold",
                                 color="black", pad=10)
                    ax.axis("off")

                fig.suptitle(
                    f"Sample {plot_count}",
                    fontsize=13, fontweight="bold", color="black", y=0.98
                )

                plt.tight_layout(rect=[0, 0, 1, 0.93])

                wandb.log({
                    f"Evaluation/Sample_{plot_count}": wandb.Image(
                        fig, caption=f"Sample {plot_count}"
                    )
                })
                plt.close(fig)

                plot_count += 1