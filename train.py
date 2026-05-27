import os
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torchmetrics.classification import (
    MulticlassF1Score, MulticlassRecall,
    MulticlassPrecision, MulticlassAccuracy
)
from mobilnet.diceloss import HybridFocalDiceLoss
from mobilnet.experiment import log_predictions_to_wandb


class Trainer:
    def __init__(self, model, datamodule, cfg, class_weights=None, log_dir="logs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dm = datamodule
        self.log_dir = log_dir
        self.cfg = cfg
        self.class_names = ["dead", "severe", "moderate", "healthy"]

        os.makedirs(self.log_dir, exist_ok=True)

        cw = (torch.tensor(class_weights).float().to(self.device)
              if class_weights else None)
        self.criterion = HybridFocalDiceLoss(
            weight=cw,
            num_classes=4,
            focal_weight=cfg.get("focal_weight", 0.5),
            dice_weight=1.0 - cfg.get("focal_weight", 0.5),
            focal_gamma=cfg.get("focal_gamma", 2.0),
            label_smoothing=cfg.get("label_smoothing", 0.05),
            betas=cfg.get("betas", [0.5, 0.5, 0.5, 0.5]),
        )
        
        encoder_params = (
            list(self.model.enc0.parameters()) +
            list(self.model.enc1.parameters()) +
            list(self.model.enc2.parameters()) +
            list(self.model.enc3.parameters()) +
            list(self.model.bottleneck.parameters())
        )
        decoder_params = (
            list(self.model.bottleneck_compress.parameters()) +
            list(self.model.dec4.parameters()) +
            list(self.model.dec3.parameters()) +
            list(self.model.dec2.parameters()) +
            list(self.model.dec1.parameters()) +
            list(self.model.final.parameters())
        )
        self.opt = torch.optim.AdamW([
            {'params': encoder_params, 'lr': cfg["lr"] * 0.1},
            {'params': decoder_params, 'lr': cfg["lr"]},
            {'params': self.model.input_norm.parameters(), 'lr': cfg["lr"]},
        ], weight_decay=cfg["weight_decay"])
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                      self.opt, T_max=cfg['epochs'], eta_min=1e-6)
        self.scaler = torch.amp.GradScaler('cuda')
        self.best_f1 = 0.0
        self._autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        metric_args = {"num_classes": 4, "average": None}
        self.f1_metric = MulticlassF1Score(**metric_args).to(self.device)
        self.recall_metric = MulticlassRecall(**metric_args).to(self.device)
        self.precision_metric = MulticlassPrecision(**metric_args).to(self.device)
        self.accuracy_metric = MulticlassAccuracy(**metric_args).to(self.device)

    def _step(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        if len(loader) == 0:
            raise RuntimeError(
                f"The {'train' if train else 'val'} DataLoader is empty."
            )

        total_loss = 0

        with torch.set_grad_enabled(train):
            for x, y in tqdm(loader, leave=False,
                             desc="Train" if train else "Val"):
                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast(self._autocast_device):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)

                if train:
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()

                total_loss += loss.item()

                preds = logits.argmax(1)
                self.f1_metric.update(preds, y)
                self.recall_metric.update(preds, y)
                self.precision_metric.update(preds, y)
                self.accuracy_metric.update(preds, y)

        f1s = self.f1_metric.compute()
        recs = self.recall_metric.compute()
        pres = self.precision_metric.compute()
        accs = self.accuracy_metric.compute()

        self.f1_metric.reset()
        self.recall_metric.reset()
        self.precision_metric.reset()
        self.accuracy_metric.reset()

        metrics = {
            "loss": total_loss / len(loader),
            "macro_f1": f1s.mean().item(),
            "macro_recall": recs.mean().item(),
            "macro_precision": pres.mean().item(),
            "macro_accuracy": accs.mean().item(),
        }

        for i, name in enumerate(self.class_names):
            metrics[f"f1_{name}"] = f1s[i].item()
            metrics[f"recall_{name}"] = recs[i].item()
            metrics[f"precision_{name}"] = pres[i].item()
            metrics[f"accuracy_{name}"] = accs[i].item()

        return metrics

    def _rgb_only_eval(self, loader):
        """Evaluate with NIR channel zeroed out.
        Measures how well the RGB-only pathway performs — this is
        what you'll actually get at inference on unseen RGB images.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y in tqdm(loader, leave=False, desc="RGB-only Val"):
                x, y = x.to(self.device), y.to(self.device)

                # Zero out NIR channel (index 3)
                x_rgb = x.clone()
                x_rgb[:, 3:, :, :] = 0.0

                with torch.amp.autocast(self._autocast_device):
                    logits = self.model(x_rgb)
                    loss = self.criterion(logits, y)

                total_loss += loss.item()
                preds = logits.argmax(1)
                self.f1_metric.update(preds, y)

        f1s = self.f1_metric.compute()
        self.f1_metric.reset()

        metrics = {
            "loss": total_loss / len(loader),
            "macro_f1": f1s.mean().item(),
        }
        for i, name in enumerate(self.class_names):
            metrics[f"f1_{name}"] = f1s[i].item()

        return metrics

    def _tta_eval(self, loader, zero_nir=True):
        """Test-Time Augmentation evaluation.

        Runs 8 geometric augmentations per image (original + flips + 90° rotations),
        averages softmax probabilities, then takes argmax. Only uses transforms
        that are exactly invertible (no interpolation artifacts).

        The 8 augmentations:
          0: original
          1: horizontal flip
          2: vertical flip
          3: horizontal + vertical flip
          4: 90° rotation
          5: 90° rotation + horizontal flip
          6: 90° rotation + vertical flip
          7: 90° rotation + horizontal + vertical flip

        Args:
            loader: DataLoader
            zero_nir: if True, zeroes NIR channel (RGB-only evaluation)
        """
        self.model.eval()

        metric_kwargs = {"num_classes": 4, "average": None}
        tta_f1 = MulticlassF1Score(**metric_kwargs).to(self.device)

        def _apply_augment(x, idx):
            """Apply one of 8 geometric augmentations."""
            if idx & 1:  # horizontal flip
                x = x.flip(-1)
            if idx & 2:  # vertical flip
                x = x.flip(-2)
            if idx & 4:  # 90° rotation (rotate dims H,W)
                x = x.transpose(-2, -1)
            return x

        def _undo_augment(x, idx):
            """Reverse the augmentation on the output logits."""
            if idx & 4:  # undo 90° rotation
                x = x.transpose(-2, -1)
            if idx & 2:  # undo vertical flip
                x = x.flip(-2)
            if idx & 1:  # undo horizontal flip
                x = x.flip(-1)
            return x

        with torch.no_grad():
            for x, y in tqdm(loader, leave=False, desc="TTA Eval"):
                x, y = x.to(self.device), y.to(self.device)

                if zero_nir:
                    x = x.clone()
                    x[:, 3:, :, :] = 0.0

                # Accumulate softmax probabilities across 8 augmentations
                avg_probs = torch.zeros(
                    x.shape[0], 4, x.shape[2], x.shape[3],
                    device=self.device
                )

                for aug_idx in range(8):
                    x_aug = _apply_augment(x, aug_idx)

                    with torch.amp.autocast(self._autocast_device):
                        logits = self.model(x_aug)

                    # Undo the spatial transform on the output
                    logits = _undo_augment(logits, aug_idx)
                    avg_probs += F.softmax(logits, dim=1)

                avg_probs /= 8.0
                preds = avg_probs.argmax(1)
                tta_f1.update(preds, y)

        f1s = tta_f1.compute()

        metrics = {
            "macro_f1": f1s.mean().item(),
        }
        for i, name in enumerate(self.class_names):
            metrics[f"f1_{name}"] = f1s[i].item()

        return metrics

    def fit(self, epochs, early_stop_patience=8):
        print(f"🚀 Training on {self.device} | Classes: {self.class_names}")
        patience_counter = 0

        for e in range(epochs):
            train_m = self._step(self.dm.train_dataloader(), train=True)
            val_m = self._step(self.dm.val_dataloader(), train=False)

            # --- RGB-only evaluation (tracks generalization readiness) ---
            rgb_only_m = self._rgb_only_eval(self.dm.val_dataloader())

            self.scheduler.step()
            # WandB Logging
            log_dict = {"epoch": e, "lr": self.opt.param_groups[0]['lr']}
            for k, v in train_m.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_m.items():
                log_dict[f"val/{k}"] = v
            for k, v in rgb_only_m.items():
                log_dict[f"val_rgb_only/{k}"] = v
            wandb.log(log_dict)

            # --- DETAILED CONSOLE LOGGING ---
            print(f"\n" + "=" * 70)
            print(f"Epoch {e:02d} | Train Loss: {train_m['loss']:.4f} "
                  f"| Val F1: {val_m['macro_f1']:.4f} "
                  f"| RGB-only F1: {rgb_only_m['macro_f1']:.4f}")
            print(f"-" * 70)
            print(f"{'Class':<12} | {'Prec':<8} | {'Rec':<8} "
                  f"| {'F1':<8} | {'Acc':<8} | {'RGB-only F1':<12}")
            print(f"-" * 70)

            for n in self.class_names:
                p = val_m[f'precision_{n}']
                r = val_m[f'recall_{n}']
                f = val_m[f'f1_{n}']
                a = val_m[f'accuracy_{n}']
                rf = rgb_only_m[f'f1_{n}']
                print(f"{n:<12} | {p:.3f}    | {r:.3f}    "
                      f"| {f:.3f}    | {a:.3f}    | {rf:.3f}")
            print(f"=" * 70 + "\n")

            # Save best checkpoint based on RGB-only F1.
            # The thesis goal is robustness at RGB-only inference, so the
            # checkpoint that best predicts without NIR is the correct target.
            # Using full (RGB+NIR) F1 would save a model that relies on NIR,
            # which is not what gets deployed.
            if rgb_only_m['macro_f1'] > self.best_f1:
                self.best_f1 = rgb_only_m['macro_f1']
                torch.save(self.model.state_dict(),
                           os.path.join(self.log_dir, "moddrop_5_best_model.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"🛑 Early stopping triggered.")
                    break

    def test(self, num_plots=15):
        print("\n" + "🔍" * 5 + " STARTING AUTOMATIC EVALUATION " + "🔍" * 5)
        path = os.path.join(self.log_dir, "moddrop_5_best_model.pth")

        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            print(f"✅ Loaded best performing weights from: {path}")
        else:
            print("⚠️ No saved weights found; evaluating current state.")

        # Full evaluation (RGB + NIR)
        test_metrics = self._step(self.dm.test_dataloader(), train=False)

        # RGB-only evaluation (no TTA)
        rgb_only_test = self._rgb_only_eval(self.dm.test_dataloader())

        # TTA evaluation — RGB-only with 8 geometric augmentations
        print("Running TTA evaluation (8 augmentations)...")
        tta_rgb_test = self._tta_eval(self.dm.test_dataloader(), zero_nir=True)

        # TTA evaluation — Full (RGB+NIR) with 8 geometric augmentations
        tta_full_test = self._tta_eval(self.dm.test_dataloader(), zero_nir=False)

        # Log to WandB
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.log({f"test_rgb_only/{k}": v for k, v in rgb_only_test.items()})
        wandb.log({f"test_tta_rgb/{k}": v for k, v in tta_rgb_test.items()})
        wandb.log({f"test_tta_full/{k}": v for k, v in tta_full_test.items()})

        # Detailed Console Output
        print("\n" + "=" * 75)
        print(f"{'CLASS':<12} | {'Full F1':<9} | {'RGB F1':<9} | "
              f"{'TTA Full':<9} | {'TTA RGB':<9} | {'TTA gain':<9}")
        print("-" * 75)
        for name in self.class_names:
            f1 = test_metrics[f'f1_{name}']
            rf1 = rgb_only_test[f'f1_{name}']
            tf1 = tta_full_test[f'f1_{name}']
            trf1 = tta_rgb_test[f'f1_{name}']
            gain = trf1 - rf1
            print(f"{name:<12} | {f1:.4f}    | {rf1:.4f}    | "
                  f"{tf1:.4f}    | {trf1:.4f}    | {gain:+.4f}")
        print("-" * 75)
        full_gain = tta_full_test['macro_f1'] - test_metrics['macro_f1']
        rgb_gain = tta_rgb_test['macro_f1'] - rgb_only_test['macro_f1']
        print(f"{'MACRO':<12} | {test_metrics['macro_f1']:.4f}    | "
              f"{rgb_only_test['macro_f1']:.4f}    | "
              f"{tta_full_test['macro_f1']:.4f}    | "
              f"{tta_rgb_test['macro_f1']:.4f}    | {rgb_gain:+.4f}")
        print("=" * 75)
        print(f"TTA improvement: Full {full_gain:+.4f} | RGB-only {rgb_gain:+.4f}")
        print()

        log_predictions_to_wandb(
            self.model, self.dm.test_dataloader(), self.device,
            stats_path=self.dm.stats_path,
            num_plots=num_plots,
        )