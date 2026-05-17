"""
Optuna Hyperparameter Optimisation for Vegetation Segmentation.

Objective: maximise RGB-only val Macro-F1 (the deployment metric).

Two-phase strategy (run sequentially):
  Phase 1 — Optimiser + loss structure (lr, weight_decay, focal_weight,
            label_smoothing, focal_gamma).
  Phase 2 — Tversky betas (with ordering constraint: dead >= severe >= moderate >= healthy).
"""

import os
import gc
import json
import torch
import torch.nn.functional as F
import optuna
from optuna.pruners import MedianPruner
from tqdm import tqdm
from torchmetrics.classification import MulticlassF1Score

from mobilnet.experiment import DataModule
from mobilnet.model import EfficientUNet4Class
from mobilnet.diceloss import HybridFocalDiceLoss


# ---------------------------------------------------------------
# Stripped-down trainer for HPO (no WandB, no TTA, no test)
# ---------------------------------------------------------------
class _HPOTrainer:
    """Minimal trainer that returns RGB-only val F1 per epoch.

    Key differences from the full Trainer:
    - No WandB logging (HPO runs are disposable).
    - No TTA evaluation (too slow for HPO).
    - No test evaluation (test set is never touched during HPO).
    - Reports intermediate values to Optuna for pruning.
    """

    CLASS_NAMES = ["dead", "severe", "moderate", "healthy"]

    def __init__(self, model, datamodule, cfg, class_weights, loss_overrides):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dm = datamodule
        self._autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Loss with trial-specific overrides ---
        cw = (torch.tensor(class_weights).float().to(self.device)
              if class_weights else None)

        self.criterion = HybridFocalDiceLoss(
            weight=cw,
            focal_weight=loss_overrides["focal_weight"],
            dice_weight=1.0 - loss_overrides["focal_weight"],
            num_classes=4,
        )

        # Override betas
        self.criterion.beta_t = torch.tensor(
            loss_overrides["betas"], dtype=torch.float32
        ).to(self.device)

        # Override label_smoothing and gamma — stored for use in forward
        self._label_smoothing = loss_overrides["label_smoothing"]
        self._focal_gamma = loss_overrides["focal_gamma"]

        # --- Optimizer ---
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
            {"params": encoder_params, "lr": cfg["lr"] * 0.1},
            {"params": decoder_params, "lr": cfg["lr"]},
            {"params": self.model.input_norm.parameters(), "lr": cfg["lr"]},
        ], weight_decay=cfg["weight_decay"])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg["epochs"], eta_min=1e-6
        )
        self.scaler = torch.amp.GradScaler("cuda")

        self.f1_metric = MulticlassF1Score(
            num_classes=4, average=None
        ).to(self.device)

    def _patched_forward(self, inputs, targets):
        """Forward pass through the loss with trial-specific gamma and
        label_smoothing, without modifying the HybridFocalDiceLoss class."""
        device = inputs.device

        # --- Focal Loss (with trial gamma + label_smoothing) ---
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.criterion.weight,
            reduction="none",
            label_smoothing=self._label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self._focal_gamma * ce_loss).mean()

        # --- Tversky Loss (uses criterion's beta_t) ---
        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = (
            F.one_hot(targets, num_classes=4)
            .permute(0, 3, 1, 2)
            .float()
            .to(device)
        )
        dims = (0, 2, 3)
        intersection = torch.sum(inputs_soft * targets_one_hot, dims)
        fp = torch.sum(inputs_soft * (1.0 - targets_one_hot), dims)
        fn = torch.sum((1.0 - inputs_soft) * targets_one_hot, dims)

        beta_t = self.criterion.beta_t.to(device)
        alpha_t = 1.0 - beta_t

        tversky_index = (intersection + 1e-6) / (
            intersection + alpha_t * fp + beta_t * fn + 1e-6
        )
        dice_loss = (1.0 - tversky_index).mean()

        return (self.criterion.focal_weight * focal_loss +
                self.criterion.dice_weight * dice_loss)

    def _train_epoch(self):
        self.model.train()
        for x, y in self.dm.train_dataloader():
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(self._autocast_device):
                logits = self.model(x)
                loss = self._patched_forward(logits, y)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

    def _rgb_only_val_f1(self):
        """The ONLY metric that matters for HPO: RGB-only macro F1."""
        self.model.eval()
        self.f1_metric.reset()
        with torch.no_grad():
            for x, y in self.dm.val_dataloader():
                x, y = x.to(self.device), y.to(self.device)
                x_rgb = x.clone()
                x_rgb[:, 3:, :, :] = 0.0
                with torch.amp.autocast(self._autocast_device):
                    logits = self.model(x_rgb)
                self.f1_metric.update(logits.argmax(1), y)
        f1s = self.f1_metric.compute()
        return f1s.mean().item()

    def run(self, epochs, trial):
        """Train for `epochs`, reporting to Optuna for pruning."""
        best_f1 = 0.0
        patience = 0

        for e in range(epochs):
            self._train_epoch()
            self.scheduler.step()
            rgb_f1 = self._rgb_only_val_f1()

            # Report to Optuna for pruning
            trial.report(rgb_f1, e)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if rgb_f1 > best_f1:
                best_f1 = rgb_f1
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break

        return best_f1


# ---------------------------------------------------------------
# Optuna objective functions
# ---------------------------------------------------------------
def _make_phase1_objective(dm, class_weights, epochs, moddrop_prob):
    """Phase 1: optimise lr, weight_decay, focal_weight, label_smoothing, gamma.
    Betas are fixed at a neutral baseline (all 0.5 = standard Dice)."""

    def objective(trial):
        cfg = {
            "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "epochs": epochs,
        }
        focal_weight = trial.suggest_float("focal_weight", 0.2, 0.8)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)

        loss_overrides = {
            "focal_weight": focal_weight,
            "label_smoothing": label_smoothing,
            "focal_gamma": focal_gamma,
            # Neutral betas — standard Dice (alpha=beta=0.5) for all classes.
            # Phase 1 finds the best optimizer/loss-structure first.
            "betas": [0.5, 0.5, 0.5, 0.5],
        }

        model = EfficientUNet4Class(
            in_channels=4, num_classes=4,
            use_domain_head=False,
            moddrop_nir_prob=moddrop_prob,
        )

        trainer = _HPOTrainer(model, dm, cfg, class_weights, loss_overrides)
        best_f1 = trainer.run(epochs, trial)

        # Free GPU memory between trials
        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()

        return best_f1

    return objective


def _make_phase2_objective(dm, class_weights, epochs, moddrop_prob, phase1_best):
    """Phase 2: optimise per-class Tversky betas.

    Ordering constraint enforced:
        beta_dead >= beta_severe >= beta_moderate >= beta_healthy

    This encodes the domain assumption that missing stressed/dead vegetation
    is costlier than a false alarm — an agronomic cost argument, NOT a
    frequency argument (your classes are roughly balanced after GMM splitting).

    lr, weight_decay, focal_weight, label_smoothing, gamma are fixed
    at Phase 1 winners.
    """

    def objective(trial):
        # Phase 1 winners — fixed
        cfg = {
            "lr": phase1_best["lr"],
            "weight_decay": phase1_best["weight_decay"],
            "epochs": epochs,
        }

        # --- Betas with ordering constraint ---
        # Class order: 0=dead, 1=severe, 2=moderate, 3=healthy
        beta_dead = trial.suggest_float("beta_dead", 0.5, 0.9)
        beta_severe = trial.suggest_float("beta_severe", 0.4, beta_dead)
        beta_moderate = trial.suggest_float("beta_moderate", 0.3, beta_severe)
        beta_healthy = trial.suggest_float("beta_healthy", 0.2, beta_moderate)

        loss_overrides = {
            "focal_weight": phase1_best["focal_weight"],
            "label_smoothing": phase1_best["label_smoothing"],
            "focal_gamma": phase1_best["focal_gamma"],
            "betas": [beta_dead, beta_severe, beta_moderate, beta_healthy],
        }

        model = EfficientUNet4Class(
            in_channels=4, num_classes=4,
            use_domain_head=False,
            moddrop_nir_prob=moddrop_prob,
        )

        trainer = _HPOTrainer(model, dm, cfg, class_weights, loss_overrides)
        best_f1 = trainer.run(epochs, trial)

        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()

        return best_f1

    return objective


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------
def run_hpo(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    stats_path: str,
    log_dir: str,
    n_trials_phase1: int = 25,
    n_trials_phase2: int = 20,
    epochs_per_trial: int = 20,
    early_stop_patience: int = 5,
    moddrop_nir_prob: float = 0.5,
    batch_size: int = 8,
):
    """Run two-phase HPO and save results.

    Args:
        train_dir, val_dir, test_dir: dataset directories.
        stats_path: path to class_stats.json.
        log_dir: output directory for results JSON.
        n_trials_phase1: number of Optuna trials for optimizer/loss params.
        n_trials_phase2: number of Optuna trials for Tversky betas.
        epochs_per_trial: max epochs per trial (shorter than full training).
        early_stop_patience: patience within each trial.
        moddrop_nir_prob: NIR dropout probability (fixed, not searched).
        batch_size: batch size.

    Returns:
        dict with best config from both phases.
    """
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print("  HYPERPARAMETER OPTIMISATION")
    print(f"  Objective: RGB-only val Macro-F1 (deployment metric)")
    print(f"  Phase 1: {n_trials_phase1} trials (optimizer + loss structure)")
    print(f"  Phase 2: {n_trials_phase2} trials (Tversky betas)")
    print(f"  Epochs per trial: {epochs_per_trial}")
    print("=" * 70)

    # Shared DataModule — created once, reused across all trials
    dm = DataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        stats_path=stats_path,
    )
    class_weights = dm.get_class_weights()

    # ======================= PHASE 1 =======================
    print("\n" + "=" * 70)
    print("  PHASE 1: Optimizer + Loss Structure")
    print("=" * 70)

    study1 = optuna.create_study(
        study_name="phase1_optimizer_loss",
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=6),
    )

    study1.optimize(
        _make_phase1_objective(
            dm, class_weights, epochs_per_trial, moddrop_nir_prob
        ),
        n_trials=n_trials_phase1,
        show_progress_bar=True,
    )

    phase1_best = study1.best_params
    phase1_best_value = study1.best_value

    print(f"\n  Phase 1 best RGB-only F1: {phase1_best_value:.4f}")
    print(f"  Best params: {json.dumps(phase1_best, indent=2)}")

    # ======================= PHASE 2 =======================
    print("\n" + "=" * 70)
    print("  PHASE 2: Tversky Betas (with ordering constraint)")
    print("  Fixed from Phase 1: lr, weight_decay, focal_weight,")
    print("                      label_smoothing, focal_gamma")
    print("=" * 70)

    study2 = optuna.create_study(
        study_name="phase2_tversky_betas",
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=6),
    )

    study2.optimize(
        _make_phase2_objective(
            dm, class_weights, epochs_per_trial, moddrop_nir_prob, phase1_best
        ),
        n_trials=n_trials_phase2,
        show_progress_bar=True,
    )

    phase2_best = study2.best_params
    phase2_best_value = study2.best_value

    print(f"\n  Phase 2 best RGB-only F1: {phase2_best_value:.4f}")
    print(f"  Best params: {json.dumps(phase2_best, indent=2)}")

    # ======================= FINAL CONFIG =======================
    final_config = {
        "phase1": {
            "best_rgb_only_f1": phase1_best_value,
            "params": phase1_best,
        },
        "phase2": {
            "best_rgb_only_f1": phase2_best_value,
            "params": phase2_best,
        },
        "final_training_config": {
            "lr": phase1_best["lr"],
            "weight_decay": phase1_best["weight_decay"],
            "focal_weight": phase1_best["focal_weight"],
            "label_smoothing": phase1_best["label_smoothing"],
            "focal_gamma": phase1_best["focal_gamma"],
            "betas": [
                phase2_best["beta_dead"],
                phase2_best["beta_severe"],
                phase2_best["beta_moderate"],
                phase2_best["beta_healthy"],
            ],
            "moddrop_nir_prob": moddrop_nir_prob,
            "note": (
                "Use these values to run a full training "
                "(45 epochs, early stopping on RGB-only val F1). "
                "The HPO used shorter runs — final training may reach "
                "higher F1."
            ),
        },
    }

    out_path = os.path.join(log_dir, "hpo_results.json")
    with open(out_path, "w") as f:
        json.dump(final_config, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # ======================= SUMMARY =======================
    fc = final_config["final_training_config"]
    print("\n" + "=" * 70)
    print("  FINAL CONFIG FOR FULL TRAINING")
    print("=" * 70)
    print(f"  lr:              {fc['lr']:.6f}")
    print(f"  weight_decay:    {fc['weight_decay']:.6f}")
    print(f"  focal_weight:    {fc['focal_weight']:.4f}")
    print(f"  dice_weight:     {1.0 - fc['focal_weight']:.4f}")
    print(f"  label_smoothing: {fc['label_smoothing']:.4f}")
    print(f"  focal_gamma:     {fc['focal_gamma']:.4f}")
    print(f"  betas (dead):    {fc['betas'][0]:.4f}")
    print(f"  betas (severe):  {fc['betas'][1]:.4f}")
    print(f"  betas (moderate):{fc['betas'][2]:.4f}")
    print(f"  betas (healthy): {fc['betas'][3]:.4f}")
    print("=" * 70)

    return final_config