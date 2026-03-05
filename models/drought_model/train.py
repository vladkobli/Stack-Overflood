"""
Training loop with validation, metrics, early stopping and
checkpoint saving.

Includes:
- Label smoothing for better calibration
- Mixup augmentation for regularisation
- Cosine annealing with optional warmup
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from .config import Config
from .model import FloodFusionNet

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Binary focal loss for hard-example mining.

    Reduces the loss contribution of easy (well-classified) examples
    so the model focuses on hard, borderline cases.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        # p_t = probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        return (alpha_t * focal_weight * bce).mean()


# ────────────────────────────────────────────────────────────────────
# Metric helpers
# ────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute a full set of binary-classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"TP": tp, "FP": fp, "TN": tn, "FN": fn})
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Search for the threshold that maximises F1 score."""
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.01):
        preds = (y_prob >= thr).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    return round(float(best_thr), 2)


def _fmt_metrics(m: Dict[str, float]) -> str:
    parts = []
    for k in ("accuracy", "f1", "precision", "recall", "auc_roc", "threshold"):
        if k in m:
            parts.append(f"{k}={m[k]:.4f}")
    return "  ".join(parts)


# ────────────────────────────────────────────────────────────────────
# Training engine
# ────────────────────────────────────────────────────────────────────

class Trainer:
    """Encapsulates the full training + validation loop."""

    def __init__(self, cfg: Config, model: FloodFusionNet) -> None:
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)

        # Loss – supports optional class-weight for imbalanced data
        pos_w = (
            torch.tensor([cfg.pos_weight], device=self.device)
            if cfg.pos_weight is not None
            else None
        )
        if cfg.use_focal_loss:
            self.criterion = FocalLoss(
                gamma=cfg.focal_gamma,
                alpha=cfg.focal_alpha,
                pos_weight=pos_w,
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        # Label smoothing params
        self.label_smoothing = cfg.label_smoothing

        # Mixup params
        self.mixup_alpha = cfg.mixup_alpha

        # Optimiser
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # LR scheduler – cosine annealing with warmup
        self.warmup_epochs = cfg.warmup_epochs
        self.use_cosine = cfg.use_cosine_annealing
        if self.use_cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(cfg.epochs - self.warmup_epochs, 1),
                eta_min=cfg.learning_rate * 0.01,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=cfg.scheduler_factor,
                patience=cfg.scheduler_patience,
            )

        # ── SWA ─────────────────────────────────────────────────────
        self.use_swa = cfg.use_swa
        self.swa_start = max(1, int(cfg.epochs * cfg.swa_start_frac))
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(
                self.optimizer, swa_lr=cfg.swa_lr,
            )
        else:
            self.swa_model = None
            self.swa_scheduler = None

        # Book-keeping
        self.best_val_loss = float("inf")
        self.best_f1 = 0.0
        self.best_threshold = 0.5
        self.epochs_no_improve = 0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
        }

    # ── label smoothing helper ───────────────────────────────────────

    def _smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing: 0 → ε, 1 → 1-ε."""
        if self.label_smoothing > 0:
            return labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return labels

    # ── mixup helper ─────────────────────────────────────────────────

    def _mixup_batch(
        self,
        temporal: torch.Tensor,
        static: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Apply mixup: blend pairs of samples with a random λ."""
        if self.mixup_alpha <= 0:
            return temporal, static, labels

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1 - lam)  # ensure lam >= 0.5

        batch_size = temporal.size(0)
        perm = torch.randperm(batch_size, device=temporal.device)

        temporal = lam * temporal + (1 - lam) * temporal[perm]
        static = lam * static + (1 - lam) * static[perm]
        labels = lam * labels + (1 - lam) * labels[perm]
        return temporal, static, labels

    # ── warmup LR adjustment ────────────────────────────────────────

    def _warmup_lr(self, epoch: int) -> None:
        """Linearly ramp LR from 0 → base_lr over warmup_epochs."""
        if self.warmup_epochs <= 0 or epoch > self.warmup_epochs:
            return
        scale = epoch / self.warmup_epochs
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.cfg.learning_rate * scale

    # ── single epoch ─────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for temporal, static, labels in loader:
            temporal = temporal.to(self.device)
            static = static.to(self.device)
            labels = labels.to(self.device)

            # Mixup
            temporal, static, labels = self._mixup_batch(
                temporal, static, labels
            )
            # Label smoothing
            labels = self._smooth_labels(labels)

            self.optimizer.zero_grad()
            logits = self.model(temporal, static).squeeze(-1)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        return running_loss / max(n_samples, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        n_samples = 0
        all_probs, all_labels = [], []

        for temporal, static, labels in loader:
            temporal = temporal.to(self.device)
            static = static.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(temporal, static).squeeze(-1)
            loss = self.criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

        val_loss = running_loss / max(n_samples, 1)
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        opt_thr = find_optimal_threshold(y_true, y_prob)
        metrics = compute_metrics(y_true, y_prob, threshold=opt_thr)
        return val_loss, metrics

    # ── full training run ────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, list]:
        """Run the training loop for ``cfg.epochs`` epochs.

        Returns the training history dict.
        """
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Training on device: %s", self.device)
        logger.info(
            "Model parameters: %s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()

            # Warmup LR for the first few epochs
            self._warmup_lr(epoch)

            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)

            # Step scheduler (after warmup)
            in_swa = self.use_swa and epoch >= self.swa_start
            if in_swa:
                # During SWA phase: use flat SWA LR and update averaged model
                self.swa_model.update_parameters(self.model)  # type: ignore[union-attr]
                self.swa_scheduler.step()                      # type: ignore[union-attr]
            elif epoch > self.warmup_epochs:
                if self.use_cosine:
                    self.scheduler.step()
                else:
                    self.scheduler.step(val_loss)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            logger.info(
                "Epoch %3d/%d  │  train_loss=%.4f  val_loss=%.4f  │ "
                "%s  │  lr=%.1e  (%.1fs)",
                epoch,
                self.cfg.epochs,
                train_loss,
                val_loss,
                _fmt_metrics(val_metrics),
                lr,
                elapsed,
            )

            # ── checkpointing / early stopping ──────────────────────
            val_f1 = val_metrics.get("f1", 0.0)
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_val_loss = val_loss
                self.best_threshold = val_metrics.get("threshold", 0.5)
                self.epochs_no_improve = 0
                self._save_checkpoint()
                logger.info("  ↳ Saved best model (f1=%.4f, val_loss=%.4f)", val_f1, val_loss)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.cfg.early_stop_patience:
                    logger.info(
                        "Early stopping triggered after %d epochs "
                        "without improvement.",
                        self.cfg.early_stop_patience,
                    )
                    break

        # ── Finalise SWA model ────────────────────────────────────
        if self.use_swa and hasattr(self, 'swa_model') and self.swa_model is not None:
            swa_epoch = getattr(self, '_swa_updates', 0)
            # Only finalise if SWA actually ran (model was updated)
            if any(epoch >= self.swa_start for epoch in range(1, len(self.history['train_loss']) + 1)):
                logger.info("Updating SWA batch-norm statistics …")
                update_bn(train_loader, self.swa_model, device=self.device)

                # Evaluate SWA model
                orig_model = self.model
                self.model = self.swa_model.module  # unwrap
                swa_loss, swa_metrics = self._validate(val_loader)
                logger.info(
                    "SWA model  │  val_loss=%.4f  │ %s",
                    swa_loss, _fmt_metrics(swa_metrics),
                )

                if swa_loss < self.best_val_loss:
                    self.best_val_loss = swa_loss
                    self.best_threshold = swa_metrics.get("threshold", 0.5)
                    self._save_checkpoint()
                    logger.info("  ↳ SWA model is BETTER – saved!")
                else:
                    logger.info("  ↳ SWA model not better (%.4f vs %.4f) – keeping original.",
                                swa_loss, self.best_val_loss)
                self.model = orig_model  # restore

        return self.history

    # ── evaluate on a hold-out set ───────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Load best checkpoint, run evaluation, return metrics."""
        self._load_checkpoint()
        _, metrics = self._validate(loader)
        logger.info("Test metrics  │  %s", _fmt_metrics(metrics))
        return metrics

    # ── persistence ──────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_threshold": self.best_threshold,
        }, self.cfg.checkpoint_path)

    def _load_checkpoint(self) -> None:
        state = torch.load(
            self.cfg.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            self.best_threshold = state.get("best_threshold", 0.5)
        else:
            self.model.load_state_dict(state)
        self.model.to(self.device)
