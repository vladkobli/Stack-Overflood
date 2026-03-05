"""
CLI entry-point for the MultiModal Drought Prediction pipeline.

Usage
-----
    # Train
    python -m drought_model --mode train

    # Predict
    python -m drought_model --mode predict --predict-dir some_folder/

    # Evaluate on a held-out test set
    python -m drought_model --mode evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .config import Config
from .data import build_dataloaders, FeatureNormalizer
from .model import FloodFusionNet
from .train import Trainer
from .inference import predict_folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  │  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("drought_model")


# ────────────────────────────────────────────────────────────────────
# CLI argument parser
# ────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MultiModal Late-Fusion Drought Predictor",
    )
    p.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict"],
        default="train",
        help="Pipeline mode (default: train)",
    )
    p.add_argument(
        "--flood-dir",
        type=str,
        default=None,
        help="Folder with drought GeoJSON samples (default: 'Drought_locations')",
    )
    p.add_argument(
        "--non-flood-dir",
        type=str,
        default=None,
        help="Folder with non-drought GeoJSON samples (default: 'Non-Floods')",
    )
    p.add_argument(
        "--predict-dir",
        type=str,
        default=None,
        help="Folder of GeoJSON files to predict on (predict mode only)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt) for evaluate / predict modes",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="drought_outputs",
        help="Directory for checkpoints, normaliser, predictions",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    p.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    p.add_argument(
        "--pos-weight", type=float, default=None,
        help="Positive-class weight for BCE loss (use if data is imbalanced)",
    )
    p.add_argument(
        "--seq-len", type=int, default=None,
        help="Override time-series length (default 92)",
    )
    return p.parse_args(argv)


# ────────────────────────────────────────────────────────────────────
# Mode handlers
# ────────────────────────────────────────────────────────────────────

def run_train(cfg: Config) -> None:
    """Full training pipeline - supports single or ensemble training."""
    out = Path(cfg.output_dir)

    if cfg.use_ensemble:
        logger.info("═══ Ensemble training with %d seeds ═══", len(cfg.ensemble_seeds))
        all_histories = []
        for i, seed in enumerate(cfg.ensemble_seeds):
            logger.info("─── Fold %d/%d  (seed=%d) ───", i + 1, len(cfg.ensemble_seeds), seed)
            fold_cfg = Config(**{
                f.name: getattr(cfg, f.name)
                for f in cfg.__dataclass_fields__.values()
            })
            fold_cfg.random_seed = seed
            fold_cfg.checkpoint_name = f"best_model_seed{seed}.pt"

            train_loader, val_loader, normalizer, test_loader = build_dataloaders(fold_cfg)
            model = FloodFusionNet.from_config(fold_cfg)
            trainer = Trainer(fold_cfg, model)
            history = trainer.fit(train_loader, val_loader)
            all_histories.append(history)

        # Save normaliser (from last fold — same features, similar stats)
        np.savez(out / "normalizer.npz", **normalizer.state_dict())
        logger.info("Saved normaliser to %s", out / "normalizer.npz")

        # Save per-seed training histories
        for seed, history in zip(cfg.ensemble_seeds, all_histories):
            serialisable = {
                k: [float(v) if isinstance(v, (float, np.floating)) else v for v in vs]
                for k, vs in history.items()
            }
            if "val_metrics" in serialisable:
                serialisable["val_metrics"] = [
                    {mk: float(mv) for mk, mv in m.items()}
                    for m in serialisable["val_metrics"]
                ]
            with open(out / f"history_seed{seed}.json", "w") as f:
                json.dump(serialisable, f, indent=2)
        logger.info("Saved per-seed training histories")

        # Save ensemble metadata
        meta = {
            "seeds": cfg.ensemble_seeds,
            "checkpoints": [f"best_model_seed{s}.pt" for s in cfg.ensemble_seeds],
        }
        with open(out / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved ensemble metadata to %s", out / "ensemble_meta.json")

        # Now evaluate the ensemble on the validation set from the FIRST seed
        logger.info("─── Evaluating full ensemble ───")
        eval_cfg = Config(**{
            f.name: getattr(cfg, f.name)
            for f in cfg.__dataclass_fields__.values()
        })
        eval_cfg.random_seed = cfg.ensemble_seeds[0]
        _, val_loader_eval, _, _ = build_dataloaders(eval_cfg)
        _evaluate_ensemble(cfg, val_loader_eval)

    else:
        train_loader, val_loader, normalizer, test_loader = build_dataloaders(cfg)
        model = FloodFusionNet.from_config(cfg)
        trainer = Trainer(cfg, model)
        history = trainer.fit(train_loader, val_loader)

        np.savez(out / "normalizer.npz", **normalizer.state_dict())
        logger.info("Saved normaliser to %s", out / "normalizer.npz")

        serialisable = {
            k: [float(v) if isinstance(v, (float, np.floating)) else v for v in vs]
            for k, vs in history.items()
        }
        if "val_metrics" in serialisable:
            serialisable["val_metrics"] = [
                {mk: float(mv) for mk, mv in m.items()}
                for m in serialisable["val_metrics"]
            ]
        with open(out / "history.json", "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Saved training history to %s", out / "history.json")

        if test_loader is not None:
            logger.info("Evaluating on test split …")
            trainer.evaluate(test_loader)


def _evaluate_ensemble(cfg: Config, val_loader) -> None:
    """Load all ensemble models, average probabilities, compute metrics."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from .train import find_optimal_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.output_dir)

    # Load all models
    models = []
    for seed in cfg.ensemble_seeds:
        ckpt_path = out / f"best_model_seed{seed}.pt"
        model = FloodFusionNet.from_config(cfg)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)

    # Collect averaged predictions
    all_probs, all_labels = [], []
    with torch.no_grad():
        for temporal, static, labels in val_loader:
            temporal = temporal.to(device)
            static = static.to(device)

            # Average probabilities from all models
            probs_sum = torch.zeros(temporal.size(0), device=device)
            for m in models:
                logits = m(temporal, static).squeeze(-1)
                probs_sum += torch.sigmoid(logits)
            avg_probs = probs_sum / len(models)

            all_probs.append(avg_probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    opt_thr = find_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= opt_thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    logger.info(
        "ENSEMBLE  │  accuracy=%.4f  f1=%.4f  precision=%.4f  "
        "recall=%.4f  auc_roc=%.4f  threshold=%.2f",
        acc, f1, prec, rec, auc, opt_thr,
    )


def run_evaluate(cfg: Config) -> None:
    """Load checkpoint & evaluate on the full dataset (or its test split)."""
    _, val_loader, _, test_loader = build_dataloaders(cfg)
    loader = test_loader if test_loader is not None else val_loader

    model = FloodFusionNet.from_config(cfg)
    trainer = Trainer(cfg, model)
    metrics = trainer.evaluate(loader)

    out_path = Path(cfg.output_dir) / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    logger.info("Evaluation results saved to %s", out_path)


def run_predict(cfg: Config, predict_dir: str | None = None) -> None:
    """Predict flood probability for every GeoJSON in a folder."""
    folder = predict_dir or cfg.flood_dir   # default to flood dir
    results = predict_folder(
        folder_path=folder,
        cfg=cfg,
        checkpoint_path=cfg.checkpoint_path,
    )

    out_path = Path(cfg.output_dir) / "predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Predictions saved to %s", out_path)

    # Quick summary
    probs = [r["probability"] for r in results]
    preds = [r["prediction"] for r in results]
    logger.info(
        "Summary  │  n=%d  drought=%d  no-drought=%d  "
        "mean_prob=%.3f  max_prob=%.3f",
        len(results),
        sum(preds),
        len(preds) - sum(preds),
        np.mean(probs),
        np.max(probs) if probs else 0,
    )


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)

    # Build config, applying CLI overrides
    cfg = Config()
    if args.flood_dir is not None:
        cfg.flood_dir = args.flood_dir
    if args.non_flood_dir is not None:
        cfg.non_flood_dir = args.non_flood_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.checkpoint is not None:
        cfg.checkpoint_name = str(
            Path(args.checkpoint).relative_to(cfg.output_dir)
            if Path(args.checkpoint).is_relative_to(cfg.output_dir)
            else Path(args.checkpoint).name
        )
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.pos_weight is not None:
        cfg.pos_weight = args.pos_weight
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "evaluate":
        run_evaluate(cfg)
    elif args.mode == "predict":
        run_predict(cfg, predict_dir=args.predict_dir)


if __name__ == "__main__":
    main()
