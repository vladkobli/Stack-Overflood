"""
Inference module – load a trained checkpoint and predict on new
GeoJSON data (individual files or a folder of files).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import Config
from .data import (
    FeatureNormalizer,
    _extract_temporal,
    _extract_static,
)
from .model import FloodFusionNet

logger = logging.getLogger(__name__)


def load_model_and_normalizer(
    cfg: Config,
    checkpoint_path: str | Path | None = None,
    normalizer_path: str | Path | None = None,
) -> tuple[FloodFusionNet | list[FloodFusionNet], FeatureNormalizer, float]:
    """Reconstruct trained model(s) + normaliser from disk.

    If ensemble metadata exists, loads all ensemble models.
    Returns (model_or_list, normalizer, threshold).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)

    # Check for ensemble
    ensemble_meta_path = out_dir / "ensemble_meta.json"
    if cfg.use_ensemble and ensemble_meta_path.exists() and checkpoint_path is None:
        with open(ensemble_meta_path) as f:
            meta = json.load(f)

        models = []
        thresholds = []
        for ckpt_name in meta["checkpoints"]:
            ckpt_path = out_dir / ckpt_name
            model = FloodFusionNet.from_config(cfg)
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                thresholds.append(state.get("best_threshold", 0.5))
                model.load_state_dict(state["model_state_dict"])
            else:
                thresholds.append(0.5)
                model.load_state_dict(state)
            model.to(device)
            model.eval()
            models.append(model)

        avg_thr = sum(thresholds) / len(thresholds)
        logger.info("Loaded %d ensemble models (avg threshold=%.2f)", len(models), avg_thr)

        # Normalizer
        norm_path = normalizer_path or out_dir / "normalizer.npz"
        norm = FeatureNormalizer()
        npz = np.load(norm_path, allow_pickle=True)
        norm.load_state_dict({k: npz[k] for k in npz.files})

        return models, norm, avg_thr

    # Single model
    model = FloodFusionNet.from_config(cfg)
    ckpt = checkpoint_path or cfg.checkpoint_path
    state = torch.load(ckpt, map_location=device, weights_only=True)
    best_threshold = 0.5
    if isinstance(state, dict) and "model_state_dict" in state:
        best_threshold = state.get("best_threshold", 0.5)
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded model from %s  (threshold=%.2f)", ckpt, best_threshold)

    # ── normaliser ───────────────────────────────────────────────────
    norm_path = normalizer_path or Path(cfg.output_dir) / "normalizer.npz"
    norm = FeatureNormalizer()
    npz = np.load(norm_path, allow_pickle=True)
    norm.load_state_dict(
        {k: npz[k] for k in npz.files}
    )
    logger.info("Loaded normaliser from %s", norm_path)

    return model, norm, best_threshold


def predict_single(
    geojson_path: str | Path,
    cfg: Config,
    model: FloodFusionNet | list[FloodFusionNet],
    normalizer: FeatureNormalizer,
    threshold: float = 0.5,
) -> Dict:
    """Run prediction on a single GeoJSON file.

    If *model* is a list, averages probabilities across all models (ensemble).

    Returns
    -------
    dict with keys: file, coordinates, probability, prediction
    """
    models = model if isinstance(model, list) else [model]
    device = next(models[0].parameters()).device

    with open(geojson_path, "r") as f:
        data = json.load(f)

    props = data["features"][0]["properties"]
    coords = data["features"][0].get("geometry", {}).get(
        "coordinates", [None, None]
    )

    # Extract & normalise
    t_arr = _extract_temporal(
        props, cfg.precip_features, cfg.s1_features, cfg.seq_len,
    )
    s_arr = _extract_static(props, cfg.static_features)

    t_arr = normalizer.transform_temporal(t_arr)
    s_arr = normalizer.transform_static(s_arr)

    t_tensor = torch.from_numpy(t_arr).float().unsqueeze(0).to(device)
    s_tensor = torch.from_numpy(s_arr).float().unsqueeze(0).to(device)

    with torch.no_grad():
        probs = [m.predict_proba(t_tensor, s_tensor).item() for m in models]
        prob = sum(probs) / len(probs)

    return {
        "file": Path(geojson_path).name,
        "coordinates": coords,
        "probability": round(prob, 4),
        "prediction": int(prob >= threshold),
    }


def predict_folder(
    folder_path: str | Path,
    cfg: Config,
    model: FloodFusionNet | None = None,
    normalizer: FeatureNormalizer | None = None,
    checkpoint_path: str | Path | None = None,
    normalizer_path: str | Path | None = None,
    threshold: float | None = None,
) -> List[Dict]:
    """Run predictions on every ``*.geojson`` in *folder_path*."""

    if model is None or normalizer is None:
        model, normalizer, saved_thr = load_model_and_normalizer(
            cfg,
            checkpoint_path=checkpoint_path,
            normalizer_path=normalizer_path,
        )
        if threshold is None:
            threshold = saved_thr
    if threshold is None:
        threshold = 0.5

    folder = Path(folder_path)
    files = sorted(folder.glob("*.geojson"))
    logger.info("Predicting on %d files in %s", len(files), folder)

    results: List[Dict] = []
    for fp in files:
        try:
            results.append(
                predict_single(fp, cfg, model, normalizer, threshold)
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", fp.name, exc)

    logger.info("Predicted %d files from %s", len(results), folder)
    return results


def predict_to_geojson(
    input_path: str | Path,
    output_path: str | Path,
    cfg: Config,
    **kwargs,
) -> None:
    """Run predictions on a single GeoJSON and write results back."""

    model, normalizer, threshold = load_model_and_normalizer(cfg, **kwargs)

    with open(input_path, "r") as f:
        data = json.load(f)

    props = data["features"][0]["properties"]
    result = predict_single(input_path, cfg, model, normalizer)

    data["features"][0]["properties"]["flood_probability"] = result["probability"]
    data["features"][0]["properties"]["flood_prediction"] = result["prediction"]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Wrote prediction to %s", out)
