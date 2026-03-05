"""
Data loading and preprocessing.

Reads per-sample GeoJSON files from ``flood datasets/`` and
``non-flood datasets/`` folders, extracts temporal (precip + S1)
and static (DEM + S2 + nearest-water) features, normalises them,
and wraps everything in a PyTorch Dataset / DataLoader pair.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _resolve(d: dict, dotpath: str) -> float:
    """Traverse a nested dict with a dot-separated key path.

    Example:
        _resolve(props, "dem.slope_at_center")
        → props["dem"]["slope_at_center"]
    """
    keys = dotpath.split(".")
    current: Any = d
    for k in keys:
        try:
            current = current[k]
        except (KeyError, TypeError):
            return float("nan")
    try:
        return float(current)
    except (ValueError, TypeError):
        return float("nan")


def _extract_temporal(
    props: dict,
    precip_features: List[str],
    s1_features: List[str],
    seq_len: int,
) -> np.ndarray:
    """Merge the two time-series (precip & S1) by date and return
    shape ``(seq_len, n_precip + n_s1)``.

    The precip time-series lives at ``properties.timeseries_precip``
    and the S1 time-series at
    ``properties.water_shift_outputs.timeseries_s1``.  Both are
    sorted by date, aligned, and the *last* ``seq_len`` steps are
    kept (left-padded with 0 if shorter).
    """

    # ── gather precip ts ────────────────────────────────────────────
    precip_ts: list = props.get("timeseries_precip", [])
    precip_by_date: Dict[str, dict] = {
        r.get("date", ""): r for r in precip_ts
    }

    # ── gather S1 ts (nested inside water_shift_outputs) ────────────
    ws = props.get("water_shift_outputs", {})
    s1_ts: list = ws.get("timeseries_s1", [])
    s1_by_date: Dict[str, dict] = {
        r.get("date", ""): r for r in s1_ts
    }

    # ── align on the union of dates ─────────────────────────────────
    all_dates = sorted(set(precip_by_date) | set(s1_by_date))
    all_dates = all_dates[-seq_len:]           # keep last seq_len days

    n_feats = len(precip_features) + len(s1_features)
    arr = np.full((seq_len, n_feats), np.nan, dtype=np.float32)

    offset = seq_len - len(all_dates)          # left-pad if shorter
    for t_idx, date in enumerate(all_dates):
        row_idx = offset + t_idx
        col = 0

        # precip features
        p_rec = precip_by_date.get(date, {})
        for fname in precip_features:
            try:
                arr[row_idx, col] = float(p_rec.get(fname, np.nan))
            except (ValueError, TypeError):
                pass
            col += 1

        # S1 features
        s1_rec = s1_by_date.get(date, {})
        for fname in s1_features:
            try:
                arr[row_idx, col] = float(s1_rec.get(fname, np.nan))
            except (ValueError, TypeError):
                pass
            col += 1

    return arr


def _extract_static(
    props: dict,
    feature_paths: List[str],
) -> np.ndarray:
    """Return shape (n_features,) of scalar/static values."""
    return np.array(
        [_resolve(props, fp) for fp in feature_paths],
        dtype=np.float32,
    )


# ────────────────────────────────────────────────────────────────────
# Normalisation statistics
# ────────────────────────────────────────────────────────────────────

class FeatureNormalizer:
    """Simple z-score normaliser that stores per-feature mean & std."""

    def __init__(self) -> None:
        self.temporal_mean: Optional[np.ndarray] = None
        self.temporal_std: Optional[np.ndarray] = None
        self.static_mean: Optional[np.ndarray] = None
        self.static_std: Optional[np.ndarray] = None

    # ── fit ──────────────────────────────────────────────────────────
    def fit(
        self,
        temporal_arrays: List[np.ndarray],
        static_arrays: List[np.ndarray],
    ) -> "FeatureNormalizer":
        """Compute mean/std from training data (NaN-safe)."""
        # temporal: list of (seq_len, F)  →  stack → (N, seq_len, F)
        tall = np.concatenate(temporal_arrays, axis=0)          # (N*T, F)
        self.temporal_mean = np.nanmean(tall, axis=0)           # (F,)
        self.temporal_std = np.nanstd(tall, axis=0) + 1e-8      # (F,)

        sall = np.stack(static_arrays, axis=0)                  # (N, S)
        self.static_mean = np.nanmean(sall, axis=0)             # (S,)
        self.static_std = np.nanstd(sall, axis=0) + 1e-8        # (S,)
        return self

    # ── transform ────────────────────────────────────────────────────
    def transform_temporal(self, arr: np.ndarray) -> np.ndarray:
        out = (arr - self.temporal_mean) / self.temporal_std
        return np.nan_to_num(out, nan=0.0)

    def transform_static(self, arr: np.ndarray) -> np.ndarray:
        out = (arr - self.static_mean) / self.static_std
        return np.nan_to_num(out, nan=0.0)

    # ── persistence ──────────────────────────────────────────────────
    def state_dict(self) -> Dict[str, Any]:
        return {
            "temporal_mean": self.temporal_mean,
            "temporal_std": self.temporal_std,
            "static_mean": self.static_mean,
            "static_std": self.static_std,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.temporal_mean = d["temporal_mean"]
        self.temporal_std = d["temporal_std"]
        self.static_mean = d["static_mean"]
        self.static_std = d["static_std"]


# ────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────

class FloodDataset(Dataset):
    """PyTorch Dataset that serves (temporal, static, label) tuples.

    When ``augment=True`` (training), applies:
    - Gaussian noise to temporal & static features
    - Random feature-column masking (temporal)
    - Random temporal shift (roll along time axis)
    """

    def __init__(
        self,
        temporal: List[np.ndarray],
        static: List[np.ndarray],
        labels: List[int],
        normalizer: Optional[FeatureNormalizer] = None,
        augment: bool = False,
        noise_std: float = 0.05,
        mask_prob: float = 0.10,
        max_shift: int = 5,
    ) -> None:
        assert len(temporal) == len(static) == len(labels)
        self.temporal = temporal
        self.static = static
        self.labels = labels
        self.normalizer = normalizer
        self.augment = augment
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.max_shift = max_shift

    def __len__(self) -> int:
        return len(self.labels)

    def _augment_temporal(self, t: np.ndarray) -> np.ndarray:
        """Apply augmentations to a (seq_len, F) temporal array."""
        # 1) Gaussian noise
        t = t + np.random.normal(0.0, self.noise_std, size=t.shape).astype(
            np.float32
        )
        # 2) Random feature masking – zero-out entire columns
        n_feats = t.shape[1]
        mask = np.random.rand(n_feats) < self.mask_prob
        t[:, mask] = 0.0
        # 3) Temporal shift – roll by up to ±max_shift steps
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift != 0:
            t = np.roll(t, shift, axis=0)
            if shift > 0:
                t[:shift] = 0.0       # zero-pad the introduced region
            else:
                t[shift:] = 0.0
        return t

    def _augment_static(self, s: np.ndarray) -> np.ndarray:
        """Light Gaussian noise on static features."""
        return s + np.random.normal(0.0, self.noise_std, size=s.shape).astype(
            np.float32
        )

    def __getitem__(self, idx: int):
        t = self.temporal[idx].copy()
        s = self.static[idx].copy()

        if self.normalizer is not None:
            t = self.normalizer.transform_temporal(t)
            s = self.normalizer.transform_static(s)

        if self.augment:
            t = self._augment_temporal(t)
            s = self._augment_static(s)

        return (
            torch.from_numpy(t).float(),       # (seq_len, n_temporal)
            torch.from_numpy(s).float(),        # (n_static,)
            torch.tensor(self.labels[idx]).float(),  # scalar
        )


# ────────────────────────────────────────────────────────────────────
# Public API: load → split → dataloaders
# ────────────────────────────────────────────────────────────────────

def _load_folder(
    folder: Path,
    label: int,
    cfg: Config,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Load every ``*.geojson`` in *folder* and return lists."""

    files = sorted(folder.glob("*.geojson"))
    logger.info("Scanning %s  →  %d files (label=%d)", folder.name, len(files), label)

    temporal_list: List[np.ndarray] = []
    static_list: List[np.ndarray] = []
    label_list: List[int] = []
    skipped = 0

    for fp in files:
        try:
            with open(fp, "r") as f:
                geojson = json.load(f)
            props = geojson["features"][0]["properties"]
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("Skipping %s: %s", fp.name, exc)
            skipped += 1
            continue

        temporal_list.append(
            _extract_temporal(
                props, cfg.precip_features, cfg.s1_features, cfg.seq_len,
            )
        )
        static_list.append(
            _extract_static(props, cfg.static_features)
        )
        label_list.append(label)

    if skipped:
        logger.warning("Skipped %d files in %s", skipped, folder.name)

    return temporal_list, static_list, label_list


def load_dataset(cfg: Config):
    """Load from the two class folders and merge."""
    flood_dir = Path(cfg.flood_dir)
    nonflood_dir = Path(cfg.non_flood_dir)

    t1, s1, l1 = _load_folder(flood_dir, label=1, cfg=cfg)
    t0, s0, l0 = _load_folder(nonflood_dir, label=0, cfg=cfg)

    temporal = t1 + t0
    static = s1 + s0
    labels = l1 + l0

    pos = sum(labels)
    neg = len(labels) - pos
    logger.info(
        "Total  →  floods=%d  non-floods=%d  (%.1f%% positive)",
        pos, neg, 100 * pos / max(len(labels), 1),
    )
    return temporal, static, labels


def build_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, FeatureNormalizer, Optional[DataLoader]]:
    """End-to-end: load → normalise → split → DataLoaders.

    Returns
    -------
    train_loader, val_loader, normalizer, test_loader (or None)
    """
    temporal, static, labels = load_dataset(cfg)

    n = len(labels)
    indices = np.arange(n)
    rng = np.random.default_rng(cfg.random_seed)
    rng.shuffle(indices)

    n_test = int(n * cfg.test_split)
    n_val = int(n * cfg.val_split)
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:] if n_test > 0 else np.array([], dtype=int)

    # Fit normaliser on training split only
    normalizer = FeatureNormalizer()
    normalizer.fit(
        [temporal[i] for i in train_idx],
        [static[i] for i in train_idx],
    )

    # Training set WITH augmentation
    train_ds = FloodDataset(
        [temporal[i] for i in train_idx],
        [static[i] for i in train_idx],
        [labels[i] for i in train_idx],
        normalizer,
        augment=cfg.augment,
        noise_std=cfg.aug_noise_std,
        mask_prob=cfg.aug_mask_prob,
        max_shift=cfg.aug_max_shift,
    )
    # Val / test set WITHOUT augmentation
    val_ds = FloodDataset(
        [temporal[i] for i in val_idx],
        [static[i] for i in val_idx],
        [labels[i] for i in val_idx],
        normalizer,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader: Optional[DataLoader] = None
    if len(test_idx) > 0:
        test_ds = FloodDataset(
            [temporal[i] for i in test_idx],
            [static[i] for i in test_idx],
            [labels[i] for i in test_idx],
            normalizer,
            augment=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
        )

    logger.info(
        "Splits  →  train=%d  val=%d  test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    return train_loader, val_loader, normalizer, test_loader
