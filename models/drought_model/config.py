"""
Configuration dataclass that controls every tuneable knob of the
drought-prediction training / inference pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    """Central configuration for the drought-prediction pipeline."""

    # ── data ────────────────────────────────────────────────────────────
    flood_dir: str = "Drought_locations"
    non_flood_dir: str = "Non-Floods"
    val_split: float = 0.2          # fraction kept for validation
    test_split: float = 0.0         # fraction kept for test (0 = no test set)
    random_seed: int = 42

    # ── temporal stream ────────────────────────────────────────────────
    # Columns from `timeseries_precip` entries:
    precip_features: List[str] = field(
        default_factory=lambda: [
            "precip",
            "precipprob",
            "precipcover",
            "humidity",
            "tempmin",
            "tempmax",
            "temp",
            "precip_sum_14d",
        ]
    )
    # Columns from `water_shift_outputs.timeseries_s1` entries:
    s1_features: List[str] = field(
        default_factory=lambda: [
            "water_pct",
        ]
    )
    seq_len: int = 92               # number of time-steps (daily)

    # ── static stream ──────────────────────────────────────────────────
    # Each entry is a *dot-separated* path inside the properties dict,
    # e.g. "dem.slope_at_center" → props["dem"]["slope_at_center"]
    static_features: List[str] = field(
        default_factory=lambda: [
            # DEM
            "dem.height_above_river",
            "dem.local_relief",
            "dem.slope_at_center",
            "dem.site_slope_p50",
            "dem.river_slope_p50",
            # S2 spectral indices
            "s2.veg_pct",
            "s2.ndvi_mean",
            "s2.ndmi_mean",
            "s2.ndmi_mean_over_veg",
            "s2.ndwi_mean",
            "s2.water_pct_ndwi",
            # distance to nearest water body
            "nearest_water.dist_m",
        ]
    )

    # ── model architecture ─────────────────────────────────────────────
    # Stream A – 1-D CNN + GRU
    temporal_cnn_filters: int = 128
    temporal_cnn_kernel: int = 3
    temporal_gru_hidden: int = 128
    temporal_gru_layers: int = 2
    temporal_gru_bidir: bool = True
    temporal_dropout: float = 0.3

    # Stream B – MLP
    static_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    static_dropout: float = 0.3

    # Fusion head
    fusion_hidden_dim: int = 128
    fusion_dropout: float = 0.3

    # ── training ───────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    pos_weight: Optional[float] = None   # set >1 if floods are rare
    early_stop_patience: int = 100     # effectively disabled – run all epochs
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # LR schedule
    use_cosine_annealing: bool = True
    warmup_epochs: int = 5

    # Stochastic Weight Averaging
    use_swa: bool = False
    swa_start_frac: float = 0.75       # start SWA at 75% of training
    swa_lr: float = 1e-4               # flat LR used during SWA phase

    # Loss function
    use_focal_loss: bool = False
    focal_gamma: float = 2.0           # focus on hard examples
    focal_alpha: float = 0.25          # balance factor

    # Regularisation
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0          # 0 = disabled

    # Data augmentation (applied to training set only)
    augment: bool = True
    aug_noise_std: float = 0.05
    aug_mask_prob: float = 0.10
    aug_max_shift: int = 5

    # ── paths ──────────────────────────────────────────────────────────
    output_dir: str = "drought_outputs"
    checkpoint_name: str = "best_model.pt"
    use_ensemble: bool = True
    ensemble_seeds: list[int] = field(default_factory=lambda: [42, 123, 2024])

    # ── derived (computed at runtime) ──────────────────────────────────
    @property
    def n_temporal_features(self) -> int:
        """Total number of features per time-step (precip + S1)."""
        return len(self.precip_features) + len(self.s1_features)

    @property
    def n_static_features(self) -> int:
        return len(self.static_features)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / self.checkpoint_name
