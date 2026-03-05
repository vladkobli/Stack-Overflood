"""
Model architecture for the MultiModal Late-Fusion Flood Predictor.

┌───────────────────────────────────────────────────────────────┐
│                      FloodFusionNet                           │
│                                                               │
│  ┌──────────────────┐    ┌──────────────────┐                │
│  │   Stream A        │    │   Stream B        │               │
│  │  (Temporal)       │    │  (Static / MLP)   │               │
│  │                   │    │                   │               │
│  │  Conv1D → BN →    │    │  Linear → BN →    │               │
│  │  ReLU → Dropout   │    │  ReLU → Dropout   │               │
│  │       ↓           │    │       ↓           │               │
│  │  GRU  → last h    │    │  Linear → BN →    │               │
│  │       ↓           │    │  ReLU → Dropout   │               │
│  │  Temporal embed.  │    │  Static embed.    │               │
│  └────────┬─────────┘    └────────┬─────────┘               │
│           └────────┬──────────────┘                           │
│                    ↓                                          │
│           Concatenation                                       │
│                    ↓                                          │
│          Fusion MLP → Sigmoid  →  P(flood)                   │
└───────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────
# Stream A – Spatial-Temporal (1-D CNN + GRU)
# ────────────────────────────────────────────────────────────────────

class TemporalStream(nn.Module):
    """Process a multi-variate time-series of shape (B, T, F).

    Pipeline
    --------
    1.  Permute → (B, F, T) for Conv1d
    2.  1-D convolution with BatchNorm + ReLU
    3.  Permute back → (B, T', F') for GRU
    4.  Multi-layer GRU  →  last hidden state
    5.  Return embedding of size ``gru_hidden * dirs``
    """

    def __init__(
        self,
        n_features: int,
        cnn_filters: int = 64,
        cnn_kernel: int = 3,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        # ── 1-D CNN block ─────────────────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=cnn_filters,
                kernel_size=cnn_kernel,
                padding=cnn_kernel // 2,   # same-padding
            ),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── GRU ─────────────────────────────────────────────────────
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=cnn_filters,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        h_dim = gru_hidden * (2 if bidirectional else 1)
        self.out_dim = h_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, F)  –  batch of time-series.

        Returns
        -------
        h : (B, gru_hidden * dirs)  –  temporal embedding.
        """
        # CNN expects (B, C, T)
        x = x.permute(0, 2, 1)          # (B, F, T)
        x = self.conv(x)                 # (B, cnn_filters, T)
        x = x.permute(0, 2, 1)          # (B, T, cnn_filters)

        # GRU – last hidden state
        _, h_n = self.gru(x)             # h_n: (layers*dirs, B, H)

        if self.bidirectional:
            # concat forward & backward last-layer hidden states
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, H*2)
        else:
            h = h_n[-1]                                 # (B, H)

        return h


# ────────────────────────────────────────────────────────────────────
# Stream B – Static / Contextual (MLP)
# ────────────────────────────────────────────────────────────────────

class StaticStream(nn.Module):
    """Simple feed-forward network for scalar features.

    Input  → [Linear → BatchNorm → ReLU → Dropout] × N → embedding
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]

        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, S)  –  batch of static feature vectors.

        Returns
        -------
        h : (B, out_dim)  –  static embedding.
        """
        return self.net(x)


# ────────────────────────────────────────────────────────────────────
# Late-Fusion head
# ────────────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    """Concatenate two embeddings → MLP → sigmoid probability."""

    def __init__(
        self,
        temporal_dim: int,
        static_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        in_dim = temporal_dim + static_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),     # logit
        )

    def forward(
        self, h_temporal: torch.Tensor, h_static: torch.Tensor,
    ) -> torch.Tensor:
        """Return **logit** (not yet sigmoid) of shape (B, 1)."""
        combined = torch.cat([h_temporal, h_static], dim=1)
        return self.head(combined)


# ────────────────────────────────────────────────────────────────────
# Full model
# ────────────────────────────────────────────────────────────────────

class FloodFusionNet(nn.Module):
    """Multi-modal late-fusion network for flood prediction.

    Composes *TemporalStream*, *StaticStream*, and *FusionHead*.
    """

    def __init__(
        self,
        n_temporal_features: int,
        n_static_features: int,
        # Stream A
        cnn_filters: int = 64,
        cnn_kernel: int = 3,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        temporal_dropout: float = 0.3,
        gru_bidir: bool = False,
        # Stream B
        static_hidden_dims: list[int] | None = None,
        static_dropout: float = 0.3,
        # Fusion
        fusion_hidden_dim: int = 64,
        fusion_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.temporal_stream = TemporalStream(
            n_features=n_temporal_features,
            cnn_filters=cnn_filters,
            cnn_kernel=cnn_kernel,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=temporal_dropout,
            bidirectional=gru_bidir,
        )

        self.static_stream = StaticStream(
            n_features=n_static_features,
            hidden_dims=static_hidden_dims,
            dropout=static_dropout,
        )

        self.fusion = FusionHead(
            temporal_dim=self.temporal_stream.out_dim,
            static_dim=self.static_stream.out_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
        )

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_temporal : (B, T, F_t)
        x_static   : (B, F_s)

        Returns
        -------
        logit : (B, 1)  –  raw logit; apply ``torch.sigmoid`` for P(flood).
        """
        h_t = self.temporal_stream(x_temporal)
        h_s = self.static_stream(x_static)
        logit = self.fusion(h_t, h_s)
        return logit

    # ── convenience ──────────────────────────────────────────────────
    def predict_proba(
        self,
        x_temporal: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Return P(flood) in [0, 1] of shape (B,)."""
        logit = self.forward(x_temporal, x_static)
        return torch.sigmoid(logit).squeeze(-1)

    @classmethod
    def from_config(cls, cfg) -> "FloodFusionNet":
        """Build a model directly from a ``Config`` object."""
        return cls(
            n_temporal_features=cfg.n_temporal_features,
            n_static_features=cfg.n_static_features,
            cnn_filters=cfg.temporal_cnn_filters,
            cnn_kernel=cfg.temporal_cnn_kernel,
            gru_hidden=cfg.temporal_gru_hidden,
            gru_layers=cfg.temporal_gru_layers,
            temporal_dropout=cfg.temporal_dropout,
            gru_bidir=cfg.temporal_gru_bidir,
            static_hidden_dims=cfg.static_hidden_dims,
            static_dropout=cfg.static_dropout,
            fusion_hidden_dim=cfg.fusion_hidden_dim,
            fusion_dropout=cfg.fusion_dropout,
        )
