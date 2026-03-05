# MultiModal Late-Fusion Flood Prediction Model

A **PyTorch** model that fuses **temporal** (precipitation, water-percentage, …)
and **static** (DEM / slope / distance-to-water, …) features with a
**late-fusion** architecture to predict whether a location is flood-prone.

```
┌────────────────────────────────────────────────────────────────┐
│                       FloodFusionNet                           │
│                                                                │
│   Stream A (Temporal)          Stream B (Static)               │
│   ┌──────────────────┐        ┌──────────────────┐            │
│   │ Conv1D → BN →    │        │ Linear → BN →    │            │
│   │ ReLU → Dropout   │        │ ReLU → Dropout   │            │
│   │      ↓           │        │      ↓           │            │
│   │ GRU → last h     │        │ Linear → BN →    │            │
│   │      ↓           │        │ ReLU → Dropout   │            │
│   │ Temporal embed.  │        │ Static embed.    │            │
│   └────────┬─────────┘        └────────┬─────────┘            │
│            └──────────┬────────────────┘                       │
│                       ↓                                        │
│              Concatenation                                     │
│                       ↓                                        │
│            Fusion MLP → Sigmoid → P(flood)                     │
└────────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
experiment/
├── flood_model/
│   ├── __init__.py              # package marker
│   ├── __main__.py              # CLI entry-point
│   ├── config.py                # Config dataclass (all hyperparams)
│   ├── data.py                  # GeoJSON loader, Dataset, DataLoader
│   ├── model.py                 # TemporalStream, StaticStream, FusionHead, FloodFusionNet
│   ├── train.py                 # Trainer with early stopping & metrics
│   ├── inference.py             # Predict on new GeoJSON files
│   └── generate_sample_data.py  # Synthetic data generator for testing
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python -m flood_model --mode train --data data/flood_dataset.geojson --epochs 50
```

Key training flags:

| Flag            | Default | Description                                |
| --------------- | ------- | ------------------------------------------ |
| `--epochs`      | 100     | Max training epochs                        |
| `--batch-size`  | 64      | Mini-batch size                            |
| `--lr`          | 1e-3    | Learning rate (Adam)                       |
| `--pos-weight`  | —       | Positive-class weight (set >1 if rare)     |
| `--seq-len`     | 60      | Number of daily time-steps to use          |
| `--output-dir`  | outputs | Where to save checkpoints and metrics      |

Training produces:

- `outputs/best_model.pt` — best checkpoint (by validation loss)
- `outputs/normalizer.npz` — feature normalisation statistics
- `outputs/history.json` — per-epoch loss & metrics

### 3. Evaluate

```bash
python -m flood_model --mode evaluate --data data/flood_dataset.geojson
```

### 4. Predict on new data

```bash
python -m flood_model --mode predict \
    --data data/new_sites.geojson \
    --checkpoint outputs/best_model.pt
```

Output → `outputs/predictions.json`

---

## Customisation

All hyperparameters live in `flood_model/config.py` in the `Config` dataclass.
You can add or remove static features by editing `static_features` (dot-separated
paths into the GeoJSON properties), or change `temporal_features` to select
which time-series columns Stream A uses.

---

