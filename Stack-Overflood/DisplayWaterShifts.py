import os
import re
from glob import glob
from datetime import datetime
import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for threads / scripts)

import matplotlib.pyplot as plt

# --- thresholds
VV_THR = -16.0   # dB (water often < -16 .. -20)
VH_THR = -20.0   # dB (water often < -20 .. -25)

# dB should be roughly between ~-30 and maybe -5.

FNAME_RE = re.compile(r"data_S1_(\d{8}T\d{6}Z)_(\-?\d+\.\d+)_(\-?\d+\.\d+)\.h5$")


def parse_time_from_filename(path: str) -> datetime:
    name = os.path.basename(path)
    m = FNAME_RE.match(name)
    if not m:
        raise ValueError(f"Filename doesn't match expected pattern: {name}")
    dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ")
    return dt


def read_dataset(h5_path: str, key: str):
    # Your files store under group "Satellite Data"
    with h5py.File(h5_path, "r") as f:
        g = f["Satellite Data"]
        if key not in g:
            raise KeyError(f"{key} not found in {h5_path}. Available: {list(g.keys())}")
        return g[key][()]

def remove_small_components_2(mask_bool: np.ndarray, min_area: int = 2) -> np.ndarray:
    """
    Remove connected components smaller than min_area pixels.
    min_area=2 removes single-pixel specks.
    """
    m = mask_bool.astype(np.uint8)  # 0/1
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = np.zeros_like(m)
    for i in range(1, num):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1

    return out.astype(bool)


def make_water_mask(water: np.ndarray) -> np.ndarray:
    """
    Simple SAR water mask: low VV and low VH.
    Returns boolean mask (True=water).
    """    
    water = remove_small_components_2(water, min_area=4)
    return water


def remove_small_components(mask_u8, min_area=200):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out


def run_and_save(results_dir: str, out_dir: str, vv_thr=VV_THR, vh_thr=VH_THR):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob(os.path.join(results_dir, "data_S1_*.h5")))
    if not files:
        raise SystemExit(f"No files found in {results_dir}")

    # read all into lists
    records = []
    vv_stack = []
    vh_stack = []
    water_stack = []

    shape0 = None

    for fp in files:
        dt = parse_time_from_filename(fp)
        vv = read_dataset(fp, "vv_db").astype(np.float32)
        vh = read_dataset(fp, "vh_db").astype(np.float32)
        water = read_dataset(fp, "water_mask").astype(np.float32)

        # sanity check ranges (to verify dB)
        # print(dt, "vv min/max", np.nanmin(vv), np.nanmax(vv))

        if shape0 is None:
            shape0 = vv.shape
        if vv.shape != shape0:
            raise ValueError(f"Shape mismatch: {fp} has {vv.shape}, expected {shape0}")

        water = make_water_mask(water)

        # scalar stats
        valid = np.isfinite(vv) & np.isfinite(vh)
        water_pct = 100.0 * (np.sum(water) / np.sum(valid)) if np.sum(valid) else np.nan

        records.append({
            "datetime": dt,
            "file": os.path.basename(fp),
            "water_pct": water_pct,
            "vv_mean": float(np.nanmean(vv)),
            "vh_mean": float(np.nanmean(vh)),
            "vv_mean_water": float(np.nanmean(vv[water])) if np.any(water) else np.nan,
            "vh_mean_water": float(np.nanmean(vh[water])) if np.any(water) else np.nan,
        })

        vv_stack.append(vv)
        vh_stack.append(vh)
        water_stack.append(water.astype(np.uint8))

    df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)

    # Example saving:
    extent_png = os.path.join(out_dir, "water_extent_timeseries.png")
    persistence_png = os.path.join(out_dir, "water_persistence.png")
    flooded_png = os.path.join(out_dir, "flooded_area_map.png")
    # change_png = os.path.join(out_dir, "change_map.png")

    # For each plot:
    plt.figure()
    plt.plot(df["datetime"], df["water_pct"], marker="o")
    plt.ylabel("Water area (%)")
    plt.title(f"Water extent over time (VV<{vv_thr} dB & VH<{vh_thr} dB)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(extent_png, dpi=200)
    plt.close()

    # --- 2) Persistence map: fraction of dates classified as water
    water_cube = np.stack(water_stack, axis=0)  # T x H x W
    persistence = np.mean(water_cube, axis=0)   # H x W in [0..1]

    plt.figure()
    plt.imshow(persistence, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Water persistence (fraction of dates)")
    plt.title("Water persistence map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(persistence_png, dpi=200)
    plt.close()

    persistence_filtered = np.where((persistence > 0.08) & (persistence < 0.6), persistence, np.nan)
    persistence_filtered = remove_small_components((persistence_filtered > 0).astype(np.uint8), min_area=100)
    plt.figure()
    plt.imshow(persistence_filtered, cmap="gray", vmin=0, vmax=1)
    plt.colorbar(label="Water persistence (fraction of dates)")
    plt.title("Flooded Area Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(flooded_png, dpi=200)
    plt.close()

    # # --- 3) Change map: first vs last
    # first = water_cube[0].astype(bool)
    # last = water_cube[-1].astype(bool)
    # gained = (~first) & last      # land -> water

    # change = np.zeros(shape0, dtype=np.int8)
    # change[gained] = 1

    # plt.figure()
    # plt.imshow(change, cmap="gray", vmin=0, vmax=1)
    # plt.colorbar(label="-1 water lost, +1 water gained")
    # plt.title(f"Change map: {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(change_png, dpi=200)
    # plt.close()


    return {
        "thresholds": {"vv_thr": float(vv_thr), "vh_thr": float(vh_thr)},
        "plots": {
            "extent_timeseries": extent_png,
            "persistence": persistence_png,
            "flooded_area": flooded_png,
            # "change_map": change_png,
        },
        "timeseries_s1": [
            {"datetime": d.isoformat(), "water_pct": float(w) if np.isfinite(w) else None}
            for d, w in zip(df["datetime"], df["water_pct"])
        ]
    }
