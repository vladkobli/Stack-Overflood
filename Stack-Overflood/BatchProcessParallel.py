import os
import json
import time
import math
import random
import traceback
import requests
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import os
from datetime import timedelta, timezone
from dotenv import load_dotenv
load_dotenv()

# =========================
# Project imports
# =========================
from sentinelhub import SHConfig, bbox_to_dimensions

# Functions.py
from Functions import (
    AOIRequest,
    square_bbox_utm,
    list_s1_acquisitions,
    list_s2_acquisitions,
    dt_to_day_interval,
    fetch_s1_vv_vh_db,
    fetch_s2_all_bands,
)

# DataProcessing.py
from DataProcessing import process_dataset_s1, process_dataset_s2

# StoringData.py
from StoringData import store_to_hdf5, store_to_hdf5_s2

# New nearest water + DEM files
from NearestWaterBody import nearest_water_distance
from DEMToH5 import compute_dem_risk_features

# Display_water_shifts.py
from DisplayWaterShifts import run_and_save


# =========================
# Config / constants
# =========================
results_path = "Stack-Overflood/Results"
entries_csv_path = "Stack-Overflood/used_data/entries.csv"
tile_index_path = "Stack-Overflood/used_data/euhydro_tile_index_25km.csv"

# Time offset around center date
OFFSET_YEARS = 0
OFFSET_MONTHS = 1
OFFSET_DAYS = 15

# AOI / resolutions
S1_SIDE_M = 5000
S1_RES_M = 10

S2_SIDE_M = 5000
S2_RES_M = 10

DEM_RES_M = 30
DEM_SITE_WINDOW_M = 1000
DEM_RIVER_WINDOW_M = 150

# Weather
VISUALCROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
if not VISUALCROSSING_API_KEY:
    raise RuntimeError("Missing VISUAL_CROSSING_API_KEY in .env")

# Retries
MAX_RETRIES = 4
BASE_BACKOFF_SEC = 2.0
JITTER_SEC = 0.75

# Threading
MAX_ENTRIES_IN_PARALLEL = 2   # process 2 locations at once
MAX_ENTRY_THREADS = 4         # context, S1, S2, weather inside one entry

MAX_S1_WORKERS = 4            # S1 acquisitions per entry in parallel
MAX_S2_WORKERS = 2            # S2 acquisitions per entry in parallel

RETRY_ATTEMPTS = 4
RETRY_BASE_SLEEP = 2.0


# =========================
# Utility: JSON-safe conversion
# =========================
def to_jsonable(obj: Any) -> Any:
    """Convert numpy/pandas types recursively to JSON-serializable python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # careful if huge; avoid storing arrays in geojson
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj) if not isinstance(obj, (str, bytes, dict, list, tuple)) else False:
        return None
    return obj


# =========================
# Utility: Retry wrapper
# =========================
def retry_call(fn, *args, retries=MAX_RETRIES, base_delay=BASE_BACKOFF_SEC, jitter=JITTER_SEC, retry_on=(Exception,), **kwargs):
    """
    Retry a function with exponential backoff + jitter.
    Good for Sentinel Hub / HTTP transient overloads.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except retry_on as e:
            last_exc = e
            is_last = (attempt == retries)
            msg = str(e).lower()

            # Retry mostly transient/network/server issues
            transient_keywords = [
                "429", "too many requests", "timeout", "timed out",
                "502", "503", "504", "bad gateway", "service unavailable",
                "connection reset", "temporarily unavailable", "server overload"
            ]
            transient = any(k in msg for k in transient_keywords)

            # If exception doesn't look transient, still retry a couple of times
            if is_last:
                raise

            sleep_s = base_delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            print(f"⚠️ Retry {attempt}/{retries} for {getattr(fn, '__name__', fn)} after error: {e}")
            print(f"   sleeping {sleep_s:.2f}s...")
            time.sleep(sleep_s)

    raise last_exc


# =========================
# GeoJSON writer (feature summary)
# =========================
def write_entry_geojson(path: str, lon: float, lat: float, props: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": to_jsonable(props)
            }
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2, ensure_ascii=False)


# =========================
# Weather and precipitation time series
# =========================
def fetch_precip_timeseries_visualcrossing(lat: float, lon: float, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    if not VISUALCROSSING_API_KEY:
        print("⚠️ VISUALCROSSING_API_KEY missing. Skipping weather fetch.")
        return []

    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{lat},{lon}/{start_date}/{end_date}"
        f"?unitGroup=metric&include=days&key={VISUALCROSSING_API_KEY}&contentType=json"
    )

    def _do():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.json()

    data = retry_call(_do, retries=MAX_RETRIES)

    out = []
    for d in data.get("days", []):
        out.append({
            "date": d.get("datetime"),
            "precip": d.get("precip"),
            "precipprob": d.get("precipprob"),
            "precipcover": d.get("precipcover"),
            "temp": d.get("temp"),
            "tempmin": d.get("tempmin"),
            "tempmax": d.get("tempmax"),
            "humidity": d.get("humidity"),
        })

    out.sort(key=lambda x: x["date"] if x.get("date") else "")
    return out

# =========================
# Retries function
# =========================
def with_retries(fn, *args, attempts=RETRY_ATTEMPTS, base_sleep=RETRY_BASE_SLEEP, **kwargs):
    last_err = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if i == attempts - 1:
                break
            sleep_s = base_sleep * (2 ** i) + random.uniform(0, 0.8)
            print(f"⚠️ Retry {i+1}/{attempts-1} after error: {e} | sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise last_err

# =========================
# Sentinel-1 worker (thread)
# =========================
def _process_one_s1_acquisition(s1_dt, lat, lon, side_m, resolution_m, config, results_dir):
    s1_interval = dt_to_day_interval(s1_dt)
    req_s1 = AOIRequest(lon, lat, side_m, resolution_m, s1_interval[0], s1_interval[1])

    vv_db, vh_db, s1_mask, _, _ = with_retries(fetch_s1_vv_vh_db, req_s1, config)
    s1_pack = {"vv_db": vv_db, "vh_db": vh_db, "mask": s1_mask}

    result = process_dataset_s1(s1_pack)

    s1_tag = s1_dt.strftime("S1_%Y%m%dT%H%M%SZ")
    out_name = f"data_{s1_tag}_{lat:.5f}_{lon:.5f}.h5"

    s1_dir = os.path.join(results_dir, "S1")
    os.makedirs(s1_dir, exist_ok=True)
    out_path = os.path.join(s1_dir, out_name)

    store_to_hdf5(result, out_path)
    print(f"✅ Saved {out_name}")


def process_s1_series(lat, lon, start_date, end_date, side_m, resolution_m, config, results_dir):
    req_base = AOIRequest(
        center_lon=lon,
        center_lat=lat,
        side_m=side_m,
        resolution_m=resolution_m,
        start_date=start_date,
        end_date=end_date,
    )
    bbox = square_bbox_utm(req_base.center_lon, req_base.center_lat, req_base.side_m)

    s1_dts = list_s1_acquisitions(config, bbox, start_date, end_date)
    if not s1_dts:
        print("⚠️ No Sentinel-1 acquisitions found in interval.")
        return

    print(f"📡 Found {len(s1_dts)} Sentinel-1 acquisitions")

    with ThreadPoolExecutor(max_workers=MAX_S1_WORKERS) as ex:
        futures = [
            ex.submit(_process_one_s1_acquisition, s1_dt, lat, lon, side_m, resolution_m, config, results_dir)
            for s1_dt in s1_dts
        ]

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"⚠️ S1 acquisition failed: {e}")


# =========================
# Sentinel-2 worker (thread)
# =========================
# Expanding S2 window in case of no available acquisitions
def find_one_s2_with_expanding_window(
    config,
    bbox,
    center_date_str: str,
    initial_months_before=1.5,
    initial_months_after=1.5,
    max_expand_steps=20,
    expand_days_each_step=15,
    list_cloud=15,
):
    """
    Returns:
        (best_dt, used_start_date, used_end_date, found_count)
    or
        (None, start_date, end_date, 0)
    """

    center_ts = pd.to_datetime(center_date_str)
    # convert to days
    before_days = int(round(initial_months_before * 30))
    after_days = int(round(initial_months_after * 30))

    start_ts = center_ts - pd.Timedelta(days=before_days)
    end_ts   = center_ts + pd.Timedelta(days=after_days)

    for step in range(max_expand_steps + 1):
        start_date = start_ts.date().isoformat()
        end_date = end_ts.date().isoformat()

        s2_dts = list_s2_acquisitions(
            config=config,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud=list_cloud
        )

        if s2_dts:
            # choose one acquisition (closest to center date)
            center_dt_utc = center_ts.to_pydatetime().replace(tzinfo=timezone.utc)
            best_dt = min(s2_dts, key=lambda d: abs((d - center_dt_utc).total_seconds()))
            return best_dt, start_date, end_date, len(s2_dts)

        # expand symmetrically
        start_ts -= pd.Timedelta(days=expand_days_each_step)
        end_ts   += pd.Timedelta(days=expand_days_each_step)

    # none found even after expansions
    return None, start_ts.date().isoformat(), end_ts.date().isoformat(), 0


def _process_one_s2_acquisition(s2_dt, lat, lon, side_m, resolution_m, config, results_dir, fetch_cloud=80):
    s2_interval = dt_to_day_interval(s2_dt)
    req_s2 = AOIRequest(lon, lat, side_m, resolution_m, s2_interval[0], s2_interval[1])

    s2_pack = with_retries(fetch_s2_all_bands, req_s2, config, mosaicking="TILE", max_cloud=fetch_cloud)

    m = s2_pack.get("mask")
    if m is None or float(np.mean(m)) < 0.001:
        print(f"⚠️ S2 mask empty for {s2_dt}, skipping")
        return None

    result_s2 = process_dataset_s2(s2_pack)

    s2_tag = s2_dt.strftime("S2_%Y%m%dT%H%M%SZ")
    s2_name = f"data_{s2_tag}_{lat:.5f}_{lon:.5f}.h5"

    s2_dir = os.path.join(results_dir, "S2")
    os.makedirs(s2_dir, exist_ok=True)
    s2_path = os.path.join(s2_dir, s2_name)

    store_to_hdf5_s2(result_s2, s2_path)
    print(f"✅ Saved {s2_name}")

    return {
        "date": s2_dt.strftime("%Y-%m-%d"),
        **result_s2.get("s2_scalar_stats", {})
    }

# Process S2 time series (all acquisitions in interval)
def process_s2_series(lat, lon, start_date, end_date, side_m, resolution_m, config, results_dir,
                      list_cloud=15, fetch_cloud=80):
    req_base = AOIRequest(
        center_lon=lon,
        center_lat=lat,
        side_m=side_m,
        resolution_m=resolution_m,
        start_date=start_date,
        end_date=end_date,
    )
    bbox = square_bbox_utm(req_base.center_lon, req_base.center_lat, req_base.side_m)

    s2_dts = list_s2_acquisitions(config, bbox, start_date, end_date, max_cloud=list_cloud)
    if not s2_dts:
        print("⚠️ No Sentinel-2 acquisitions found in interval.")
        return []

    print(f"🛰️ Found {len(s2_dts)} Sentinel-2 acquisitions")

    s2_timeseries = []
    with ThreadPoolExecutor(max_workers=MAX_S2_WORKERS) as ex:
        futures = [
            ex.submit(_process_one_s2_acquisition, s2_dt, lat, lon, side_m, resolution_m, config, results_dir, fetch_cloud)
            for s2_dt in s2_dts
        ]

        for fut in as_completed(futures):
            try:
                item = fut.result()
                if item is not None:
                    s2_timeseries.append(item)
            except Exception as e:
                print(f"⚠️ S2 acquisition failed: {e}")

    s2_timeseries.sort(key=lambda x: x["date"])
    return s2_timeseries

# Process a single S2 acquisition
def process_s2_single(
    lat, lon,
    center_date,
    side_m, resolution_m,
    config, results_dir,
    list_cloud=15,
    fetch_cloud=80
):
    req_base = AOIRequest(
        center_lon=lon,
        center_lat=lat,
        side_m=side_m,
        resolution_m=resolution_m,
        start_date=center_date,  # temporary placeholders
        end_date=center_date,
    )
    bbox = square_bbox_utm(req_base.center_lon, req_base.center_lat, req_base.side_m)

    # Find exactly one S2 datetime (with expanding search)
    best_dt, used_start, used_end, found_count = find_one_s2_with_expanding_window(
        config=config,
        bbox=bbox,
        center_date_str=center_date,
        initial_months_before=1.5,
        initial_months_after=1.5,
        max_expand_steps=8,
        expand_days_each_step=15,
        list_cloud=list_cloud,
    )

    info = {
        "count_found_in_final_window": int(found_count),
        "count_saved": 0,
        "files": [],
        "timeseries_s2": [],
        "selection_window": {"start_date": used_start, "end_date": used_end},
    }

    if best_dt is None:
        print("⚠️ No Sentinel-2 acquisitions found even after expanding window.")
        return info

    # Fetch only that day
    s2_interval = dt_to_day_interval(best_dt)
    req_s2 = AOIRequest(lon, lat, side_m, resolution_m, s2_interval[0], s2_interval[1])

    try:
        s2_pack = with_retries(fetch_s2_all_bands, req_s2, config, mosaicking="TILE", max_cloud=fetch_cloud)
    except Exception as e:
        print(f"⚠️ S2 fetch failed for selected date {best_dt}: {e}")
        info["error"] = str(e)
        return info

    m = s2_pack.get("mask")
    if m is None or float(np.mean(m)) < 0.001:
        print(f"⚠️ Selected S2 mask empty for {best_dt}, skipping")
        info["error"] = "Selected S2 mask empty"
        return info

    result_s2 = process_dataset_s2(s2_pack)

    s2_tag = best_dt.strftime("S2_%Y%m%dT%H%M%SZ")
    s2_name = f"data_{s2_tag}_{lat:.5f}_{lon:.5f}.h5"

    s2_dir = os.path.join(results_dir, "S2")
    os.makedirs(s2_dir, exist_ok=True)
    s2_path = os.path.join(s2_dir, s2_name)

    store_to_hdf5_s2(result_s2, s2_path)
    print(f"✅ Saved {s2_name}")

    info["count_saved"] = 1
    info["files"] = [s2_name]
    info["selected_datetime"] = best_dt.isoformat()
    info["timeseries_s2"] = [{
        "date": best_dt.strftime("%Y-%m-%d"),
        **result_s2.get("s2_scalar_stats", {})
    }]

    return info

# =========================
# Context worker (thread): nearest water + DEM
# =========================
def process_context_features(
    lat: float,
    lon: float,
    config: SHConfig,
    tile_index_csv: str,
) -> Dict[str, Any]:
    """
    Computes nearest water once + DEM once.
    Returns dict with nearest_water and dem.
    """
    # nearest water
    dist_m, river_lat, river_lon, meta = retry_call(
        nearest_water_distance,
        lon=lon,
        lat=lat,
        tile_index_csv=tile_index_csv,
        tile_m=25000,
        search_radius_tiles=1,
    )

    nearest_water = {
        "dist_m": float(dist_m) if dist_m is not None and np.isfinite(dist_m) else None,
        "nearest_lat": float(river_lat) if river_lat is not None else None,
        "nearest_lon": float(river_lon) if river_lon is not None else None,
        "source": meta,
    }

    # DEM features
    dem_feats = None
    if (river_lat is not None) and (river_lon is not None):
        dem_feats = retry_call(
            compute_dem_risk_features,
            query_lon=lon, query_lat=lat,
            river_lon=river_lon, river_lat=river_lat,
            site_window_m=DEM_SITE_WINDOW_M,
            river_window_m=DEM_RIVER_WINDOW_M,
            resolution_m=DEM_RES_M,
            config=config,
        )
    else:
        print("⚠️ No nearest river coords; DEM risk features skipped.")

    return {
        "nearest_water": nearest_water,
        "dem": dem_feats
    }


# =========================
# One-entry orchestration
# =========================
import os
import traceback
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sentinelhub import SHConfig

def process_one_entry(lat: float, lon: float, center_date: str, config: SHConfig) -> None:
    center = pd.to_datetime(center_date).date()
    offset = pd.DateOffset(years=OFFSET_YEARS, months=OFFSET_MONTHS, days=OFFSET_DAYS)

    start_date = (pd.to_datetime(center) - offset).date().isoformat()
    end_date = (pd.to_datetime(center) + offset).date().isoformat()

    entry_id = f"{center.isoformat()}_{lat:.5f}_{lon:.5f}"
    results_dir = os.path.join(results_path, entry_id)
    plots_dir = os.path.join(results_dir, "_plots")
    geojson_path = os.path.join(results_dir, "summary.geojson")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    props: Dict[str, Any] = {
        "query": {"lat": float(lat), "lon": float(lon)},
        "center_date": center.isoformat(),
        "interval": {"start_date": start_date, "end_date": end_date},
    }

    # Save early placeholder
    write_entry_geojson(geojson_path, lon, lat, props)

    # Run parallel tasks
    with ThreadPoolExecutor(max_workers=MAX_ENTRY_THREADS) as ex:
        fut_context = ex.submit(process_context_features, lat, lon, config, tile_index_path)

        # IMPORTANT: use keyword args if your function signature changed
        fut_s1 = ex.submit(
            process_s1_series,
            lat=lat, lon=lon,
            start_date=start_date, end_date=end_date,
            side_m=S1_SIDE_M, resolution_m=S1_RES_M,
            config=config,
            results_dir=results_dir
        )

        # For series extraction
        # fut_s2 = ex.submit(
        #     process_s2_series,
        #     lat=lat, lon=lon,
        #     start_date=start_date, end_date=end_date,
        #     side_m=S2_SIDE_M, resolution_m=S2_RES_M,
        #     config=config,
        #     results_dir=results_dir,
        #     list_cloud=15,
        #     fetch_cloud=80
        # )

        # For single extraction
        fut_s2 = ex.submit(
            process_s2_single,
            lat=lat, lon=lon,
            center_date=center.isoformat(),
            side_m=S2_SIDE_M, resolution_m=S2_RES_M,
            config=config,
            results_dir=results_dir,
            list_cloud=15,
            fetch_cloud=80
        )

        fut_weather = ex.submit(
            fetch_precip_timeseries_visualcrossing,
            float(lat), float(lon), start_date, end_date
        )

        # -------- Context --------
        try:
            ctx = fut_context.result()
            props["nearest_water"] = ctx.get("nearest_water") if isinstance(ctx, dict) else None
            props["dem"] = ctx.get("dem") if isinstance(ctx, dict) else None
        except Exception as e:
            print("❌ Context thread failed:", e)
            traceback.print_exc()
            props["nearest_water"] = None
            props["dem"] = None

        # -------- S1 --------
        try:
            s1_info = fut_s1.result()

            # If your new process_s1_series returns a summary dict
            if isinstance(s1_info, dict):
                props["s1_summary"] = {
                    "count_found": s1_info.get("count_found", 0),
                    "count_saved": s1_info.get("count_saved", 0),
                }
            else:
                # If it returns None (worker-only version)
                s1_dir = os.path.join(results_dir, "S1")
                files = []
                if os.path.isdir(s1_dir):
                    files = sorted([f for f in os.listdir(s1_dir) if f.lower().endswith(".h5")])

                props["s1_summary"] = {
                    "count_found": None,       # unknown unless function returns it
                    "count_saved": len(files),
                }

        except Exception as e:
            print("❌ S1 thread failed:", e)
            traceback.print_exc()
            props["s1_summary"] = {"error": str(e)}

        # -------- S2 --------
        try:
            s2_info = fut_s2.result()

            # Case 1: returns summary dict
            if isinstance(s2_info, dict):
                props["s2_summary"] = {
                    "count_found": s2_info.get("count_found", 0),
                    "count_saved": s2_info.get("count_saved", 0),
                }
                props["timeseries_s2"] = s2_info.get("timeseries_s2", [])

            # Case 2: returns just timeseries list (new split style)
            elif isinstance(s2_info, list):
                props["timeseries_s2"] = s2_info

                s2_dir = os.path.join(results_dir, "S2")
                s2_files = []
                if os.path.isdir(s2_dir):
                    s2_files = sorted([f for f in os.listdir(s2_dir) if f.lower().endswith(".h5")])

                props["s2_summary"] = {
                    "count_found": None,   # unless returned by process_s2_series
                    "count_saved": len(s2_files),
                }

            else:
                props["s2_summary"] = {"warning": "Unexpected S2 return type"}
                props["timeseries_s2"] = []

        except Exception as e:
            print("❌ S2 thread failed:", e)
            traceback.print_exc()
            props["s2_summary"] = {"error": str(e)}
            props["timeseries_s2"] = []

        # -------- Weather --------
        try:
            props["timeseries_precip"] = fut_weather.result()
        except Exception as e:
            print("❌ Weather thread failed:", e)
            traceback.print_exc()
            props["timeseries_precip"] = []

    # Save partial summary after threads complete
    write_entry_geojson(geojson_path, lon, lat, props)

    # Run water-shift plots after S1 is finished (depends on S1 .h5 files)
    s1_dir = os.path.join(results_dir, "S1")
    try:
        if os.path.isdir(s1_dir):
            out = run_and_save(results_dir=s1_dir, out_dir=plots_dir)
            props["water_shift_outputs"] = out
        else:
            props["water_shift_outputs"] = {"error": "S1 directory missing"}
    except Exception as e:
        print(f"⚠️ Plot generation failed: {e}")
        traceback.print_exc()
        props["water_shift_outputs"] = {"error": str(e)}

    # Final write
    write_entry_geojson(geojson_path, lon, lat, props)
    print(f"📄 Summary saved: {geojson_path}")


# =========================
# SH Config
# =========================
def build_sh_config() -> SHConfig:
    cfg = SHConfig()
    cfg.sh_client_id = os.getenv("SH_CLIENT_ID")
    cfg.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise RuntimeError("Missing Sentinel Hub credentials. Set SH_CLIENT_ID and SH_CLIENT_SECRET.")
    return cfg


# =========================
# Main
# =========================
def main():
    os.makedirs(results_path, exist_ok=True)

    config = build_sh_config()
    
    if not os.path.exists(entries_csv_path):
        raise FileNotFoundError(f"entries.csv not found: {entries_csv_path}")    

    entries_df = pd.read_csv(entries_csv_path)
    required = {"center_lat", "center_lon", "center_date"}
    missing = required - set(entries_df.columns)
    if missing:
        raise ValueError(f"entries.csv missing columns: {missing}. Expected at least {required}")
    
    with ThreadPoolExecutor(max_workers=MAX_ENTRIES_IN_PARALLEL) as ex:
        futures = []
        for _, entry in entries_df.iterrows():
            lat = float(entry["center_lat"])
            lon = float(entry["center_lon"])
            center_date = str(entry["center_date"])
            futures.append(ex.submit(process_one_entry, lat, lon, center_date, config))

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"❌ Entry failed: {e}")


if __name__ == "__main__":
    main()