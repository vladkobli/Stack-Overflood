import numpy as np

# vv and vh thresholds for water map
VV_THR = -16.0
VH_THR = -20.0 


def compute_nearest_water_distance(water_map, target_point):
    water_pixels = np.column_stack(np.where(water_map == 1))
    if water_pixels.shape[0] == 0:
        return None
    distances = np.sqrt((water_pixels[:, 0] - target_point[0]) ** 2 + (water_pixels[:, 1] - target_point[1]) ** 2)
    return np.min(distances)


def safe_norm(x, valid):
    x2 = x.copy().astype(np.float32)
    x2[~valid] = np.nan
    if not np.any(np.isfinite(x2)):
        return np.full_like(x2, np.nan, dtype=np.float32)
    mn = np.nanmin(x2)
    mx = np.nanmax(x2)
    if mx - mn < 1e-6:
        return np.zeros_like(x2, dtype=np.float32)
    return (x2 - mn) / (mx - mn)


def process_dataset_s1(s1_pack):
    vv_db = s1_pack["vv_db"]
    vh_db = s1_pack["vh_db"]
    s1_mask = s1_pack["mask"]

    valid = s1_mask & np.isfinite(vv_db) & np.isfinite(vh_db)

    result = {
        # SAR
        "vv_db": vv_db.astype(np.float32),
        "vh_db": vh_db.astype(np.float32),
    }

    # Optional: normalized vv/vh (only if you still need it)
    vv = vv_db.copy(); vh = vh_db.copy()
    vv[~valid] = np.nan; vh[~valid] = np.nan

    vv_n = safe_norm(vv_db, valid)
    vh_n = safe_norm(vh_db, valid)
    result["vv_band"] = vv_n.astype(np.float32)
    result["vh_band"] = vh_n.astype(np.float32)

    valid = np.isfinite(vv) & np.isfinite(vh)
    water = (vv < VV_THR) & (vh < VH_THR) & valid

    result["water_mask"] = water.astype(np.float32)

    # Cached location features (same as you had)
    result = {
        "vv_db": vv_db.astype(np.float32),
        "vh_db": vh_db.astype(np.float32),
        "vv_band": vv_n.astype(np.float32),
        "vh_band": vh_n.astype(np.float32),
        "valid_mask": valid.astype(np.uint8),
        "water_mask": water.astype(np.float32),
    }

    return result

def _nd(a, b):
    denom = a + b
    out = np.full_like(a, np.nan, dtype=np.float32)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > 1e-12)
    out[m] = (a[m] - b[m]) / denom[m]
    return out

def _rgb_to_uint8(rgb: np.ndarray, lo=0.0, hi=0.30) -> np.ndarray:
    """
    Convert float reflectance RGB (0..1-ish) to uint8 for quick viewing.
    - Handles NaNs safely (sets them to 0)
    - Clips to a display range (lo..hi) then scales to 0..255
    """
    x = rgb.astype(np.float32)

    # Replace NaNs/inf with 0 for display
    x = np.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo)

    # Clip and scale
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    return (x * 255.0 + 0.5).astype(np.uint8)


def process_dataset_s2(s2_pack=None):
    result = {}  # always defined

    if s2_pack is None:
        return result

    s2_mask = s2_pack["mask"].astype(bool)
    valid = s2_mask
    result["valid_mask"] = valid.astype(np.uint8)

    for b in ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]:
        if b in s2_pack:
            arr = s2_pack[b].astype(np.float32)
            # arr[~valid] = np.nan
            result[b] = arr

    ndvi = _nd(result["B08"], result["B04"]) if ("B08" in result and "B04" in result) else None
    ndmi = _nd(result["B08"], result["B11"]) if ("B08" in result and "B11" in result) else None
    ndwi = _nd(result["B03"], result["B11"]) if ("B03" in result and "B11" in result) else None

    if ndvi is not None: result["ndvi"] = ndvi.astype(np.float32)
    if ndmi is not None: result["ndmi"] = ndmi.astype(np.float32)
    if ndwi is not None: result["ndwi"] = ndwi.astype(np.float32)

    if all(k in result for k in ["B04","B03","B02"]):
        rgb = np.stack([result["B04"], result["B03"], result["B02"]], axis=-1)
        result["rgb_u8"] = _rgb_to_uint8(rgb)

    # -----------------------------
    # NEW: scalar stats for GeoJSON
    # -----------------------------
    stats = {}

    def safe_mean(x, m):
        mm = m & np.isfinite(x)
        return float(np.nanmean(x[mm])) if np.any(mm) else None

    if ndvi is not None:
        ndvi_valid = valid & np.isfinite(ndvi)
        veg_mask = ndvi > 0.30  # adjust if you want 0.2
        stats["veg_pct"] = float(100.0 * np.sum(veg_mask & ndvi_valid) / np.sum(ndvi_valid)) if np.sum(ndvi_valid) else None
        stats["ndvi_mean"] = safe_mean(ndvi, valid)

    if ndmi is not None:
        stats["ndmi_mean"] = safe_mean(ndmi, valid)
        # optional: moisture only over vegetation
        if ndvi is not None:
            veg = valid & np.isfinite(ndvi) & (ndvi > 0.30)
            stats["ndmi_mean_over_veg"] = safe_mean(ndmi, veg)

    if ndwi is not None:
        stats["ndwi_mean"] = safe_mean(ndwi, valid)
        # optional: water % (rough threshold)
        water_mask = ndwi > 0.10
        ndwi_valid = valid & np.isfinite(ndwi)
        stats["water_pct_ndwi"] = float(100.0 * np.sum(water_mask & ndwi_valid) / np.sum(ndwi_valid)) if np.sum(ndwi_valid) else None

    result["s2_scalar_stats"] = stats
    
    return result
