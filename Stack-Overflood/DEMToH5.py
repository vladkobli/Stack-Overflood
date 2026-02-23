# DEM.py  (Sentinel Hub version - COP30)
import os
import math
import cv2
import numpy as np
from pyproj import Transformer
import h5py

from sentinelhub import (
    SHConfig, CRS, BBox, DataCollection, MimeType,
    SentinelHubRequest, bbox_to_dimensions
)

# ----------------------------
# Geometry helpers
# ----------------------------

def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)

def square_bbox_utm(center_lon: float, center_lat: float, side_m: float) -> BBox:
    """
    Create a square bbox in UTM meters around lon/lat center.
    This ensures dx/dy are real meters (good for slope).
    """
    epsg = utm_epsg_from_lonlat(center_lon, center_lat)
    crs_utm = CRS(epsg)

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = to_utm.transform(center_lon, center_lat)
    half = side_m / 2.0

    return BBox((cx - half, cy - half, cx + half, cy + half), crs=crs_utm)


#### .h5 helpers
P_KEYS_ELEV = ["p10", "p25", "p50", "p75", "p90"]
P_KEYS_SLOPE = ["p50", "p75", "p90"]

def _percentile_dict_to_array(d: dict, keys: list[str]) -> np.ndarray:
    return np.array([float(d.get(k, np.nan)) for k in keys], dtype=np.float32)

def save_dem_features_to_h5(h5_path: str, dem_feats: dict):
    """
    Saves DEM features + DEM windows to /DEM group.
    Keeps BOTH:
      - individual scalar datasets
      - percentile arrays with clear labels
      - raw DEM windows (site + river) for inspection/plotting later
    """
    with h5py.File(h5_path, "a") as f:
        g = f.require_group("DEM")

        # --- helpers: overwrite if exists
        def put(name, data):
            if name in g:
                del g[name]
            g.create_dataset(name, data=data)

        # -------------------------
        # 1) Percentile arrays
        # -------------------------
        site_elev_arr = _percentile_dict_to_array(dem_feats["site_elev_m"], P_KEYS_ELEV)
        river_elev_arr = _percentile_dict_to_array(dem_feats["river_elev_m"], P_KEYS_ELEV)
        site_slope_arr = _percentile_dict_to_array(dem_feats["site_slope_deg"], P_KEYS_SLOPE)
        river_slope_arr = _percentile_dict_to_array(dem_feats["river_slope_deg"], P_KEYS_SLOPE)

        put("site_elev_percentiles_m", site_elev_arr)
        g["site_elev_percentiles_m"].attrs["keys"] = P_KEYS_ELEV

        put("river_elev_percentiles_m", river_elev_arr)
        g["river_elev_percentiles_m"].attrs["keys"] = P_KEYS_ELEV

        put("site_slope_percentiles_deg", site_slope_arr)
        g["site_slope_percentiles_deg"].attrs["keys"] = P_KEYS_SLOPE

        put("river_slope_percentiles_deg", river_slope_arr)
        g["river_slope_percentiles_deg"].attrs["keys"] = P_KEYS_SLOPE

        # -------------------------
        # 2) Scalar features
        # -------------------------
        put("slope_at_center_deg", np.float32(dem_feats.get("slope_at_center", np.nan)))
        put("height_above_river_m", np.float32(dem_feats.get("height_above_river", np.nan)))
        put("height_above_river_center_m", np.float32(dem_feats.get("height_above_river_center_m", np.nan)))
        put("local_relief_p75_minus_p25_m", np.float32(dem_feats.get("local_relief", np.nan)))

        # -------------------------
        # 3) DEM windows as arrays
        # -------------------------
        if "site_dem_window_m" in dem_feats:
            put("site_dem_window_m", dem_feats["site_dem_window_m"].astype(np.float32))

        if "river_dem_window_m" in dem_feats:
            put("river_dem_window_m", dem_feats["river_dem_window_m"].astype(np.float32))

        # -------------------------
        # 4) Metadata (optional but useful)
        # -------------------------
        g.attrs["resolution_m"] = float(dem_feats.get("resolution_m", np.nan))
        g.attrs["site_window_m"] = float(dem_feats.get("site_window_m", np.nan))
        g.attrs["river_window_m"] = float(dem_feats.get("river_window_m", np.nan))
        g.attrs["query_lat"] = float(dem_feats.get("query_lat", np.nan))
        g.attrs["query_lon"] = float(dem_feats.get("query_lon", np.nan))
        g.attrs["river_lat"] = float(dem_feats.get("river_lat", np.nan))
        g.attrs["river_lon"] = float(dem_feats.get("river_lon", np.nan))


# ----------------------------
# DEM fetching (Sentinel Hub)
# ----------------------------

def fetch_dem_cop30(bbox: BBox, resolution_m: float, config: SHConfig):
    """
    Returns:
      dem: 2D float32 (meters), NaN where invalid
      mask: bool validity mask
    """
    size = bbox_to_dimensions(bbox, resolution=resolution_m)

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{ bands: ["DEM", "dataMask"] }],
        output: { bands: 2, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(s) {
      return [s.DEM, s.dataMask];
    }
    """

    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.DEM,
            # time interval required by SDK even if DEM is static:
            time_interval=("2020-01-01", "2020-01-02"),
            other_args={"dataFilter": {"demInstance": "COPERNICUS_30"}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config
    )

    out = req.get_data()[0]  # H x W x 2
    dem = out[..., 0].astype(np.float32)
    mask = out[..., 1] > 0.5
    dem[~mask] = np.nan
    return dem, mask


# ----------------------------
# Stats + slope/gradient
# ----------------------------
def slope_deg_map(z: np.ndarray, pixel_m: float) -> np.ndarray:
    """
    Computes slope (degrees) for each pixel from DEM gradients.
    z: DEM meters in a metric CRS grid (you fetch in UTM bbox, so OK).
    """
    if not np.isfinite(z).any():
        return np.full_like(z, np.nan, dtype=np.float32)

    zf = z.astype(np.float32).copy()
    nanmask = ~np.isfinite(zf)
    if nanmask.any():
        zf[nanmask] = np.nanmedian(zf)

    gy, gx = np.gradient(zf, pixel_m, pixel_m)  # dz/dy, dz/dx
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    slope = np.degrees(slope_rad).astype(np.float32)

    slope[~np.isfinite(z)] = np.nan
    return slope


def sample_center_value(arr: np.ndarray) -> float:
    """Return value at the center pixel (robust to even sizes)."""
    if arr.size == 0:
        return np.nan
    cy = arr.shape[0] // 2
    cx = arr.shape[1] // 2
    return float(arr[cy, cx]) if np.isfinite(arr[cy, cx]) else np.nan


def nanpercentiles(arr: np.ndarray, ps=(10, 25, 50, 75, 90)) -> dict:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return {f"p{p}": np.nan for p in ps}
    out = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(vals, p))
    return out


# ----------------------------
# Remove outliers
# ----------------------------
def dem_outlier_cleanup_mad(dem: np.ndarray, ksize: int = 5, z_thr: float = 4.0) -> np.ndarray:
    """
    Replace DEM pixels that are local outliers using median + MAD robust z-score.
    - dem: float32 DEM with NaNs allowed
    - ksize: odd window size (3,5,7,...)
    - z_thr: threshold for robust z-score (3..5 typical)
    """
    dem = dem.astype(np.float32)
    out = dem.copy()

    # OpenCV medianBlur doesn't like NaNs. We'll fill NaNs temporarily.
    nan_mask = ~np.isfinite(dem)
    if np.any(nan_mask):
        # fill NaNs with local median approximation (global median fallback)
        fill_val = np.nanmedian(dem) if np.isfinite(np.nanmedian(dem)) else 0.0
        dem_filled = dem.copy()
        dem_filled[nan_mask] = fill_val
    else:
        dem_filled = dem

    # local median
    med = cv2.medianBlur(dem_filled, ksize)

    # local MAD: median(|x - med|)
    abs_dev = np.abs(dem_filled - med)
    mad = cv2.medianBlur(abs_dev, ksize)

    # robust z-score
    denom = 1.4826 * mad + 1e-6
    z = (dem_filled - med) / denom

    # outlier mask (ignore original NaNs)
    bad = (np.abs(z) > z_thr) & (~nan_mask)

    # replace outliers with local median
    out[bad] = med[bad]

    # restore NaNs
    out[nan_mask] = np.nan
    return out

# ----------------------------
# Main risk computation (same API)
# ----------------------------

def compute_dem_risk_features(
    query_lon, query_lat,
    river_lon, river_lat,
    site_window_m=1000,
    river_window_m=150,
    regional_window_m=None,
    resolution_m=30,
    config=None,
):
    """
    Returns DEM-based flood-risk features:
      - query DEM stats + slope
      - river DEM stats + slope
      - Δh (height above river) robust + center-to-center
      - slope at query point + slope percentiles (optional regional)
    """

    # --- fetch DEM around query + river in metric UTM bboxes
    site_bbox = square_bbox_utm(query_lon, query_lat, site_window_m)
    river_bbox = square_bbox_utm(river_lon, river_lat, river_window_m)

    z_site, m_site = fetch_dem_cop30(site_bbox, resolution_m, config)
    z_river, m_river = fetch_dem_cop30(river_bbox, resolution_m, config)

    # z_site = dem_outlier_cleanup_mad(z_site, ksize=3, z_thr=6.0)

    # --- elevation percentiles (robust stats)
    site_elev = nanpercentiles(z_site, ps=(10, 25, 50, 75, 90))
    river_elev = nanpercentiles(z_river, ps=(10, 25, 50, 75, 90))

    # --- slope maps
    slope_site = slope_deg_map(z_site, resolution_m)
    slope_river = slope_deg_map(z_river, resolution_m)

    site_slope = nanpercentiles(slope_site, ps=(50, 75, 90))
    river_slope = nanpercentiles(slope_river, ps=(50, 75, 90))

    # --- “slope at query point” (center pixel of site window)
    slope_at_query_deg = sample_center_value(slope_site)

    # --- Δh options:
    # (A) robust “height above river”: low-ish site ground minus median river area
    #     (using p10 helps avoid a single high building pixel)
    height_above_river_robust = np.nan
    if np.isfinite(site_elev["p50"]) and np.isfinite(river_elev["p10"]):
        height_above_river_robust = float(site_elev["p50"] - river_elev["p10"])

    # (B) center-to-center (sometimes useful, but can be noisy)
    site_center_elev = sample_center_value(z_site)
    river_center_elev = sample_center_value(z_river)
    height_above_river_center = np.nan
    if np.isfinite(site_center_elev) and np.isfinite(river_center_elev):
        height_above_river_center = float(site_center_elev - river_center_elev)

    # --- local relief near query (terrain variability)
    # small → flat surface (floodwater can spread)
    # large → rugged/variable terrain (water tends to channel)
    local_relief_p75_p25 = np.nan
    if np.isfinite(site_elev["p75"]) and np.isfinite(site_elev["p25"]):
        local_relief_p75_p25 = float(site_elev["p75"] - site_elev["p25"])

    out = {
        # "query_lon": float(query_lon),
        # "query_lat": float(query_lat),
        # "river_lon": float(river_lon),
        # "river_lat": float(river_lat),

        "resolution_m": float(resolution_m),
        "site_window_m": float(site_window_m),
        "river_window_m": float(river_window_m),

        "site_elev_m": site_elev,
        "river_elev_m": river_elev,

        "site_slope_deg": site_slope,
        "river_slope_deg": river_slope,

        "slope_at_center": slope_at_query_deg,

        "height_above_river": height_above_river_robust,
        "height_above_river_center_m": height_above_river_center,
        "local_relief": local_relief_p75_p25,
    }
    # Uncomment if you want DEM
    # out["site_dem_window_m"] = z_site.astype(np.float32)
    # out["river_dem_window_m"] = z_river.astype(np.float32)


    # --- optional regional context (bigger AOI around query)
    if regional_window_m:
        reg_bbox = square_bbox_utm(query_lon, query_lat, regional_window_m)
        z_reg, _ = fetch_dem_cop30(reg_bbox, resolution_m, config)
        slope_reg = slope_deg_map(z_reg, resolution_m)

        out["regional_window_m"] = float(regional_window_m)
        out["regional_elev_percentiles_m"] = nanpercentiles(z_reg, ps=(10, 25, 50, 75, 90))
        out["regional_slope_percentiles_deg"] = nanpercentiles(slope_reg, ps=(50, 75, 90))

    return out