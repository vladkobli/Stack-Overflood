import os
import pyproj
import rasterio
import numpy as np
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from skimage.transform import resize
from sentinelhub import (
    SHConfig, BBox, CRS, DataCollection, MimeType,
    SentinelHubRequest, SentinelHubCatalog, bbox_to_dimensions
)
import math
from pyproj import Transformer
from dataclasses import dataclass
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta, timezone

@dataclass
class AOIRequest:
    center_lon: float
    center_lat: float
    side_m: float
    resolution_m: float
    start_date: str
    end_date: str
  
# === 6. READING CROPPED BAND ===
def read_band_with_crop(band_path, aoi, target_shape=None):
    with rasterio.open(band_path) as src:
        # Get raster bounds as a shapely box in its own CRS
        raster_bounds = box(*src.bounds)

        # Transform AOI to raster CRS
        project = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        aoi_proj = shapely_transform(project, aoi)

        # Check for intersection
        if not raster_bounds.intersects(aoi_proj):
            raise ValueError(f"AOI does not intersect raster: {band_path}")

        # Now convert bounds to pixel indices
        minx, miny, maxx, maxy = aoi_proj.bounds
        row_start, col_start = src.index(minx, maxy)
        row_end, col_end = src.index(maxx, miny)

        # Clamp to raster dimensions
        row_start, row_end = max(0, row_start), min(src.height, row_end)
        col_start, col_end = max(0, col_start), min(src.width, col_end)

        height = row_end - row_start
        width = col_end - col_start

        if height <= 0 or width <= 0:
            raise ValueError(f"Cropped window has invalid shape: height={height}, width={width}")

        window = rasterio.windows.Window(col_start, row_start, width, height)
        band_data = src.read(1, window=window).astype("float32") / 10000

        return band_data
      
      
# === 7. COMPUTING FLOOD INDICES ===
def compute_flood_indices(img_data_dir, aoi):
    # Compute NDWI, NDVI, and NDMI for flood detection
    band_paths = {}
    for root, _, files in os.walk(img_data_dir):
        for file in files:
            if file.endswith('B03.jp2'):
                band_paths['B03'] = os.path.join(root, file)
            elif file.endswith('B04.jp2'):
                band_paths['B04'] = os.path.join(root, file)
            elif file.endswith('B08.jp2'):
                band_paths['B08'] = os.path.join(root, file)
            elif file.endswith('B11.jp2'):
                band_paths['B11'] = os.path.join(root, file)
        if len(band_paths) == 4:
            break

    if len(band_paths) < 4:
        print("Error: Required bands not found.")
        exit()

    # Read bands with cropping
    bands = {band: read_band_with_crop(path, aoi) for band, path in band_paths.items()}
    
    # Resample B11 to match B08 resolution (10m/px)
    target_shape = bands['B08'].shape
    bands["B11_2"] = resize(bands["B11"], target_shape, mode='reflect', anti_aliasing=False)

    # Compute NDWI, NDVI, NDMI
    #ndwi = (bands['B03'] - bands['B08']) / (bands['B03'] + bands['B08'])
    ndwi = (bands['B03'] - bands['B11_2']) / (bands['B03'] + bands['B11_2'])
    ndvi = (bands['B08'] - bands['B04']) / (bands['B08'] + bands['B04'])
    # ndvi = (2 * bands['B08'] + 1 - np.sqrt((2 * bands['B08'] + 1)**2 - 8 * (bands['B08'] - bands['B04']))) / 2
    ndmi = (bands['B08'] - bands['B11_2']) / (bands['B08'] + bands['B11_2'])

    return ndwi, ndvi, ndmi


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """Return EPSG code for UTM zone at lon/lat."""
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)

def fetch_s2_indices(req: AOIRequest, config: SHConfig):
    """
    Fetch S2 L2A indices already computed & aligned:
      - NDWI (McFeeters: G - NIR)
      - NDVI (NIR - R)
      - NDMI (NIR - SWIR1) where SWIR1 is B11
    Returns arrays and a dataMask to invalidate pixels.
    """
    bbox = square_bbox_utm(req.center_lon, req.center_lat, req.side_m)
    size = bbox_to_dimensions(bbox, resolution=req.resolution_m)

    evalscript = """
    //VERSION=3
    function setup() {
    return {
        input: [{ bands: ["B03", "B04", "B08", "B11", "dataMask"] }],
        output: { bands: 4, sampleType: "FLOAT32" } // MNDWI, NDVI, NDMI, mask
    };
    }

    function nd(a, b) {
    var denom = a + b;
    if (denom === 0) return NaN;
    return (a - b) / denom;
    }

    function evaluatePixel(s) {
    var ndwi = nd(s.B03, s.B11); // (Green - SWIR1)/(Green + SWIR1)
    var ndvi  = nd(s.B08, s.B04); // (NIR - Red)/(NIR + Red)
    var ndmi  = nd(s.B08, s.B11); // (NIR - SWIR1)/(NIR + SWIR1)
    return [ndwi, ndvi, ndmi, s.dataMask];
    }
"""

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(req.start_date, req.end_date),
            other_args={
                "dataFilter": {
                    # Optionally restrict clouds more with mosaicking
                    "maxCloudCoverage": 20
                },
                # Mosaicking decides which pixels to choose when multiple scenes overlap
                "processing": {
                    "mosaicking": "ORBIT"  # or "TILE"
                }
            }
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    out = request.get_data()[0]  # H x W x 4
    ndwi = out[..., 0].astype(np.float32)
    ndvi = out[..., 1].astype(np.float32)
    ndmi = out[..., 2].astype(np.float32)
    mask = out[..., 3] > 0.5

    # apply mask
    ndwi[~mask] = np.nan
    ndvi[~mask] = np.nan
    ndmi[~mask] = np.nan
    return ndwi, ndvi, ndmi, mask, bbox, size


def square_bbox_utm(center_lon: float, center_lat: float, side_m: float) -> BBox:
    """
    Create a square bbox in UTM meters around a lon/lat center.
    This gives you a TRUE 'resolution=10m' grid and stable alignment.
    """
    epsg = utm_epsg_from_lonlat(center_lon, center_lat)
    crs_utm = CRS(epsg)

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = to_utm.transform(center_lon, center_lat)
    half = side_m / 2.0

    return BBox((cx - half, cy - half, cx + half, cy + half), crs=crs_utm)


def fetch_s1_vv_vh_db(req: AOIRequest, config: SHConfig, orbit="IW"):
    """
    Fetch S1 VV/VH in dB on the same grid + dataMask.
    Returns vv_db, vh_db, mask
    """
    bbox = square_bbox_utm(req.center_lon, req.center_lat, req.side_m)
    size = bbox_to_dimensions(bbox, resolution=req.resolution_m)

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{ bands: ["VV", "VH", "dataMask"] }],
        output: { bands: 3, sampleType: "FLOAT32" }
      };
    }

    function toDb(x) {
    x = Math.max(x, 1e-6);
    return 10.0 * Math.log(x) / Math.LN10;
    }

    function evaluatePixel(s) {
      return [toDb(s.VV), toDb(s.VH), s.dataMask];
    }
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL1_IW,
            time_interval=(req.start_date, req.end_date),
            other_args={
                "dataFilter": {"polarization": "DV"},
                "processing": {
                    "backCoeff": "SIGMA0_ELLIPSOID",
                    "orthorectify": True,
                    "demInstance": "COPERNICUS",
                }
            }
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    out = request.get_data()[0]  # H x W x 3
    vv_db = out[..., 0].astype(np.float32)
    vh_db = out[..., 1].astype(np.float32)
    mask = out[..., 2] > 0.5

    vv_db[~mask] = np.nan
    vh_db[~mask] = np.nan
    return vv_db, vh_db, mask, bbox, size


def fetch_s2_all_bands(req, config, mosaicking="ORBIT", max_cloud=30):
    """
    Returns dict:
      {
        "B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12","mask"
      }
    All arrays are aligned to req.resolution_m grid (resampled by Sentinel Hub).
    """
    bbox = square_bbox_utm(req.center_lon, req.center_lat, req.side_m)
    size = bbox_to_dimensions(bbox, resolution=req.resolution_m)

    # All bands
    # evalscript = """
    # //VERSION=3
    # function setup() {
    #   return {
    #     input: [{
    #       bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12","dataMask"]
    #     }],
    #     output: { bands: 13, sampleType: "FLOAT32" }
    #   };
    # }
    # function evaluatePixel(s) {
    #   return [
    #     s.B01, s.B02, s.B03, s.B04, s.B05, s.B06, s.B07,
    #     s.B08, s.B8A, s.B09, s.B11, s.B12,
    #     s.dataMask
    #   ];
    # }
    # """
    
    # Only required bands
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{
          bands: ["B02","B03","B04","B08","B11","dataMask"]
        }],
        output: { bands: 6, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(s) {
      return [
        s.B02, s.B03, s.B04, s.B08, s.B11,
        s.dataMask
      ];
    }
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(req.start_date, req.end_date),
            other_args={
                "dataFilter": {"maxCloudCoverage": max_cloud},
                "processing": {"mosaicking": mosaicking},
            },
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    out = request.get_data()[0]  # H x W x 13
    # All bands
    # bands = {
    #     "B01": out[..., 0].astype(np.float32),
    #     "B02": out[..., 1].astype(np.float32),
    #     "B03": out[..., 2].astype(np.float32),
    #     "B04": out[..., 3].astype(np.float32),
    #     "B05": out[..., 4].astype(np.float32),
    #     "B06": out[..., 5].astype(np.float32),
    #     "B07": out[..., 6].astype(np.float32),
    #     "B08": out[..., 7].astype(np.float32),
    #     "B8A": out[..., 8].astype(np.float32),
    #     "B09": out[..., 9].astype(np.float32),
    #     "B11": out[..., 10].astype(np.float32),
    #     "B12": out[..., 11].astype(np.float32),
    #     "mask": (out[..., 12] > 0.5),
    # }
    bands = {
        "B02": out[..., 0].astype(np.float32),
        "B03": out[..., 1].astype(np.float32),
        "B04": out[..., 2].astype(np.float32),
        "B08": out[..., 3].astype(np.float32),
        "B11": out[..., 4].astype(np.float32),
        "mask": (out[..., 5] > 0.5),
    }

    # apply mask to bands (optional but recommended)
    m = bands["mask"]
    for k in ["B02", "B03", "B04", "B08", "B11"]:
        arr = bands[k]
        arr[~m] = np.nan
        bands[k] = arr

    return bands


def list_s1_acquisitions(config, bbox, start_date, end_date):
    """
    Returns a list of acquisition datetimes (UTC) for Sentinel-1 IW within interval.
    """
    catalog = SentinelHubCatalog(config=config)
    time_interval = (start_date, end_date)

    search_iter = catalog.search(
        collection=DataCollection.SENTINEL1_IW,
        bbox=bbox,
        time=time_interval,
        fields={"include": ["properties.datetime"], "exclude": []}
    )

    dts = []
    for item in search_iter:
        # item["properties"]["datetime"] is ISO string
        dt = item["properties"]["datetime"]
        if isinstance(dt, str):
            dts.append(datetime.fromisoformat(dt.replace("Z", "+00:00")))
        else:
            dts.append(dt)

    # sort & unique (sometimes duplicates)
    dts = sorted(set(dts))
    return dts

def list_s2_acquisitions(config, bbox, start_date, end_date, max_cloud=80):
    """
    Returns list of datetimes (UTC) for Sentinel-2 L2A acquisitions intersecting bbox in interval.
    Uses new SentinelHub Catalog API (filter/CQL2).
    """
    catalog = SentinelHubCatalog(config=config)

    time_interval = (f"{start_date}T00:00:00Z", f"{end_date}T23:59:59Z")

    # CQL2 filter: cloud cover threshold
    # NOTE: eo:cloud_cover is a property in STAC items
    cql2_filter = f"eo:cloud_cover <= {float(max_cloud)}"

    search_iter = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=time_interval,
        filter=cql2_filter,
        fields={"include": ["properties.datetime"], "exclude": []},
    )

    dts = []
    for item in search_iter:
        dt_str = item["properties"]["datetime"]  # e.g. "2024-06-10T10:15:21Z"
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        dts.append(dt)

    return sorted(set(dts))

def find_best_s2_acquisition(config, bbox, target_dt, max_days=3):
    catalog = SentinelHubCatalog(config=config)

    start = (target_dt - timedelta(days=max_days)).date().isoformat()
    end   = (target_dt + timedelta(days=max_days)).date().isoformat()

    search_iter = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=(start, end),
        fields={"include": ["properties.datetime", "properties.eo:cloud_cover"], "exclude": []}
    )

    candidates = []
    for item in search_iter:
        dt = item["properties"]["datetime"]
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00")) if isinstance(dt, str) else dt
        cloud = item["properties"].get("eo:cloud_cover", None)  # may be missing
        candidates.append((dt, cloud))

    if not candidates:
        return None

    # prefer low cloud if available, then closest time
    # You can tune weights here.
    def score(x):
        dt, cloud = x
        time_diff = abs((dt - target_dt).total_seconds())
        cloud_penalty = (cloud if cloud is not None else 50) * 1000  # cloud matters but not too extreme
        return time_diff + cloud_penalty

    best_dt, best_cloud = min(candidates, key=score)
    return best_dt


def dt_to_day_interval(dt):
    day = dt.date().isoformat()
    return (day, day)  # simplest


def dt_to_wide_interval(dt):
    start = (dt.date() - timedelta(days=1)).isoformat()
    end   = (dt.date() + timedelta(days=1)).isoformat()
    return (start, end)

def empty_s2_like(shape):
    ndwi = np.full(shape, np.nan, np.float32)
    ndvi = np.full(shape, np.nan, np.float32)
    ndmi = np.full(shape, np.nan, np.float32)
    mask = np.zeros(shape, dtype=bool)
    return ndwi, ndvi, ndmi, mask
