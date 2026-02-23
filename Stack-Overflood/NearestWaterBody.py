import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import warnings
from shapely.ops import nearest_points

EU_CRS = "EPSG:3035"
# Rough Europe extent in EPSG:3035
EU_EXTENT_3035 = (2000000, 1000000, 8000000, 5500000)  # xmin,ymin,xmax,ymax

def point_to_tile_id(lon, lat, tile_m=25000, extent=EU_EXTENT_3035):
    xmin, ymin, xmax, ymax = extent
    p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(EU_CRS).iloc[0]
    x, y = p.x, p.y
    c = int((x - xmin) // tile_m)
    r = int((y - ymin) // tile_m)
    return f"R{r:04d}_C{c:04d}", (r, c)

def neighbor_tile_ids(rc, radius=1):
    r0, c0 = rc
    out = []
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            out.append(f"R{(r0+dr):04d}_C{(c0+dc):04d}")
    return out

def closest_point_lonlat(geom_3035, p_3035: Point):
    """
    Return (lon, lat) of the closest point on geom to p.
    Done in EPSG:3035 for correct distance, then converted to EPSG:4326.
    """
    if geom_3035 is None or geom_3035.is_empty:
        return None

    # nearest_points returns (point_on_geom, point_on_p) or vice versa depending on order
    p_on_geom, _ = nearest_points(geom_3035, p_3035)

    # Convert that point back to lon/lat
    pt_ll = gpd.GeoSeries([p_on_geom], crs=EU_CRS).to_crs("EPSG:4326").iloc[0]
    return float(pt_ll.x), float(pt_ll.y)  # lon, lat

def nearest_geom_distance_m(gdf_3035: gpd.GeoDataFrame, p_3035: Point,
                            start_radius_m=2000, grow=2.0, max_radius_m=200000):
    """
    Returns: (min_dist_m, nearest_row_index)
    Uses sindex.intersection(bbox) -> exact distance.
    Works with old GeoPandas sindex API.
    """
    if gdf_3035.empty:
        return np.inf, None

    sidx = gdf_3035.sindex

    r = float(start_radius_m)
    while r <= max_radius_m:
        # bbox around point
        minx, miny, maxx, maxy = (p_3035.x - r, p_3035.y - r, p_3035.x + r, p_3035.y + r)
        cand_idx = list(sidx.intersection((minx, miny, maxx, maxy)))

        if cand_idx:
            cands = gdf_3035.iloc[cand_idx]
            # compute exact distances in meters (EPSG:3035)
            dists = cands.geometry.distance(p_3035)
            j = int(dists.idxmin())
            return float(dists.loc[j]), j

        r *= grow

    return np.inf, None

def nearest_water_distance(lon, lat, tile_index_csv, tile_m=25000, search_radius_tiles=1):
    idx = pd.read_csv(tile_index_csv)

    tid, rc = point_to_tile_id(lon, lat, tile_m=tile_m)
    candidates_tiles = neighbor_tile_ids(rc, radius=search_radius_tiles)

    sub = idx[idx["tile_id"].isin(candidates_tiles)]
    if sub.empty:
        return None, None, None, None

    # Query point in metric CRS
    p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(EU_CRS).iloc[0]

    best_dist = float("inf")
    best_meta = None

    for (gpkg_path, layer), _ in sub.groupby(["gpkg_path", "layer"]):
        gdf = gpd.read_file(gpkg_path, layer=layer)
        gdf = gdf[gdf.geometry.notna()]
        if gdf.empty:
            continue
        gdf = gdf.to_crs(EU_CRS)

        dist_m, row_label = nearest_geom_distance_m(gdf, p)
        if row_label is None:
            continue

        if dist_m < best_dist:
            geom = gdf.loc[row_label].geometry
            nearest_lon, nearest_lat = closest_point_lonlat(geom, p)

            best_dist = dist_m
            best_meta = {
                "gpkg": gpkg_path,
                "layer": layer,
                "row": int(row_label) if isinstance(row_label, (int, np.integer)) else str(row_label),
            }

    return best_dist, nearest_lat, nearest_lon, best_meta

# Suppress warnings
# 1) "Measured (M) geometry types are not supported..."
warnings.filterwarnings(
    "ignore",
    message=r"Measured \(M\) geometry types are not supported\..*",
    category=UserWarning,
)

# 2) "Non-conformant content for record ... BEGLIFEVER ..."
warnings.filterwarnings(
    "ignore",
    message=r"Non-conformant content for record .* in column BEGLIFEVER, .* successfully parsed",
    category=RuntimeWarning,
)
