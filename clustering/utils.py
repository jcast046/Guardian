"""Utility functions for clustering operations.

Provides helper functions for coordinate transformations, CRS handling,
data validation, parameter hashing, and coordinate scaling.
"""

import json
import hashlib
from typing import Tuple, Optional, Set, Dict, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def choose_projected_crs(bbox_lonlat: Tuple[float, float, float, float]) -> str:
    """Choose appropriate meters-based projected CRS based on bounding box.
    
    Args:
        bbox_lonlat: Bounding box as (minx, miny, maxx, maxy) in EPSG:4326 (lon, lat).
        
    Returns:
        CRS code string (e.g., "EPSG:32145").
        
    Selection rules:
        - VA North (lat >= 37.5): EPSG:32145 (State Plane meters)
        - VA South (lat < 37.5): EPSG:32146 (State Plane meters)
        - Or UTM Zone 17N (lon < -78): EPSG:26917
        - Or UTM Zone 18N (lon >= -78): EPSG:26918
    """
    minx, miny, maxx, maxy = bbox_lonlat
    center_lat = (miny + maxy) / 2.0
    center_lon = (minx + maxx) / 2.0
    
    if center_lat >= 37.5:
        return "EPSG:32145"
    else:
        return "EPSG:32146"


def validate_crs_units(crs: str, expected_units: str = "m") -> bool:
    """Validate that CRS has expected units.
    
    Args:
        crs: CRS code string (e.g., "EPSG:32145").
        expected_units: Expected units ("m" for meters, "ft" for feet).
        
    Returns:
        True if CRS units match expected units.
        
    Raises:
        ValueError: If CRS units don't match expected units.
        
    Note:
        EPSG:3857 (Web Mercator) is excluded due to high distortion away from equator.
    """
    meters_crs_codes = {"EPSG:32145", "EPSG:32146", "EPSG:26917", "EPSG:26918"}
    feet_crs_codes = {"EPSG:2283"}
    
    if expected_units == "m" and crs in meters_crs_codes:
        return True
    if expected_units == "ft" and crs in feet_crs_codes:
        return True
    
    try:
        gdf = gpd.GeoDataFrame([1], geometry=[Point(0, 0)], crs=crs)
        crs_obj = gdf.crs
        if crs_obj is None:
            raise ValueError(f"Invalid CRS: {crs}")
        
        if not crs_obj.is_projected:
            raise ValueError(f"CRS {crs} is geographic (degrees), not projected. Use a projected CRS for meters-based calculations.")
        
        try:
            axis_info = crs_obj.axis_info
            if axis_info and len(axis_info) > 0:
                unit_name = getattr(axis_info[0], 'unit_name', '').lower()
                if expected_units == "m":
                    if unit_name and "metre" not in unit_name and "meter" not in unit_name and "m" not in unit_name:
                        raise ValueError(
                            f"CRS {crs} has units '{unit_name}', expected meters. "
                            f"Use EPSG:32145, EPSG:32146, EPSG:26917, or EPSG:26918 for meters-based CRS."
                        )
                elif expected_units == "ft":
                    if unit_name and "foot" not in unit_name and "feet" not in unit_name and "ft" not in unit_name:
                        raise ValueError(
                            f"CRS {crs} has units '{unit_name}', expected feet. "
                            f"Use EPSG:2283 for feet-based CRS."
                        )
        except (AttributeError, IndexError):
            if expected_units == "m" and crs not in meters_crs_codes:
                import warnings
                warnings.warn(
                    f"Could not verify units for CRS {crs}. "
                    f"Expected meters. Use known meters-based CRS: EPSG:32145, EPSG:32146, EPSG:26917, EPSG:26918."
                )
        
        return True
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Could not validate CRS {crs}: {e}")


def to_projected(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    crs_from: str = "EPSG:4326", 
    crs_to: Optional[str] = None
) -> Tuple[gpd.GeoDataFrame, str]:
    """Convert DataFrame to GeoDataFrame in projected CRS.
    
    Args:
        df: DataFrame with coordinate columns.
        x_col: Name of longitude/x column.
        y_col: Name of latitude/y column.
        crs_from: Source CRS (default: "EPSG:4326").
        crs_to: Target projected CRS. If None, auto-select based on data bbox.
        
    Returns:
        Tuple of (GeoDataFrame in projected CRS, units_string).
        
    Raises:
        ValueError: If CRS units are not meters when expected.
    """
    geometry = [Point(lon, lat) for lon, lat in zip(df[x_col], df[y_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs_from)
    
    if crs_to is None:
        bbox = gdf.total_bounds
        crs_to = choose_projected_crs(bbox)
    
    validate_crs_units(crs_to, expected_units="m")
    gdf_proj = gdf.to_crs(crs_to)
    
    return gdf_proj, "m"


def to_geographic(
    gdf: gpd.GeoDataFrame, 
    crs_to: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """Convert GeoDataFrame back to geographic CRS.
    
    Args:
        gdf: GeoDataFrame in projected CRS.
        crs_to: Target geographic CRS (default: "EPSG:4326").
        
    Returns:
        GeoDataFrame with geometry in EPSG:4326.
    """
    return gdf.to_crs(crs_to)


def validate_coordinates(
    df: pd.DataFrame, 
    x_col: str = "lon", 
    y_col: str = "lat"
) -> pd.DataFrame:
    """Validate and clean coordinate data.
    
    Args:
        df: DataFrame with coordinate columns.
        x_col: Name of longitude column.
        y_col: Name of latitude column.
        
    Returns:
        Cleaned DataFrame with invalid coordinates removed.
    """
    df = df.copy()
    df = df.dropna(subset=[x_col, y_col])
    df = df[
        (df[x_col] >= -180) & (df[x_col] <= 180) &
        (df[y_col] >= -90) & (df[y_col] <= 90)
    ]
    
    return df.reset_index(drop=True)


def scale_coordinates(
    coords: np.ndarray, 
    method: str = "standard"
) -> Tuple[np.ndarray, Any]:
    """Scale coordinates for algorithms that require it.
    
    Args:
        coords: Array of shape (n_samples, n_features) with coordinates.
        method: Scaling method ("standard", "minmax", "robust").
        
    Returns:
        Tuple of (scaled coordinates, scaler object for inverse transformation).
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    coords_scaled = scaler.fit_transform(coords)
    return coords_scaled, scaler


HYPERPARAM_KEYS: Dict[str, Set[str]] = {
    "kmeans": {"n_clusters", "random_state"},
    "dbscan": {"eps_meters", "min_samples", "hotspot_mode"},
    "kde": {
        "bandwidth_meters", "kernel", "label_policy", "hotspot_mode",
        "iso_mass", "grid_res_m", "min_area_m2"
    }
}


def canonical_params_json(method: str, params: Dict[str, Any], include: Set[str]) -> str:
    """Create canonical JSON representation of hyperparameters.
    
    Args:
        method: Algorithm method name (e.g., "kmeans", "dbscan", "kde").
        params: Dictionary of all parameters.
        include: Set of parameter keys to include in hash.
        
    Returns:
        Canonical JSON string (sorted keys, compact separators).
        
    Note:
        Only includes hyperparameters specified in `include` set.
        Always includes __method__ for cross-method collision prevention.
        Keys are sorted for deterministic output.
    """
    filtered = {k: params[k] for k in sorted(params.keys()) if k in include}
    filtered["__method__"] = method
    return json.dumps(filtered, sort_keys=True, separators=(",", ":"))


def param_hash_from_json(params_json: str) -> str:
    """Generate deterministic SHA-1 hash from parameter JSON.
    
    Args:
        params_json: Canonical JSON string of parameters.
        
    Returns:
        10-character hex digest of SHA-1 hash.
    """
    return hashlib.sha1(params_json.encode()).hexdigest()[:10]


M_PER_MILE = 1609.344


def load_points_df(path: str, x_col: str = "lon", y_col: str = "lat") -> pd.DataFrame:
    """Load DataFrame from JSONL/CSV with coordinate validation.
    
    Handles both flat and nested JSON structures. Extracts coordinates
    from nested structures if needed.
    
    Args:
        path: Path to input file (.jsonl, .json, or .csv).
        x_col: Name of longitude/x column (default: "lon").
        y_col: Name of latitude/y column (default: "lat").
        
    Returns:
        DataFrame with validated coordinate columns.
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If required columns are missing or file format is unsupported.
    """
    import os
    import json
    import re
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".json"):
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('{') and '\n' in content:
                parts = re.split(r'}\s*\n\s*{', content)
                for i, part in enumerate(parts):
                    if i > 0:
                        part = '{' + part
                    if i < len(parts) - 1:
                        part = part + '}'
                    try:
                        records.append(json.loads(part))
                    except json.JSONDecodeError:
                        continue
            else:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        if not records:
            raise ValueError(f"No valid JSON records found in {path}")
        
        df = pd.DataFrame(records)
        
        if x_col not in df.columns or y_col not in df.columns:
            for idx, row in df.iterrows():
                if pd.isna(row.get(x_col)) or pd.isna(row.get(y_col)):
                    spatial = row.get("spatial", {})
                    if isinstance(spatial, dict):
                        if pd.isna(row.get(x_col)) and "last_seen_lon" in spatial:
                            df.at[idx, x_col] = spatial["last_seen_lon"]
                        if pd.isna(row.get(y_col)) and "last_seen_lat" in spatial:
                            df.at[idx, y_col] = spatial["last_seen_lat"]
                    
                    location = row.get("location", {})
                    if isinstance(location, dict):
                        if pd.isna(row.get(x_col)) and "lon" in location:
                            df.at[idx, x_col] = location["lon"]
                        if pd.isna(row.get(y_col)) and "lat" in location:
                            df.at[idx, y_col] = location["lat"]
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext} (use .jsonl, .json, or .csv)")
    
    required = {x_col, y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(                        f"Missing required columns: {sorted(missing)}. "
                        f"Available columns: {sorted(df.columns.tolist())}")
    
    df = df.dropna(subset=[x_col, y_col]).copy()
    df = df[(df[x_col].between(-180, 180)) & (df[y_col].between(-90, 90))].copy()
    
    return df


def _centroid_lonlat(geom) -> Tuple[float, float]:
    """Extract lon/lat from Point or centroid of Polygon.
    
    Args:
        geom: Shapely geometry (Point, Polygon, etc.).
        
    Returns:
        Tuple of (lon, lat) in degrees.
    """
    if geom.geom_type == "Point":
        return float(geom.x), float(geom.y)
    else:
        centroid = geom.centroid
        return float(centroid.x), float(centroid.y)


def _equiv_area_radius_m(geom) -> float:
    """Calculate equivalent area radius for polygons.
    
    Computes the radius of a circle with the same area as the polygon.
    
    Args:
        geom: Shapely geometry (Polygon, MultiPolygon, etc.).
        
    Returns:
        Equivalent radius in meters (assuming geometry is in meters CRS).
    """
    area_m2 = geom.area
    if area_m2 <= 0:
        return 0.0
    return np.sqrt(area_m2 / np.pi)


def gdf_to_points_json(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Convert GeoDataFrame hotspots to point JSON format.
    
    Exports hotspots GeoDataFrame to JSON array of objects format.
    Expects gdf to have 'radius_m' column (computed by runner scripts).
    For KDE polygons without radius_m, computes equivalent area radius.
    
    Args:
        gdf: GeoDataFrame from hotspots() method with score, geometry, etc.
        out_path: Output file path for JSON file.
        
    JSON Format:
        Array of objects: [{"lon": ..., "lat": ..., "weight": ..., "radius_miles": ..., ...}, ...]
    """
    import os
    
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    items = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        lon, lat = _centroid_lonlat(geom)
        weight = float(row["score"])
        
        if "radius_m" in row and pd.notna(row["radius_m"]):
            radius_m = float(row["radius_m"])
        elif geom.geom_type != "Point":
            if "crs_proj" in row and pd.notna(row["crs_proj"]):
                from shapely.geometry import shape
                import geopandas as gpd
                temp_gdf = gpd.GeoDataFrame([1], geometry=[geom], crs=gdf.crs)
                temp_gdf_proj = temp_gdf.to_crs(row["crs_proj"])
                geom_proj = temp_gdf_proj.geometry.iloc[0]
                radius_m = _equiv_area_radius_m(geom_proj)
            else:
                radius_m = _equiv_area_radius_m(geom)
        else:
            radius_m = 0.0
        
        radius_miles = radius_m / M_PER_MILE if radius_m > 0 else 0.0
        
        item = {
            "lon": lon,
            "lat": lat,
            "weight": weight,
            "radius_miles": radius_miles,
        }
        
        if "method" in row:
            item["method"] = str(row["method"])
        if "params_hash" in row:
            item["params_hash"] = str(row["params_hash"])
        if "params_json" in row:
            item["params_json"] = str(row["params_json"])
        if "crs" in row:
            item["crs"] = str(row["crs"])
        
        items.append(item)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, separators=(",", ":"), ensure_ascii=False, indent=2)


def points_json_to_gdf(json_path: str) -> gpd.GeoDataFrame:
    """Load point JSON and create GeoDataFrame.
    
    Reverse of gdf_to_points_json(). Loads JSON array of objects
    or legacy array-of-arrays format and creates GeoDataFrame with Point geometries.
    
    Supports both formats:
    - New format: [{"lon": ..., "lat": ..., "weight": ..., ...}, ...]
    - Legacy format: [[lon, lat, weight, radius_miles], ...] (from eda_hotspot.py)
    
    Args:
        json_path: Path to JSON file.
        
    Returns:
        GeoDataFrame with Point geometries and metadata columns.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    if not items:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    is_legacy_format = isinstance(items[0], list)
    geometries = []
    data = []
    
    for item in items:
        if is_legacy_format:
            if len(item) >= 2:
                lon = float(item[0])
                lat = float(item[1])
                weight = float(item[2]) if len(item) > 2 else 0.0
                radius_miles = float(item[3]) if len(item) > 3 else 0.0
            else:
                continue
            
            row_data = {
                "weight": weight,
                "radius_miles": radius_miles,
                "method": "kde",
            }
        else:
            lon = float(item["lon"])
            lat = float(item["lat"])
            
            row_data = {
                "weight": float(item.get("weight", 0.0)),
                "radius_miles": float(item.get("radius_miles", 0.0)),
            }
            
            if "method" in item:
                row_data["method"] = str(item["method"])
            if "params_hash" in item:
                row_data["params_hash"] = str(item["params_hash"])
            if "params_json" in item:
                row_data["params_json"] = str(item["params_json"])
            if "crs" in item:
                row_data["crs"] = str(item["crs"])
        
        geometries.append(Point(lon, lat))
        data.append(row_data)
    
    gdf = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")
    
    if "radius_miles" in gdf.columns:
        gdf["radius_m"] = gdf["radius_miles"] * M_PER_MILE
    
    return gdf

