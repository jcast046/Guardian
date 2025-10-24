#!/usr/bin/env python3
"""EDA + Hotspotting for Guardian synthetic cases (Visualization Only).

This script focuses ONLY on visualization and KDE analysis.
For normalization, validation, and tabular counts, use run_all_llms.py.

Author: Joshua Castillo

Example:
    python eda_hotspot.py --input eda_out/eda_cases_min.jsonl --outdir eda_out --state VA --bw 1500
"""
# Standard library imports
from __future__ import annotations
import argparse
import os
import sys
from typing import Tuple, Optional

# Third-party imports for data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Optional geospatial imports with graceful fallback
# These are required for KDE mapping functionality
try:
    import geopandas as gpd  # type: ignore
    from shapely.geometry import Point  # type: ignore
except ImportError:  # pragma: no cover
    print("[WARN] Geo stack not fully available. Install geopandas/shapely for mapping.")
    gpd = None
    Point = None

#  machine learning imports for KDE computation
try:
    from sklearn.neighbors import KernelDensity  # type: ignore
except ImportError:  # pragma: no cover
    KernelDensity = None

#  basemap tiles for enhanced map visualization
_HAS_CTX = False
try:
    import contextily as ctx  # type: ignore
    _HAS_CTX = True
except ImportError:
    pass


# ---- ZERO-ARG DEFAULTS ----
DATA_DIR = "data"
OUT_DIR  = "eda_out"

DEFAULTS = dict(
    input=f"{OUT_DIR}/eda_cases_min.jsonl",
    outdir=OUT_DIR,
    state="VA",
    bw=30000.0,   # 30 km kernel bandwidth for state-scale analysis
    # common map behavior
    fixed_extent=True,
    shared_scale=True,
    # overlays (auto-skip if files missing)
    boundary_path=f"{DATA_DIR}/geo/va_boundary.geojson",
    clip_shape=f"{DATA_DIR}/geo/va_boundary.geojson",
    roads=f"{DATA_DIR}/va_roads.geojson",
    clip_to_boundary=True,
    roads_class_col="FUNC_CLASS",
    roads_cmap="rainbow",
    roads_width=0.8,
    roads_alpha=0.9,
    # heat colormaps per style
    dot_heat_cmap="inferno",
    dark_heat_cmap="magma",
)

def _exists(p): 
    return p and os.path.exists(p)


# -----------------------------
# Utility Functions
# -----------------------------

def ensure_outdir(path: str) -> None:
    """Create output directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Raises:
        OSError: If directory creation fails due to permissions or other system issues
    """
    os.makedirs(path, exist_ok=True)


def load_cases(path: str, state: str | None) -> pd.DataFrame:
    """Load and preprocess case data from JSONL or CSV files.
    
    This function handles multiple input formats and performs comprehensive data cleaning
    including coordinate validation, data type conversion, and optional state filtering.
    
    Args:
        path: Path to input file (.jsonl, .json, or .csv)
        state: Optional state filter (e.g., "VA"). Case-insensitive.
        
    Returns:
        Cleaned DataFrame with required columns:
        - age (int): Age in years
        - gender (str): Gender ("M" or "F")
        - county (str): County name
        - lat (float): Latitude (-90 to 90)
        - lon (float): Longitude (-180 to 180)
        - age_band (str): Age category ("≤12" or "13–17")
        
    Raises:
        SystemExit: If input file doesn't exist, has unsupported format, or missing required columns
        
    Note:
        The function performs extensive data validation:
        - Validates coordinate ranges (lat: -90 to 90, lon: -180 to 180)
        - Converts age to numeric and drops invalid entries
        - Standardizes gender values to uppercase
        - Creates age bands for demographic analysis
    """
    # Validate input file exists
    if not os.path.exists(path):
        print(f"[ERROR] Input not found: {path}")
        sys.exit(1)
    # Determine file format and load accordingly
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".json"):
        # Handle both compact and formatted JSONL formats
        import json
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('{') and '\n' in content:
                # Formatted JSONL - split by '}\n{' pattern for multi-line JSON
                import re
                # Split by '}\n{' but keep the braces for proper JSON reconstruction
                parts = re.split(r'}\s*\n\s*{', content)
                for i, part in enumerate(parts):
                    if i > 0:
                        part = '{' + part  # Restore opening brace
                    if i < len(parts) - 1:
                        part = part + '}'   # Restore closing brace
                    try:
                        records.append(json.loads(part))
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON objects
            else:
                # Compact JSONL - use pandas for better performance
                df = pd.read_json(path, lines=True)
                return df
        df = pd.DataFrame(records)
    elif ext == ".csv":
        # Load CSV files using pandas
        df = pd.read_csv(path)
    else:
        print(f"[ERROR] Unsupported input extension: {ext} (use .jsonl or .csv)")
        sys.exit(1)

    # Validate required columns are present
    required = {"age", "gender", "county", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {sorted(missing)}")
        sys.exit(1)

    # Apply state filter if specified and state column exists
    if state and "state" in df.columns:
        before = len(df)
        df = df[df["state"].astype(str).str.upper() == state.upper()].copy()
        print(f"[INFO] Filtered by state={state}: {before} -> {len(df)} rows")
    else:
        if state:
            print("[WARN] --state given but no 'state' column present; ignoring filter.")

    # Comprehensive data cleaning and validation
    # Remove rows with missing critical data
    df = df.dropna(subset=["age", "gender", "county", "lat", "lon"]).copy()
    
    # Validate coordinate ranges (standard lat/lon bounds)
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    
    # Standardize text fields to ensure consistency
    df["gender"] = df["gender"].astype(str).str.upper().str.strip()
    df["county"] = df["county"].astype(str).str.strip()
    
    # Convert age to numeric, dropping invalid entries
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age"]).copy()
    df["age"] = df["age"].astype(int)

    # Create age bands for demographic analysis and hotspotting
    df["age_band"] = np.where(df["age"] <= 12, "≤12", "13–17")

    return df


def create_visualizations(df: pd.DataFrame, outdir: str, top_n_counties: int = 20) -> None:
    """
    Create comprehensive visualizations for case distribution analysis.
    
    This function generates multiple chart types to analyze demographic and geographic
    patterns in the case data. It follows a single source of truth architecture where
    count computation is handled by run_all_llms.py.
    
    Args:
        df (pd.DataFrame): Cleaned case data with required columns
        outdir (str): Output directory for saving visualization files
        top_n_counties (int): Number of top counties to display in bar chart
        
    Generated Files:
        - distribution_summary.png: Combined 3-panel overview
        - age_hist.png: Age distribution histogram
        - gender_bar.png: Gender distribution bar chart
        - county_topN_bar.png: Top N counties horizontal bar chart
        
    
    """
    # Configure matplotlib for consistent, professional styling
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    
    # Create combined 3-panel overview figure for comprehensive analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Case Distribution Analysis (VA, Synthetic Dataset)", fontsize=16, fontweight='bold')
    
    # Panel 1: Age distribution histogram
    # Dynamically determine bin count based on data range
    age_bins = min(25, df["age"].nunique() or 10)
    axes[0].hist(df["age"], bins=age_bins, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_title("Age Distribution of Reported Cases", fontweight='bold')
    axes[0].set_xlabel("Age (Years)", fontsize=12)
    axes[0].set_ylabel("Number of Cases", fontsize=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].set_axisbelow(True)  # Place grid behind bars
    
    # Panel 2: Gender distribution bar chart
    gender_counts = df["gender"].value_counts()
    # Use distinct colors for binary gender data, fallback for other cases
    colors = ["lightcoral", "lightblue"] if len(gender_counts) == 2 else ["steelblue"]
    axes[1].bar(gender_counts.index, gender_counts.values, color=colors[:len(gender_counts)], 
                edgecolor="black", alpha=0.7)
    axes[1].set_title("Gender Distribution of Reported Cases", fontweight='bold')
    axes[1].set_xlabel("Gender", fontsize=12)
    axes[1].set_ylabel("Number of Cases", fontsize=12)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].set_axisbelow(True)
    
    # Panel 3: Top counties horizontal bar chart
    county_counts = df["county"].value_counts().head(top_n_counties)
    county_counts_sorted = county_counts.sort_values()  # Sort for better readability
    axes[2].barh(range(len(county_counts_sorted)), county_counts_sorted.values, 
                color="steelblue", edgecolor="black", alpha=0.7)
    axes[2].set_yticks(range(len(county_counts_sorted)))
    axes[2].set_yticklabels(county_counts_sorted.index)
    axes[2].set_title(f"Top {top_n_counties} Counties by Case Count", fontweight='bold')
    axes[2].set_xlabel("Number of Cases", fontsize=12)
    axes[2].set_ylabel("County", fontsize=12)
    axes[2].grid(axis="x", linestyle="--", alpha=0.6)
    axes[2].set_axisbelow(True)
    
    # Save combined overview with high quality settings
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "distribution_summary.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Generate individual charts for maximum flexibility and modular use
    # Individual age distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["age"], bins=age_bins, color="steelblue", edgecolor="black", alpha=0.7)
    plt.title("Age Distribution of Reported Cases (VA, Synthetic Dataset)", fontsize=14, fontweight='bold')
    plt.xlabel("Age (Years)", fontsize=12)
    plt.ylabel("Number of Cases", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "age_hist.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Individual gender distribution bar chart
    plt.figure(figsize=(8, 6))
    colors = ["lightcoral", "lightblue"] if len(gender_counts) == 2 else ["steelblue"]
    plt.bar(gender_counts.index, gender_counts.values, color=colors[:len(gender_counts)], 
            edgecolor="black", alpha=0.7)
    plt.title("Gender Distribution of Reported Cases (VA, Synthetic Dataset)", fontsize=14, fontweight='bold')
    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Number of Cases", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gender_bar.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Individual county distribution horizontal bar chart
    plt.figure(figsize=(12, 8))
    county_counts_sorted.plot(kind="barh", color="steelblue", edgecolor="black", alpha=0.7)
    plt.title(f"Top {top_n_counties} Counties by Case Count (VA, Synthetic Dataset)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Cases", fontsize=12)
    plt.ylabel("County", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "county_topN_bar.png"), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved visualizations to {outdir}")


# -----------------------------
# Geospatial Overlay Helpers
# -----------------------------

def _load_geo(path):
    """Load GeoJSON/Shapefile, return None if missing/invalid."""
    if not path or gpd is None or not os.path.exists(path):
        return None
    g = gpd.read_file(path)
    return g if not g.empty else None

def _plot_boundary(ax, boundary_gdf):
    """Plot polygon boundaries in Web Mercator, auto-filter geometry types."""
    if boundary_gdf is None: 
        return
    b = boundary_gdf.to_crs(epsg=3857)
    # keep only polygonal geometry
    b = b[b.geometry.type.isin(["Polygon","MultiPolygon"])]
    if len(b): 
        b.boundary.plot(ax=ax, color="#2b8cbe", linewidth=1.5, alpha=0.9)

def _plot_roads(ax, roads_gdf, class_col, cmap, lw, alpha):
    """Plot road lines with optional class-based coloring, handle categorical/numeric columns."""
    if roads_gdf is None: 
        return
    r = roads_gdf.to_crs(epsg=3857)
    # keep only linework
    r = r[r.geometry.type.str.contains("Line", na=False)]
    if not len(r): 
        return
    if class_col and class_col in r.columns and r[class_col].notna().any():
        if r[class_col].dtype.kind in "ifu":
            r.plot(ax=ax, column=class_col, cmap=cmap, linewidth=lw, alpha=alpha, legend=False)
        else:
            cats = r[class_col].astype(str)
            uniq = sorted(cats.unique())
            cm = plt.get_cmap(cmap, len(uniq))
            color_lu = {u: cm(i) for i,u in enumerate(uniq)}
            r.assign(_c=cats.map(color_lu)).plot(ax=ax, color=r.assign(_c=cats.map(color_lu))["_c"],
                                                 linewidth=lw, alpha=alpha)
    else:
        r.plot(ax=ax, color="purple", linewidth=lw, alpha=alpha)


def load_boundary(path: str):
    """Load a boundary polygon (e.g., Virginia) and reproject to EPSG:3857."""
    if not path or not os.path.exists(path):
        print(f"[WARN] Boundary file not found: {path}")
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        # assume lon/lat if not present
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(3857)[["geometry"]]

def _mask_raster_to_polygon(Z, X, Y, boundary_gdf):
    """
    Return a copy of Z with cells outside the boundary set to NaN.
    X, Y are the EPSG:3857 meshgrids matching Z.
    """
    if boundary_gdf is None or boundary_gdf.empty:
        return Z
    b = boundary_gdf.to_crs(epsg=3857)

    # dissolve to one polygon, clean topology
    try:
        poly = b.dissolve().geometry.unary_union.buffer(0)
    except Exception:
        poly = b.unary_union.buffer(0)

    Zm = Z.copy()
    try:
        from shapely import vectorized
        mask = vectorized.covers(poly, X, Y)  # includes boundary cells
    except Exception:
        mask = np.zeros_like(Z, dtype=bool)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                mask[i, j] = poly.covers(Point(float(X[i, j]), float(Y[i, j])))
    Zm[~mask] = np.nan
    return Zm


# -----------------------------
# Kernel Density Estimation (KDE) Functions
# -----------------------------

def _kde_heat(
    xs: np.ndarray, ys: np.ndarray, bw: float, gridsize: int = 1000,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Kernel Density Estimation (KDE) on a regular grid using scikit-learn.
    
    This function creates a density heatmap from point data using Gaussian kernels.
    It's optimized for geospatial data in Web Mercator projection (meters).
    
    Args:
        xs (np.ndarray): X coordinates in Web Mercator meters
        ys (np.ndarray): Y coordinates in Web Mercator meters  
        bw (float): Bandwidth parameter for KDE (meters). Larger values create smoother surfaces
        gridsize (int): Resolution of output grid (gridsize x gridsize)
        bounds (Optional[Tuple[float, float, float, float]]): Optional fixed bounds (x_min, x_max, y_min, y_max)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Grid coordinates and density values
            - X: X coordinate grid
            - Y: Y coordinate grid  
            - Z: Density values (log-transformed)
            
    Raises:
        RuntimeError: If scikit-learn is not available
        
    Note:
        The function uses Gaussian kernels and returns log-transformed density values
        for better visualization. Bandwidth should be chosen based on the scale of
        your data (typically 1000-3000 meters for county-level analysis).
    """
    # Validate scikit-learn availability
    if KernelDensity is None:
        raise RuntimeError("scikit-learn not available; install scikit-learn for KDE.")

    # Determine grid bounds - either from data or provided bounds
    if bounds is None:
        # Calculate bounds from data with 5% padding for better visualization
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # Add padding proportional to data range, with minimum padding
        pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1000
        pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1000
        x_min -= pad_x
        x_max += pad_x
        y_min -= pad_y
        y_max += pad_y
    else:
        # Use provided bounds for consistent extent across multiple maps
        x_min, x_max, y_min, y_max = bounds

    # Create regular grid for KDE computation
    xgrid = np.linspace(x_min, x_max, gridsize)
    ygrid = np.linspace(y_min, y_max, gridsize)
    X, Y = np.meshgrid(xgrid, ygrid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    # Fit KDE model and compute density
    kde = KernelDensity(bandwidth=bw, kernel="gaussian")
    kde.fit(np.vstack([xs, ys]).T)
    # Transform log-density back to density for visualization
    Z = np.exp(kde.score_samples(grid_points)).reshape(gridsize, gridsize)
    return X, Y, Z


def plot_kde_heatmap(
    gdf: "gpd.GeoDataFrame", title: str, outpath: str, bw: float, region: str = "VA",
    fixed_bounds: Optional[Tuple[float, float, float, float]] = None,
    vmin: Optional[float] = None, vmax: Optional[float] = None,
    # NEW:
    boundary_gdf=None, roads_gdf=None, roads_class_col=None, roads_cmap="rainbow",
    roads_width=0.8, roads_alpha=0.9, no_basemap=False, dark_style=False, heat_cmap_name="inferno"
) -> None:
    """Create and save a KDE heatmap visualization from geospatial point data with overlays and dual styles."""
    if gpd is None:
        print("[WARN] Skipping KDE map: geopandas not installed."); return
    if len(gdf) == 0:
        print(f"[WARN] No points to plot for {title}; skipping."); return

    gdf = gdf.to_crs(epsg=3857)
    xs = gdf.geometry.x.values; ys = gdf.geometry.y.values
    X, Y, Z = _kde_heat(xs, ys, bw=bw, bounds=fixed_bounds)

    # Mask the density to the VA polygon so the silhouette looks correct
    Z = _mask_raster_to_polygon(Z, X, Y, boundary_gdf)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # --- DARK STYLE CHROME ---
    if dark_style:
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # --- DRAW HEAT RASTER (with per-pixel alpha so NaNs are fully transparent) ---
    alpha_arr = np.where(np.isnan(Z), 0.0, 0.88)
    img = ax.imshow(
        Z, origin="lower",
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        cmap=plt.get_cmap(heat_cmap_name),
        vmin=vmin, vmax=vmax,
        interpolation="bilinear"
    )
    img.set_alpha(alpha_arr)

    # --- SILHOUETTE / OUTLINE ---
    if boundary_gdf is not None:
        b = boundary_gdf.to_crs(epsg=3857)
        face = "#0b0f1a" if dark_style else "none"
        edge = "#3dd5ff" if dark_style else "#2b8cbe"
        b.plot(ax=ax, facecolor=face, edgecolor=edge, linewidth=1.4, alpha=(0.9 if dark_style else 1.0), zorder=1)

    # --- LIMIT VIEW TO STATE BOUNDS (if we have them) ---
    if boundary_gdf is not None:
        b = boundary_gdf.to_crs(epsg=3857)
        x0, y0, x1, y1 = b.total_bounds
        padx, pady = 0.02*(x1-x0), 0.02*(y1-y0)
        ax.set_xlim(x0-padx, x1+padx)
        ax.set_ylim(y0-pady, y1+pady)
    elif fixed_bounds is not None:
        ax.set_xlim(fixed_bounds[0], fixed_bounds[1])
        ax.set_ylim(fixed_bounds[2], fixed_bounds[3])

    # --- BASEMAP FOR DOT-STYLE (do this AFTER xlim/ylim are set) ---
    if _HAS_CTX and not no_basemap:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron, alpha=1.0, zorder=0)
        except Exception as e:
            print(f"[WARN] Could not load basemap: {e}")

    # --- OPTIONAL ROADS OVERLAY (above basemap, below points) ---
    _plot_roads(ax, roads_gdf, roads_class_col, roads_cmap, roads_width, roads_alpha)

    # --- POINTS for reference ---
    ax.scatter(xs, ys, s=8, c="white", alpha=0.85, edgecolors="black", linewidth=0.4, zorder=3)

    # --- TITLE / AXES ---
    ax.set_title(f"{title} ({region})", fontsize=15, fontweight="bold",
                 color=("white" if dark_style else "black"), pad=20)
    if not dark_style:
        ax.set_xlabel("Longitude (Web Mercator)")
        ax.set_ylabel("Latitude (Web Mercator)")
        ax.grid(True, alpha=0.25, linestyle="--")

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label("Case Density", fontsize=12, fontweight="bold")
    if dark_style:
        cbar.ax.yaxis.set_tick_params(color="white")
        for t in cbar.ax.get_yticklabels(): t.set_color("white")

    plt.tight_layout(); plt.savefig(outpath, dpi=220, bbox_inches="tight"); plt.close()
    print(f"[OK] Saved {outpath}")


def run_hotspotting(
    df: pd.DataFrame, outdir: str, bw: float, region: str = "VA",
    fixed_extent: bool = False, shared_scale: bool = False,
    # NEW:
    clip_shape: Optional[str] = None, roads_path: Optional[str] = None,
    roads_class_col: Optional[str] = None, roads_cmap: str = "rainbow",
    roads_width: float = 0.8, roads_alpha: float = 0.9,
    no_basemap: bool = False, dark_style: bool = False, heat_cmap_name: str = "inferno",
    boundary_gdf=None,
) -> None:
    """
    Generate KDE hotspot maps for all cases and age bands.
    
    This function creates comprehensive spatial analysis by generating density maps
    for the entire dataset and demographic subsets. It supports consistent scaling
    and extent across maps for comparative analysis.
    
    Args:
        df (pd.DataFrame): Case data with lat/lon coordinates and age_band column
        outdir (str): Output directory for KDE map files
        bw (float): KDE bandwidth parameter in meters
        region (str): Region identifier for map titles (default: "VA")
        fixed_extent (bool): Use consistent map extent across all maps
        shared_scale (bool): Use consistent color scale across all maps
        
    Generated Files:
        - kde_all.png: Density map for all cases
        - kde_age_le12.png: Density map for age ≤12 cases
        - kde_age_13_17.png: Density map for age 13-17 cases
        
    Note:
        The function automatically handles coordinate system conversion and creates
        GeoDataFrames from lat/lon coordinates. Fixed extent and shared scaling
        options enable direct comparison between demographic groups.
    """
    # Validate geospatial dependencies
    if gpd is None:
        print("[WARN] Geo hotspotting skipped: install geopandas/shapely to enable maps.")
        return

    # Load overlays
    if boundary_gdf is None:
        boundary_gdf = _load_geo(clip_shape)
    roads_gdf    = _load_geo(roads_path)

    # Create GeoDataFrame from lat/lon coordinates
    gdf_all = gpd.GeoDataFrame(df.copy(), geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs="EPSG:4326")

    # Calculate fixed extent from boundary or data points if requested
    fixed_bounds = None
    if fixed_extent:
        if boundary_gdf is not None:
            b = boundary_gdf.to_crs(epsg=3857)
            x_min, y_min, x_max, y_max = b.total_bounds
            pad_x, pad_y = 0.02*(x_max-x_min), 0.02*(y_max-y_min)
            fixed_bounds = (x_min-pad_x, x_max+pad_x, y_min-pad_y, y_max+pad_y)
        else:
            gtmp = gpd.GeoDataFrame(df.copy(),
                    geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs="EPSG:4326").to_crs(epsg=3857)
            x_min, x_max = float(gtmp.geometry.x.min()), float(gtmp.geometry.x.max())
            y_min, y_max = float(gtmp.geometry.y.min()), float(gtmp.geometry.y.max())
            pad_x, pad_y = 0.05*(x_max-x_min), 0.05*(y_max-y_min)
            fixed_bounds = (x_min-pad_x, x_max+pad_x, y_min-pad_y, y_max+pad_y)
        print(f"[INFO] Using fixed extent bounds: {fixed_bounds}")

    # Calculate shared color scale from all cases if requested
    vmin = vmax = None
    if shared_scale:
        gtmp = gdf_all.to_crs(epsg=3857)
        X_all, Y_all, Z_all = _kde_heat(gtmp.geometry.x.values, gtmp.geometry.y.values, bw=bw, bounds=fixed_bounds)
        vmin, vmax = float(np.min(Z_all)), float(np.max(Z_all))
        print(f"[INFO] Using shared color scale vmin={vmin:.4e}, vmax={vmax:.4e}")

    # Generate KDE map for all cases
    plot_kde_heatmap(
        gdf_all, "KDE Hotspot Map – All Cases", os.path.join(outdir, "kde_all.png"),
        bw, region, fixed_bounds=fixed_bounds, vmin=vmin, vmax=vmax,
        boundary_gdf=boundary_gdf, roads_gdf=roads_gdf, roads_class_col=roads_class_col,
        roads_cmap=roads_cmap, roads_width=roads_width, roads_alpha=roads_alpha,
        no_basemap=no_basemap, dark_style=dark_style, heat_cmap_name=heat_cmap_name,
    )

    # Generate KDE maps for each age band
    for band, subset in df.groupby("age_band"):
        # Create GeoDataFrame for this age band
        gdf_band = gpd.GeoDataFrame(subset.copy(), geometry=[Point(xy) for xy in zip(subset["lon"], subset["lat"])], crs="EPSG:4326")
        # Determine filename based on age band
        fname = "kde_age_le12.png" if band == "≤12" else "kde_age_13_17.png"
        title = f"KDE Hotspot Map – Age {band}"
        plot_kde_heatmap(
            gdf_band, title, os.path.join(outdir, fname), bw, region,
            fixed_bounds=fixed_bounds, vmin=vmin, vmax=vmax,
            boundary_gdf=boundary_gdf, roads_gdf=roads_gdf, roads_class_col=roads_class_col,
            roads_cmap=roads_cmap, roads_width=roads_width, roads_alpha=roads_alpha,
            no_basemap=no_basemap, dark_style=dark_style, heat_cmap_name=heat_cmap_name,
        )


# -----------------------------
# Dual-Style Auto-Runner
# -----------------------------

def run_profiles_auto(df, state):
    """Generate both DOT-style and dark poster-style maps automatically."""
    # Load boundary once for both styles
    boundary_gdf = load_boundary(DEFAULTS["boundary_path"])
    
    # Add bounds validation messages
    if boundary_gdf is None or boundary_gdf.empty:
        print("[ERR] Boundary is empty – clipping will be skipped.")
    else:
        print(f"[INFO] VA bounds: {boundary_gdf.total_bounds}")
    
    # DOT-style (light basemap + roads)
    out_dot = os.path.join(DEFAULTS["outdir"], "maps_dot")
    os.makedirs(out_dot, exist_ok=True)
    run_hotspotting(
        df, out_dot, bw=DEFAULTS["bw"], region=state,
        fixed_extent=DEFAULTS["fixed_extent"], shared_scale=DEFAULTS["shared_scale"],
        clip_shape=DEFAULTS["clip_shape"] if _exists(DEFAULTS["clip_shape"]) else None,
        roads_path=DEFAULTS["roads"] if _exists(DEFAULTS["roads"]) else None,
        roads_class_col=DEFAULTS["roads_class_col"], roads_cmap=DEFAULTS["roads_cmap"],
        roads_width=DEFAULTS["roads_width"], roads_alpha=DEFAULTS["roads_alpha"],
        no_basemap=False, dark_style=False, heat_cmap_name=DEFAULTS["dot_heat_cmap"],
        boundary_gdf=boundary_gdf,
    )
    # Dark poster style (no basemap, no roads)
    out_dark = os.path.join(DEFAULTS["outdir"], "maps_dark")
    os.makedirs(out_dark, exist_ok=True)
    run_hotspotting(
        df, out_dark, bw=DEFAULTS["bw"], region=state,
        fixed_extent=DEFAULTS["fixed_extent"], shared_scale=DEFAULTS["shared_scale"],
        clip_shape=DEFAULTS["clip_shape"] if _exists(DEFAULTS["clip_shape"]) else None,
        roads_path=None, roads_class_col=None,
        roads_cmap=DEFAULTS["roads_cmap"], roads_width=DEFAULTS["roads_width"], roads_alpha=DEFAULTS["roads_alpha"],
        no_basemap=True, dark_style=True, heat_cmap_name=DEFAULTS["dark_heat_cmap"],
        boundary_gdf=boundary_gdf,
    )


# -----------------------------
# Main Execution Function
# -----------------------------

def main():
    """
    Main execution function for Guardian EDA + Hotspotting analysis.
    
    This function orchestrates the complete analysis pipeline including data loading,
    visualization generation, and spatial hotspotting. It follows a single source
    of truth architecture where count computation is handled by run_all_llms.py.
    
    Command Line Arguments:
        --input: Path to input data file (JSONL or CSV)
        --outdir: Output directory for generated files
        --state: Optional state filter for data subsetting
        --bw: KDE bandwidth parameter in meters
        --topN: Number of top counties to display
        --skip-counts: Skip visualization generation (use existing counts)
        --fixed-extent: Use consistent map extent across all KDE maps
        --shared-scale: Use consistent color scale across all KDE maps
        
    Workflow:
        1. Parse command line arguments
        2. Create output directory
        3. Load and clean case data
        4. Generate demographic visualizations (optional, controlled by --skip-counts)
        5. Create KDE hotspot maps
        
    Note:
        This function implements the single source of truth architecture where
        run_all_llms.py handles data normalization and validation, while this
        script focuses exclusively on visualization and spatial analysis.
    """
    # Configure command line argument parser
    parser = argparse.ArgumentParser(description="Guardian EDA + Hotspotting (Visualization Only)")
    parser.add_argument("--input", default=DEFAULTS["input"], help="Path to JSONL or CSV cases")
    parser.add_argument("--outdir", default=DEFAULTS["outdir"], help="Output directory")
    parser.add_argument("--state", default=DEFAULTS["state"], help="Optional state filter (e.g., VA)")
    parser.add_argument("--bw", type=float, default=DEFAULTS["bw"], help="KDE bandwidth in meters (Web Mercator)")
    parser.add_argument("--topN", type=int, default=20, help="Top-N counties bar chart")
    parser.add_argument("--skip-counts", action="store_true", help="Skip count computation (use counts from run_all_llms.py)")
    parser.add_argument("--fixed-extent", action="store_true", default=DEFAULTS["fixed_extent"], help="Use a shared geographic extent for all KDE maps")
    parser.add_argument("--shared-scale", action="store_true", default=DEFAULTS["shared_scale"], help="Use a shared color scale across KDE maps")
    parser.add_argument("--clip-shape", default=DEFAULTS["clip_shape"], help="Boundary shapefile/geojson path")
    parser.add_argument("--roads", default=DEFAULTS["roads"], help="Roads shapefile/geojson path")
    parser.add_argument("--roads-class-col", default=DEFAULTS["roads_class_col"], help="Road classification column name")
    parser.add_argument("--roads-cmap", default=DEFAULTS["roads_cmap"], help="Road colormap name")
    parser.add_argument("--roads-width", type=float, default=DEFAULTS["roads_width"], help="Road line width")
    parser.add_argument("--roads-alpha", type=float, default=DEFAULTS["roads_alpha"], help="Road transparency")
    parser.add_argument("--no-basemap", action="store_true", default=False, help="Disable basemap tiles")
    parser.add_argument("--dark-style", action="store_true", default=False, help="Use dark poster style")
    parser.add_argument("--cmap", default=DEFAULTS["dot_heat_cmap"], help="Heat colormap name")
    args = parser.parse_args()

    # Initialize output directory
    ensure_outdir(args.outdir)

    # Load and validate case data
    print("[INFO] Loading cases…")
    df = load_cases(args.input, args.state)
    print(f"[INFO] Rows after cleaning: {len(df)}")

    # Validate data availability
    if len(df) == 0:
        print("[ERROR] No data after cleaning - exiting")
        return

    # Generate demographic visualizations (optional based on --skip-counts flag)
    if not args.skip_counts:
        print("[INFO] Creating visualizations…")
        create_visualizations(df, args.outdir, top_n_counties=args.topN)
    else:
        print("[INFO] Skipping count computation (using counts from run_all_llms.py)")

    # Generate spatial hotspot analysis (both styles)
    print("[INFO] Running hotspotting (dot + dark)…")
    run_profiles_auto(df, state=(args.state or DEFAULTS["state"]))


    print("[DONE] EDA + Hotspotting complete.")


if __name__ == "__main__":
    main()
