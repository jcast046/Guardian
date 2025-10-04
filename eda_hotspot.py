#!/usr/bin/env python3
"""
EDA + Hotspotting for Guardian synthetic cases (Visualization Only)

SINGLE SOURCE OF TRUTH ARCHITECTURE
-----------------------------------
This script focuses ONLY on visualization and KDE analysis.
For normalization, validation, and tabular counts, use run_all_llms.py.

WORKFLOW
--------
1. Normalization + Validation (run_all_llms.py):
   - Produces: eda_out/eda_cases_min.jsonl (authoritative input)
   - Produces: eda_out/eda_counts.json (authoritative counts)
   - Produces: eda_out/validation_report.json

2. Visualization (eda_hotspot.py):
   - Reads: eda_out/eda_cases_min.jsonl
   - Generates: Charts, histograms, and KDE heat maps
   - Optionally skips count recomputation

USAGE
-----
# Full visualization (includes basic charts)
python eda_hotspot.py --input eda_out/eda_cases_min.jsonl --outdir eda_out --state VA --bw 1500

# KDE only (skip count recomputation)
python eda_hotspot.py --input eda_out/eda_cases_min.jsonl --outdir eda_out --state VA --bw 1500 --skip-counts

# With locked map extent + shared color scale + markdown report
python eda_hotspot.py --input eda_out/eda_cases_min.jsonl --outdir eda_out --state VA --bw 1500 \
  --fixed-extent --shared-scale


DEPENDENCIES
------------
- pandas
- numpy
- matplotlib
- geopandas
- shapely
- scikit-learn (for KernelDensity)
- contextily (optional; for basemap tiles — script works without it)

Outputs
-------
- PNGs: distribution_summary.png, age_hist.png, gender_bar.png, county_topN_bar.png (if not --skip-counts)
- PNGs: kde_all.png, kde_age_le12.png, kde_age_13_17.png

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


# -----------------------------
# Utility Functions
# -----------------------------

def ensure_outdir(path: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
        
    Raises:
        OSError: If directory creation fails due to permissions or other system issues
    """
    os.makedirs(path, exist_ok=True)


def load_cases(path: str, state: str | None) -> pd.DataFrame:
    """
    Load and preprocess case data from JSONL or CSV files.
    
    This function handles multiple input formats and performs comprehensive data cleaning
    including coordinate validation, data type conversion, and optional state filtering.
    
    Args:
        path (str): Path to input file (.jsonl, .json, or .csv)
        state (str | None): Optional state filter (e.g., "VA"). Case-insensitive.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with required columns:
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
# Kernel Density Estimation (KDE) Functions
# -----------------------------

def _kde_heat(
    xs: np.ndarray, ys: np.ndarray, bw: float, gridsize: int = 400,
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
    vmin: Optional[float] = None, vmax: Optional[float] = None
) -> None:
    """
    Create and save a KDE heatmap visualization from geospatial point data.
    
    This function generates publication-quality density maps showing spatial
    concentration patterns of case data. It supports consistent scaling and
    extent across multiple maps for comparative analysis.
    
    Args:
        gdf (gpd.GeoDataFrame): Geospatial DataFrame with point geometries
        title (str): Chart title for the heatmap
        outpath (str): Output file path for the saved image
        bw (float): KDE bandwidth parameter in meters
        region (str): Region identifier for title (default: "VA")
        fixed_bounds (Optional[Tuple[float, float, float, float]]): Fixed map extent (x_min, x_max, y_min, y_max)
        vmin (Optional[float]): Minimum value for color scale (for shared scaling)
        vmax (Optional[float]): Maximum value for color scale (for shared scaling)
        
    Returns:
        None: Saves image file to outpath
        
    Note:
        The function automatically handles coordinate system conversion to Web Mercator
        for accurate KDE computation. Optional basemap tiles are added if contextily
        is available. The 'inferno' colormap is used for high contrast visualization.
    """
    # Validate geospatial dependencies
    if gpd is None:
        print("[WARN] Skipping KDE map: geopandas not installed.")
        return

    # Check for empty data
    if len(gdf) == 0:
        print(f"[WARN] No points to plot for {title}; skipping.")
        return

    # Convert to Web Mercator projection for accurate KDE computation
    gdf = gdf.to_crs(epsg=3857)
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values

    # Compute KDE density surface
    X, Y, Z = _kde_heat(xs, ys, bw=bw, bounds=fixed_bounds)

    # Create high-quality figure
    plt.figure(figsize=(12, 10))
    
    # Generate heatmap with optional shared scaling
    img = plt.imshow(
        Z,
        origin="lower",  # Place origin at bottom-left for geographic convention
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        cmap="inferno",  # High-contrast colormap for density visualization
        alpha=0.85,      # Slight transparency for overlay effects
        vmin=vmin,       # Optional minimum for shared color scale
        vmax=vmax,       # Optional maximum for shared color scale
    )
    
    # Add professional colorbar with descriptive label
    cbar = plt.colorbar(img, shrink=0.8, aspect=20)
    cbar.set_label("Case Density", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Overlay original points for context
    plt.scatter(xs, ys, s=8, c='white', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Enhanced title with region information
    plt.title(f"{title} ({region})", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Longitude (Web Mercator)", fontsize=12)
    plt.ylabel("Latitude (Web Mercator)", fontsize=12)
    
    # Add subtle grid for better spatial reference
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Optional basemap tiles for geographic context
    if _HAS_CTX:
        try:
            ax = plt.gca()
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Stamen.TonerLite, alpha=0.6)
        except Exception as e:
            print(f"[WARN] Could not load basemap: {e}")

    # Save with high quality settings
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {outpath}")


def run_hotspotting(
    df: pd.DataFrame, outdir: str, bw: float, region: str = "VA",
    fixed_extent: bool = False, shared_scale: bool = False
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

    # Create GeoDataFrame from lat/lon coordinates
    gdf_all = gpd.GeoDataFrame(df.copy(), geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs="EPSG:4326")

    # Calculate fixed extent from all data points if requested
    fixed_bounds = None
    if fixed_extent:
        gtmp = gdf_all.to_crs(epsg=3857)  # Convert to Web Mercator for calculations
        x_min, x_max = float(gtmp.geometry.x.min()), float(gtmp.geometry.x.max())
        y_min, y_max = float(gtmp.geometry.y.min()), float(gtmp.geometry.y.max())
        # Add 5% padding for better visualization
        pad_x = 0.05 * (x_max - x_min)
        pad_y = 0.05 * (y_max - y_min)
        fixed_bounds = (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)
        print(f"[INFO] Using fixed extent bounds: {fixed_bounds}")

    # Calculate shared color scale from all cases if requested
    vmin = vmax = None
    if shared_scale:
        gtmp = gdf_all.to_crs(epsg=3857)
        X_all, Y_all, Z_all = _kde_heat(gtmp.geometry.x.values, gtmp.geometry.y.values, bw=bw, bounds=fixed_bounds)
        vmin, vmax = float(np.min(Z_all)), float(np.max(Z_all))
        print(f"[INFO] Using shared color scale vmin={vmin:.4e}, vmax={vmax:.4e}")

    # Generate KDE map for all cases
    plot_kde_heatmap(gdf_all, "KDE Hotspot Map – All Cases",
                      os.path.join(outdir, "kde_all.png"), bw, region,
                      fixed_bounds=fixed_bounds, vmin=vmin, vmax=vmax)

    # Generate KDE maps for each age band
    for band, subset in df.groupby("age_band"):
        # Create GeoDataFrame for this age band
        gdf_band = gpd.GeoDataFrame(subset.copy(), geometry=[Point(xy) for xy in zip(subset["lon"], subset["lat"])], crs="EPSG:4326")
        # Determine filename based on age band
        fname = "kde_age_le12.png" if band == "≤12" else "kde_age_13_17.png"
        title = f"KDE Hotspot Map – Age {band}"
        plot_kde_heatmap(gdf_band, title, os.path.join(outdir, fname), bw, region,
                          fixed_bounds=fixed_bounds, vmin=vmin, vmax=vmax)


# -----------------------------
# Report Generation Functions
# -----------------------------



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
    parser.add_argument("--input", default="eda_out/eda_cases_min.jsonl", help="Path to JSONL or CSV cases")
    parser.add_argument("--outdir", default="eda_out", help="Output directory")
    parser.add_argument("--state", default="VA", help="Optional state filter (e.g., VA)")
    parser.add_argument("--bw", type=float, default=1500.0, help="KDE bandwidth in meters (Web Mercator)")
    parser.add_argument("--topN", type=int, default=20, help="Top-N counties bar chart")
    parser.add_argument("--skip-counts", action="store_true", help="Skip count computation (use counts from run_all_llms.py)")
    parser.add_argument("--fixed-extent", action="store_true", help="Use a shared geographic extent for all KDE maps")
    parser.add_argument("--shared-scale", action="store_true", help="Use a shared color scale across KDE maps")
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

    # Generate spatial hotspot analysis
    print("[INFO] Running hotspotting…")
    run_hotspotting(df, args.outdir, bw=args.bw, region=args.state or "VA",
                    fixed_extent=args.fixed_extent, shared_scale=args.shared_scale)


    print("[DONE] EDA + Hotspotting complete.")


if __name__ == "__main__":
    main()
