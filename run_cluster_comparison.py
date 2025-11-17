#!/usr/bin/env python3
"""Cluster comparison runner.

Loads hotspots from all three clustering methods, computes Jaccard overlaps,
and generates comparison visualizations.

Usage:
    # Run with defaults (no arguments needed)
    python run_cluster_comparison.py
    
    # Or with custom arguments
    python run_cluster_comparison.py --in-dir eda_out --topN 10
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Optional

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union

from clustering import points_json_to_gdf, gdf_to_points_json, M_PER_MILE, to_projected, to_geographic


def jaccard_overlap(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> float:
    """Compute Jaccard overlap (IoU) between two GeoDataFrames.
    
    Args:
        gdf1: First GeoDataFrame (geometries will be unioned).
        gdf2: Second GeoDataFrame (geometries will be unioned).
        
    Returns:
        Jaccard score (intersection.area / union.area), range [0, 1].
    """
    if len(gdf1) == 0 or len(gdf2) == 0:
        return 0.0
    
    # Union all geometries in each GeoDataFrame
    union1 = unary_union(gdf1.geometry)
    union2 = unary_union(gdf2.geometry)
    
    if union1.is_empty or union2.is_empty:
        return 0.0
    
    # Compute intersection and union
    intersection = union1.intersection(union2)
    union = union1.union(union2)
    
    if union.is_empty or union.area == 0:
        return 0.0
    
    jaccard = intersection.area / union.area
    return float(jaccard)


def buffer_points_by_radius(gdf: gpd.GeoDataFrame, crs_proj: str) -> gpd.GeoDataFrame:
    """Buffer point geometries by radius_miles.
    
    Args:
        gdf: GeoDataFrame with Point geometries and radius_miles column.
        crs_proj: Projected CRS to use for buffering (must be meters-based).
        
    Returns:
        GeoDataFrame with buffered geometries (Polygons) in geographic CRS.
    """
    # Convert to projected CRS for buffering
    gdf_proj = gdf.to_crs(crs_proj)
    
    # Convert radius_miles to meters
    if "radius_m" in gdf_proj.columns:
        radius_m = gdf_proj["radius_m"]
    elif "radius_miles" in gdf_proj.columns:
        radius_m = gdf_proj["radius_miles"] * M_PER_MILE
    else:
        # No radius - return empty polygons or points
        print("[WARN] No radius column found, skipping buffering")
        return gdf
    
    # Buffer points
    gdf_proj["geometry"] = gdf_proj.geometry.buffer(radius_m)
    
    # Convert back to geographic CRS
    gdf_buffered = gdf_proj.to_crs("EPSG:4326")
    
    return gdf_buffered


def plot_cluster_comparison(
    gdf_kmeans: gpd.GeoDataFrame,
    gdf_dbscan: gpd.GeoDataFrame,
    gdf_kde: gpd.GeoDataFrame,
    out_dir: str
) -> None:
    """Generate side-by-side comparison maps.
    
    Args:
        gdf_kmeans: K-Means hotspots GeoDataFrame.
        gdf_dbscan: DBSCAN hotspots GeoDataFrame.
        gdf_kde: KDE hotspots GeoDataFrame.
        out_dir: Output directory for plots.
    """
    # Determine common bounds
    all_gdfs = [g for g in [gdf_kmeans, gdf_dbscan, gdf_kde] if len(g) > 0]
    if not all_gdfs:
        print("[WARN] No hotspots to plot")
        return
    
    # Get combined bounds
    bounds_list = [g.total_bounds for g in all_gdfs]
    minx = min(b[0] for b in bounds_list)
    miny = min(b[1] for b in bounds_list)
    maxx = max(b[2] for b in bounds_list)
    maxy = max(b[3] for b in bounds_list)
    
    # Add padding
    pad_x = (maxx - minx) * 0.1
    pad_y = (maxy - miny) * 0.1
    
    # Create side-by-side figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cluster Comparison: K-Means vs DBSCAN vs KDE", fontsize=16, fontweight="bold")
    
    # Plot K-Means
    ax = axes[0]
    if len(gdf_kmeans) > 0:
        gdf_kmeans.plot(ax=ax, color="red", markersize=50, alpha=0.7, edgecolor="black", linewidth=1)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_title("K-Means", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    
    # Plot DBSCAN
    ax = axes[1]
    if len(gdf_dbscan) > 0:
        gdf_dbscan.plot(ax=ax, color="blue", markersize=50, alpha=0.7, edgecolor="black", linewidth=1)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_title("DBSCAN", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    
    # Plot KDE
    ax = axes[2]
    if len(gdf_kde) > 0:
        gdf_kde.plot(ax=ax, color="green", alpha=0.5, edgecolor="black", linewidth=1)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_title("KDE", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "plots", "cluster_compare_side_by_side.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # Create overlay comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    if len(gdf_kmeans) > 0:
        gdf_kmeans.plot(ax=ax, color="red", markersize=30, alpha=0.6, label="K-Means", edgecolor="black", linewidth=0.5)
    if len(gdf_dbscan) > 0:
        gdf_dbscan.plot(ax=ax, color="blue", markersize=30, alpha=0.6, label="DBSCAN", edgecolor="black", linewidth=0.5)
    if len(gdf_kde) > 0:
        gdf_kde.plot(ax=ax, color="green", alpha=0.4, label="KDE", edgecolor="black", linewidth=1)
    
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_title("Cluster Comparison: Overlay", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    overlay_path = os.path.join(out_dir, "plots", "cluster_compare_overlay.png")
    plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {overlay_path}")


def main() -> None:
    """Compare clustering results from K-Means, DBSCAN, and KDE methods.
    
    Loads hotspots from all three clustering methods, computes Jaccard
    overlap scores between method pairs, generates comparison visualizations,
    and saves results to JSON.
    """
    parser = argparse.ArgumentParser(
        description="Compare clustering results from all three methods"
    )
    parser.add_argument(
        "--in-dir",
        default="eda_out",
        help="Input directory containing hotspot JSON files (default: eda_out)"
    )
    parser.add_argument(
        "--out",
        default="eda_out",
        help="Output directory (default: eda_out)"
    )
    parser.add_argument(
        "--topN",
        type=int,
        default=None,
        help="Filter to top N hotspots by weight (default: None = use all)"
    )
    parser.add_argument(
        "--kde-polygons",
        default=None,
        help="Path to KDE polygons GeoJSON (if different from kde_hotspots.json)"
    )
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    
    # Load hotspots
    print("[INFO] Loading hotspots...")
    
    # K-Means
    kmeans_path = os.path.join(args.in_dir, "kmeans_hotspots.json")
    if os.path.exists(kmeans_path):
        gdf_kmeans = points_json_to_gdf(kmeans_path)
        print(f"  K-Means: {len(gdf_kmeans)} hotspots")
    else:
        print(f"  [WARN] K-Means hotspots not found: {kmeans_path}")
        gdf_kmeans = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    # DBSCAN
    dbscan_path = os.path.join(args.in_dir, "dbscan_hotspots.json")
    if os.path.exists(dbscan_path):
        gdf_dbscan = points_json_to_gdf(dbscan_path)
        print(f"  DBSCAN: {len(gdf_dbscan)} hotspots")
    else:
        print(f"  [WARN] DBSCAN hotspots not found: {dbscan_path}")
        gdf_dbscan = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    # KDE
    kde_path = args.kde_polygons or os.path.join(args.in_dir, "kde_hotspots.json")
    if os.path.exists(kde_path):
        if kde_path.endswith(".geojson"):
            gdf_kde = gpd.read_file(kde_path)
        else:
            gdf_kde = points_json_to_gdf(kde_path)
        print(f"  KDE: {len(gdf_kde)} hotspots")
    else:
        print(f"  [WARN] KDE hotspots not found: {kde_path}")
        gdf_kde = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    # Filter to top N if requested
    if args.topN is not None:
        if len(gdf_kmeans) > 0 and "weight" in gdf_kmeans.columns:
            gdf_kmeans = gdf_kmeans.nlargest(args.topN, "weight")
        if len(gdf_dbscan) > 0 and "weight" in gdf_dbscan.columns:
            gdf_dbscan = gdf_dbscan.nlargest(args.topN, "weight")
        if len(gdf_kde) > 0 and "weight" in gdf_kde.columns:
            gdf_kde = gdf_kde.nlargest(args.topN, "weight")
        print(f"[INFO] Filtered to top {args.topN} hotspots per method")
    
    # Buffer point-based hotspots (K-Means and DBSCAN)
    print("[INFO] Buffering point hotspots for overlap calculation...")
    
    # Determine projected CRS (use K-Means if available, otherwise auto-select)
    if len(gdf_kmeans) > 0 and "crs_proj" in gdf_kmeans.columns:
        crs_proj = gdf_kmeans["crs_proj"].iloc[0]
    elif len(gdf_dbscan) > 0 and "crs_proj" in gdf_dbscan.columns:
        crs_proj = gdf_dbscan["crs_proj"].iloc[0]
    else:
        # Auto-select based on data bounds
        all_points = []
        if len(gdf_kmeans) > 0:
            all_points.extend([(p.x, p.y) for p in gdf_kmeans.geometry])
        if len(gdf_dbscan) > 0:
            all_points.extend([(p.x, p.y) for p in gdf_dbscan.geometry])
        if len(gdf_kde) > 0:
            # Extract centroids for polygons, coordinates for points
            for geom in gdf_kde.geometry:
                if geom.geom_type == "Point":
                    all_points.append((geom.x, geom.y))
                else:
                    # Polygon - use centroid
                    centroid = geom.centroid
                    all_points.append((centroid.x, centroid.y))
        
        if all_points:
            from clustering.utils import choose_projected_crs
            lons = [p[0] for p in all_points]
            lats = [p[1] for p in all_points]
            bbox = (min(lons), min(lats), max(lons), max(lats))
            crs_proj = choose_projected_crs(bbox)
        else:
            # Get bounds from KDE polygons if available
            if len(gdf_kde) > 0:
                bounds = gdf_kde.total_bounds
                from clustering.utils import choose_projected_crs
                crs_proj = choose_projected_crs((bounds[0], bounds[1], bounds[2], bounds[3]))
            else:
                crs_proj = "EPSG:32145"  # Default to VA North
    
    # Buffer K-Means and DBSCAN points
    if len(gdf_kmeans) > 0:
        gdf_kmeans_buf = buffer_points_by_radius(gdf_kmeans, crs_proj)
    else:
        gdf_kmeans_buf = gdf_kmeans
    
    if len(gdf_dbscan) > 0:
        gdf_dbscan_buf = buffer_points_by_radius(gdf_dbscan, crs_proj)
    else:
        gdf_dbscan_buf = gdf_dbscan
    
    gdf_kde_buf = gdf_kde.copy()
    
    # Compute Jaccard overlaps
    print("[INFO] Computing Jaccard overlaps...")
    
    jaccard_km_db = jaccard_overlap(gdf_kmeans_buf, gdf_dbscan_buf)
    jaccard_km_kde = jaccard_overlap(gdf_kmeans_buf, gdf_kde_buf)
    jaccard_db_kde = jaccard_overlap(gdf_dbscan_buf, gdf_kde_buf)
    
    print(f"  K-Means vs DBSCAN: {jaccard_km_db:.4f}")
    print(f"  K-Means vs KDE: {jaccard_km_kde:.4f}")
    print(f"  DBSCAN vs KDE: {jaccard_db_kde:.4f}")
    
    # Create comparison results
    results = {
        "jaccard_kmeans_dbscan": jaccard_km_db,
        "jaccard_kmeans_kde": jaccard_km_kde,
        "jaccard_dbscan_kde": jaccard_db_kde,
        "counts": {
            "kmeans": int(len(gdf_kmeans)),
            "dbscan": int(len(gdf_dbscan)),
            "kde": int(len(gdf_kde)),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save comparison JSON
    json_path = os.path.join(args.out, "cluster_compare.json")
    print(f"[INFO] Saving comparison results to {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved comparison results")
    
    # Generate comparison maps
    print("[INFO] Generating comparison maps...")
    plot_cluster_comparison(gdf_kmeans, gdf_dbscan, gdf_kde, args.out)
    
    print("[DONE] Cluster comparison complete.")


if __name__ == "__main__":
    main()

