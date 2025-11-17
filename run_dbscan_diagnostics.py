#!/usr/bin/env python3
"""DBSCAN clustering diagnostics runner.

Computes k-distance plot, selects optimal eps using knee detection,
fits DBSCAN model, and exports hotspots.

Usage:
    # Run with defaults (no arguments needed)
    python run_dbscan_diagnostics.py
    
    # Or with custom arguments
    python run_dbscan_diagnostics.py --input data.jsonl --min-samples 8
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from clustering import DBSCANClustering, load_points_df, gdf_to_points_json, to_projected
from metrics.clustering import k_distance_plot

# Try to import kneed for automatic knee detection
try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False
    print("[WARN] kneed library not available. Using percentile-based eps selection.")


def compute_cluster_radius_p75(
    clusterer: DBSCANClustering,
    cluster_label: int,
    X_proj: np.ndarray
) -> float:
    """Compute P75 distance of cluster member points to centroid.
    
    Args:
        clusterer: Fitted DBSCANClustering model.
        cluster_label: Cluster label (>= 0, excludes noise -1).
        X_proj: Projected coordinates array (n_samples, 2).
        
    Returns:
        P75 distance in meters.
    """
    # Get points in this cluster
    mask = clusterer.labels_ == cluster_label
    cluster_points = X_proj[mask]
    
    if len(cluster_points) == 0:
        return 0.0
    
    # Get cluster centroid (mean of all points)
    centroid = cluster_points.mean(axis=0)
    
    # Compute distances from points to centroid
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    
    # Return P75 (75th percentile)
    return float(np.percentile(distances, 75))


def find_eps_knee(indices: np.ndarray, sorted_distances: np.ndarray) -> float:
    """Find optimal eps using automatic knee detection or percentile method.
    
    Args:
        indices: Point indices (sorted).
        sorted_distances: Sorted k-distance values.
        
    Returns:
        Optimal eps value in meters.
    """
    if HAS_KNEED:
        # Use kneed library for automatic knee detection
        try:
            kl = KneeLocator(
                indices,
                sorted_distances,
                curve='convex',
                direction='increasing',
                online=True
            )
            if kl.knee is not None:
                return float(sorted_distances[kl.knee])
        except Exception as e:
            print(f"[WARN] KneeLocator failed: {e}. Falling back to percentile method.")
    
    # Fallback: Use 95th percentile
    eps = float(np.percentile(sorted_distances, 95))
    return eps


def main() -> None:
    """Run DBSCAN clustering diagnostics with k-distance plot.
    
    Computes k-distance plot, selects optimal eps parameter using knee
    detection, fits DBSCAN model, computes cluster radii, and exports
    hotspots to JSON.
    
    Raises:
        SystemExit: If data loading fails or no clusters are found.
    """
    parser = argparse.ArgumentParser(
        description="DBSCAN clustering diagnostics with k-distance plot"
    )
    parser.add_argument(
        "--input",
        default="eda_out/eda_cases_min.jsonl",
        help="Path to input data file (JSONL or CSV with lon/lat columns) (default: eda_out/eda_cases_min.jsonl)"
    )
    parser.add_argument(
        "--out",
        default="eda_out",
        help="Output directory (default: eda_out)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=6,
        help="Minimum samples for DBSCAN (default: 6)"
    )
    parser.add_argument(
        "--eps-meters",
        type=float,
        default=None,
        help="Eps value in meters (if not provided, auto-detect from k-distance plot)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of neighbors for k-distance plot (default: same as min-samples)"
    )
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    
    # Load data
    print(f"[INFO] Loading data from {args.input}...")
    try:
        df = load_points_df(args.input)
        print(f"[INFO] Loaded {len(df)} points")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)
    
    # Convert to projected coordinates for k-distance plot
    print("[INFO] Converting to projected coordinates...")
    gdf_proj, _ = to_projected(df, "lon", "lat", "EPSG:4326", None)
    X_proj = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
    
    # Compute k-distance plot
    k = args.k if args.k is not None else args.min_samples
    print(f"[INFO] Computing {k}-distance plot...")
    sorted_distances, indices = k_distance_plot(X_proj, k=k)
    
    # Generate k-distance plot
    print("[INFO] Generating k-distance plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sorted_distances)), sorted_distances, linewidth=2)
    plt.xlabel("Point Index (sorted)", fontsize=12)
    plt.ylabel(f"{k}-NN Distance (meters)", fontsize=12)
    plt.title(f"DBSCAN {k}-Distance Plot", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    kdist_path = os.path.join(args.out, "plots", "dbscan_kdist.png")
    plt.savefig(kdist_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {kdist_path}")
    
    # Select eps
    if args.eps_meters is not None:
        eps = args.eps_meters
        print(f"[INFO] Using provided eps: {eps:.2f} meters")
    else:
        print("[INFO] Auto-detecting eps from k-distance plot...")
        eps = find_eps_knee(indices, sorted_distances)
        print(f"  Selected eps: {eps:.2f} meters")
    
    # Fit DBSCAN model
    print(f"[INFO] Fitting DBSCAN with eps={eps:.2f}m, min_samples={args.min_samples}...")
    model = DBSCANClustering(
        eps_meters=eps,
        min_samples=args.min_samples
    )
    model.fit(df, x_col="lon", y_col="lat")
    
    # Get hotspots
    hotspots_gdf = model.hotspots()
    
    if len(hotspots_gdf) == 0:
        print("[WARN] No clusters found. Try adjusting eps or min_samples.")
        sys.exit(0)
    
    print(f"[INFO] Found {len(hotspots_gdf)} clusters")
    
    # Compute radius for each cluster
    print("[INFO] Computing cluster radii...")
    radius_m_values = []
    
    # Get unique cluster labels (exclude noise -1)
    unique_labels = np.unique(model.labels_)
    cluster_labels = unique_labels[unique_labels >= 0]
    
    # Extract projected coordinates from stored gdf_proj_
    X_proj_from_gdf = np.column_stack([
        model.gdf_proj_.geometry.x,
        model.gdf_proj_.geometry.y
    ])
    
    # Match radius to hotspot order (hotspots are ordered by cluster label)
    for cluster_label in cluster_labels:
        radius_m = compute_cluster_radius_p75(
            model,
            cluster_label,
            X_proj_from_gdf
        )
        radius_m_values.append(radius_m)
    
    hotspots_gdf["radius_m"] = radius_m_values
    
    # Export to JSON
    json_path = os.path.join(args.out, "dbscan_hotspots.json")
    print(f"[INFO] Exporting hotspots to {json_path}...")
    gdf_to_points_json(hotspots_gdf, json_path)
    print(f"[OK] Exported {len(hotspots_gdf)} hotspots")
    
    print("[DONE] DBSCAN diagnostics complete.")


if __name__ == "__main__":
    main()

