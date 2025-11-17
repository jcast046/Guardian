#!/usr/bin/env python3
"""K-Means clustering diagnostics runner.

Performs K-sweep from kmin to kmax, computes inertia and silhouette scores,
generates plots, selects optimal K using elbow method, and exports hotspots.

Usage:
    # Run with defaults (no arguments needed)
    python run_kmeans_diagnostics.py
    
    # Or with custom arguments
    python run_kmeans_diagnostics.py --input data.jsonl --kmin 5 --kmax 20
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from clustering import KMeansClustering, load_points_df, gdf_to_points_json
from metrics.clustering import silhouette_score

# Try to import kneed for automatic elbow detection
try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False
    print("[WARN] kneed library not available. Using simple gradient-based elbow detection.")


def compute_cluster_radius_p75(
    clusterer: KMeansClustering,
    cluster_id: int,
    X_proj: np.ndarray
) -> float:
    """Compute P75 distance of cluster member points to centroid.
    
    Args:
        clusterer: Fitted KMeansClustering model.
        cluster_id: Cluster label (0 to n_clusters-1).
        X_proj: Projected coordinates array (n_samples, 2).
        
    Returns:
        P75 distance in meters.
    """
    # Get points in this cluster
    mask = clusterer.labels_ == cluster_id
    cluster_points = X_proj[mask]
    
    if len(cluster_points) == 0:
        return 0.0
    
    # Get cluster center (in projected CRS)
    center = clusterer.model.cluster_centers_[cluster_id]
    
    # Compute distances from points to center
    distances = np.linalg.norm(cluster_points - center, axis=1)
    
    # Return P75 (75th percentile)
    return float(np.percentile(distances, 75))


def find_elbow_k(ks: List[int], inertias: List[float]) -> int:
    """Find elbow K using automatic knee detection or gradient method.
    
    Args:
        ks: List of K values.
        inertias: List of inertia values.
        
    Returns:
        Optimal K value.
    """
    if HAS_KNEED:
        # Use kneed library for automatic knee detection
        try:
            kl = KneeLocator(
                ks, inertias,
                curve='convex',
                direction='decreasing',
                online=True
            )
            if kl.knee is not None:
                return int(kl.knee)
        except Exception as e:
            print(f"[WARN] KneeLocator failed: {e}. Falling back to gradient method.")
    
    # Fallback: Simple gradient-based method
    # Find K where rate of change in inertia decreases most
    if len(inertias) < 3:
        return ks[np.argmin(inertias)]
    
    # Compute first derivative (rate of change)
    gradients = np.diff(inertias)
    
    # Compute second derivative (change in rate of change)
    second_gradients = np.diff(gradients)
    
    # Find maximum second derivative (biggest decrease in rate of change)
    # This indicates the elbow
    elbow_idx = np.argmax(second_gradients) + 1  # +1 because of diff
    
    return ks[elbow_idx]


def main() -> None:
    """Run K-Means clustering diagnostics with K-sweep.
    
    Performs K-sweep across specified range, computes inertia and silhouette
    scores, generates diagnostic plots, selects optimal K using elbow method,
    computes cluster radii, and exports hotspots to JSON.
    
    Raises:
        SystemExit: If data loading fails or K range validation fails.
    """
    parser = argparse.ArgumentParser(
        description="K-Means clustering diagnostics with K-sweep"
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
        "--kmin",
        type=int,
        default=3,
        help="Minimum K value (default: 3)"
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=30,
        help="Maximum K value (default: 30)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
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
    
    # Validate K range
    if args.kmin < 2:
        print("[ERROR] kmin must be >= 2")
        sys.exit(1)
    if args.kmax <= args.kmin:
        print("[ERROR] kmax must be > kmin")
        sys.exit(1)
    
    # Perform K-sweep
    ks = list(range(args.kmin, args.kmax + 1))
    inertias = []
    silhouette_scores = []
    models = []
    
    print(f"[INFO] Performing K-sweep from K={args.kmin} to K={args.kmax}...")
    for k in ks:
        print(f"  Fitting K={k}...", end=" ", flush=True)
        
        # Fit model
        model = KMeansClustering(
            n_clusters=k,
            random_state=args.random_state
        )
        model.fit(df, x_col="lon", y_col="lat")
        
        # Compute metrics
        inertia = model.model.inertia_
        sil_score = silhouette_score(model.labels_, model.X_proj_)
        
        inertias.append(inertia)
        silhouette_scores.append(sil_score)
        models.append(model)
        
        print(f"inertia={inertia:.2f}, silhouette={sil_score:.4f}")
    
    # Generate plots
    print("[INFO] Generating plots...")
    
    # Elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks, inertias, marker="o", linestyle="-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Inertia (Within-cluster sum of squares)", fontsize=12)
    plt.title("K-Means Elbow Plot", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    elbow_path = os.path.join(args.out, "plots", "kmeans_elbow.png")
    plt.savefig(elbow_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {elbow_path}")
    
    # Silhouette plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks, silhouette_scores, marker="o", linestyle="-", linewidth=2, markersize=8, color="green")
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.title("K-Means Silhouette Score", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sil_path = os.path.join(args.out, "plots", "kmeans_silhouette.png")
    plt.savefig(sil_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {sil_path}")
    
    # Choose best K
    print("[INFO] Selecting optimal K...")
    
    # Find elbow K
    elbow_k = find_elbow_k(ks, inertias)
    print(f"  Elbow K: {elbow_k}")
    
    # Find K with maximum silhouette (as tie-breaker)
    max_sil_idx = np.argmax(silhouette_scores)
    max_sil_k = ks[max_sil_idx]
    max_sil_score = silhouette_scores[max_sil_idx]
    print(f"  Max silhouette K: {max_sil_k} (score={max_sil_score:.4f})")
    
    if abs(max_sil_k - elbow_k) <= 2:
        best_k = elbow_k
        best_model = models[ks.index(best_k)]
        print(f"  Selected K: {best_k} (elbow method)")
    else:
        # Check if max silhouette is significantly better
        elbow_sil = silhouette_scores[ks.index(elbow_k)]
        if max_sil_score - elbow_sil > 0.05:
            best_k = max_sil_k
            best_model = models[max_sil_idx]
            print(f"  Selected K: {best_k} (silhouette method, score={max_sil_score:.4f})")
        else:
            best_k = elbow_k
            best_model = models[ks.index(best_k)]
            print(f"  Selected K: {best_k} (elbow method)")
    
    # Compute radius for each cluster
    print("[INFO] Computing cluster radii...")
    hotspots_gdf = best_model.hotspots()
    
    # Add radius_m column (match order of hotspots)
    radius_m_values = []
    n_clusters = len(best_model.model.cluster_centers_)
    for cluster_id in range(n_clusters):
        radius_m = compute_cluster_radius_p75(
            best_model,
            cluster_id,
            best_model.X_proj_
        )
        radius_m_values.append(radius_m)
    
    # Ensure order matches (hotspots are ordered by cluster ID 0..n_clusters-1)
    hotspots_gdf["radius_m"] = radius_m_values
    
    # Export to JSON
    json_path = os.path.join(args.out, "kmeans_hotspots.json")
    print(f"[INFO] Exporting hotspots to {json_path}...")
    gdf_to_points_json(hotspots_gdf, json_path)
    print(f"[OK] Exported {len(hotspots_gdf)} hotspots")
    
    print("[DONE] K-Means diagnostics complete.")


if __name__ == "__main__":
    main()

