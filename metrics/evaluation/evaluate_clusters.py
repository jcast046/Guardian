#!/usr/bin/env python3
"""Cluster stability evaluation script.

Evaluates cluster stability across multiple bootstrap iterations for K-Means,
DBSCAN, and KDE algorithms. Computes Jaccard overlap, Adjusted Rand Index (ARI),
silhouette scores, and Davies-Bouldin scores.

Author: Joshua Castillo

Example:
    python -m metrics.evaluation.evaluate_clusters --cases data/synthetic_cases --outdir eda_out
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_here = Path(__file__).resolve()
_proj_root = _here.parents[2]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from clustering import make_clusterer
from clustering.utils import to_projected, validate_coordinates
from metrics.clustering import (
    bootstrap_stability,
    silhouette_score,
    davies_bouldin_score,
)


def load_case_coordinates(case_files_dir: str) -> pd.DataFrame:
    """Load all case coordinates into DataFrame with lon/lat columns.
    
    Args:
        case_files_dir: Path to directory containing case JSON files.
        
    Returns:
        DataFrame with columns: lon, lat (validated coordinates only).
    """
    case_dir = Path(case_files_dir)
    if not case_dir.exists():
        raise ValueError(f"Case directory not found: {case_files_dir}")
    
    coords = []
    json_files = list(case_dir.glob("*.json"))
    
    print(f"[INFO] Loading coordinates from {len(json_files)} case files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                case = json.load(f)
            
            # Extract coordinates from nested structure
            spatial = case.get("spatial", {})
            lat = spatial.get("last_seen_lat")
            lon = spatial.get("last_seen_lon")
            
            if lat is not None and lon is not None:
                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                    coords.append({"lon": lon_f, "lat": lat_f})
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            print(f"[WARN] Failed to load {json_file.name}: {e}")
            continue
    
    if not coords:
        raise ValueError(f"No valid coordinates found in {case_files_dir}")
    
    if not coords:
        raise ValueError(f"No valid coordinates found in {case_files_dir}")
    
    df = pd.DataFrame(coords)
    
    # Validate coordinates (removes NaN and out-of-range values)
    df = validate_coordinates(df, "lon", "lat")
    
    if len(df) == 0:
        raise ValueError(f"All coordinates were filtered out after validation in {case_files_dir}")
    
    print(f"[INFO] Loaded {len(df)} valid coordinate pairs")
    return df


def get_default_clusterer_params(method: str) -> Dict[str, Any]:
    """Get default parameters for each clustering method.
    
    Args:
        method: Clustering method name ("kmeans", "dbscan", "kde").
        
    Returns:
        Dictionary of default parameters for the method.
    """
    defaults = {
        "kmeans": {
            "n_clusters": 10,
            "random_state": 42,
        },
        "dbscan": {
            "eps_meters": 5000.0,  # 5 km
            "min_samples": 3,
        },
        "kde": {
            "bandwidth_meters": 2000.0,  # 2 km
            "label_policy": "none",  # KDE uses 'none' because it focuses on density-based hotspot detection
            # via isodensity polygons rather than traditional cluster labels. With 'none', all points
            # are assigned label -1 (noise), so ARI cannot be computed (requires at least 2 clusters).
            # Jaccard overlap is the relevant stability metric for KDE as it measures consistency
            # of hotspot polygon geometries across bootstrap runs.
            "hotspot_mode": "iso_polygons",
            "iso_mass": 0.90,
            "grid_res_m": 500,
            "random_state": 42,
        },
    }
    
    if method not in defaults:
        raise ValueError(f"Unknown method: {method}. Must be one of: {list(defaults.keys())}")
    
    return defaults[method]


def evaluate_cluster_stability(
    case_coords_df: pd.DataFrame,
    methods: List[str] = ["kmeans", "dbscan", "kde"],
    n_iter: int = 10,
    sample_ratio: float = 0.85
) -> Dict[str, Any]:
    """Run bootstrap stability for each clustering method.
    
    Args:
        case_coords_df: DataFrame with lon/lat columns.
        methods: List of clustering method names to evaluate.
        n_iter: Number of bootstrap iterations.
        sample_ratio: Fraction of points to sample in each iteration.
        
    Returns:
        Dictionary with stability metrics for each method.
    """
    results = {}
    
    # Note: Clusterers expect geographic coordinates (lon/lat in degrees)
    # It handle projection internally. get projected coordinates
    # from the clusterer after fitting for metric computation.
    print(f"[INFO] Using {len(case_coords_df)} geographic coordinates (clusterer will handle projection)")
    
    for method in methods:
        try:
            print(f"\n[INFO] Evaluating {method.upper()}...")
            
            # Get default parameters
            params = get_default_clusterer_params(method)
            
            # Create clusterer
            clusterer = make_clusterer(method, **params)
            
            # Fit on full dataset first (for silhouette/DB scores)
            print(f"  Fitting on full dataset ({len(case_coords_df)} points)...")
            if len(case_coords_df) == 0:
                print(f"  [WARN] Skipping {method} - empty DataFrame")
                continue
            
            # Fit clusterer (it will handle projection internally)
            clusterer.fit(case_coords_df, x_col="lon", y_col="lat")
            labels_full = clusterer.labels()
            
            # Get projected coordinates from the clusterer for metric computation
            # The clusterer stores projected coordinates in self.X_proj_ after fitting
            if not hasattr(clusterer, 'X_proj_') or clusterer.X_proj_ is None:
                # Fallback: project ourselves if clusterer didn't store it
                gdf_proj, _ = to_projected(case_coords_df, "lon", "lat", "EPSG:4326", clusterer.crs_proj or "EPSG:32145")
                coords_proj = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
            else:
                coords_proj = clusterer.X_proj_
            
            # Compute quality metrics on full dataset
            silhouette_scores = []
            davies_bouldin_scores = []
            
            # Compute metrics multiple times for distribution (or just once)
            # For now, compute once on full dataset
            if np.sum(labels_full >= 0) >= 2 and len(np.unique(labels_full[labels_full >= 0])) >= 2:
                sil_score = silhouette_score(labels_full, coords_proj)
                db_score = davies_bouldin_score(labels_full, coords_proj)
                
                if not np.isnan(sil_score) and sil_score != -1.0:
                    silhouette_scores.append(sil_score)
                if not np.isinf(db_score) and not np.isnan(db_score):
                    davies_bouldin_scores.append(db_score)
            
            # Run bootstrap stability
            # Note: bootstrap_stability expects geographic coordinates (lon/lat in degrees)
            # It will handle projection internally via the clusterer's fit() method
            print(f"  Running bootstrap stability ({n_iter} iterations)...")
            try:
                stability_results = bootstrap_stability(
                    clusterer, case_coords_df, n_iter=n_iter, sample_ratio=sample_ratio
                )
            except Exception as e:
                print(f"  [WARN] Bootstrap stability failed for {method}: {e}")
                import traceback
                traceback.print_exc()
                # Use default empty results
                stability_results = {
                    "jaccard_overlap": 0.0,
                    "ari_scores": [],
                    "mean_ari": None,
                    "std_ari": None,
                }
            
            # Aggregate results
            results[method] = {
                "jaccard_overlap": {
                    "mean": stability_results.get("jaccard_overlap", 0.0),
                    "std": 0.0,  # Single value, no std
                },
                "ari": {
                    "mean": stability_results.get("mean_ari"),
                    "std": stability_results.get("std_ari"),
                    "scores": stability_results.get("ari_scores", []),
                },
                "silhouette": {
                    "mean": np.mean(silhouette_scores) if silhouette_scores else None,
                    "std": np.std(silhouette_scores) if silhouette_scores else None,
                    "scores": silhouette_scores,
                },
                "davies_bouldin": {
                    "mean": np.mean(davies_bouldin_scores) if davies_bouldin_scores else None,
                    "std": np.std(davies_bouldin_scores) if davies_bouldin_scores else None,
                    "scores": davies_bouldin_scores,
                },
            }
            
            # Print results with None handling
            ari_mean = results[method]['ari']['mean']
            jaccard_mean = results[method]['jaccard_overlap']['mean']
            
            if ari_mean is not None:
                print(f"  ✓ Mean ARI: {ari_mean:.3f}")
            else:
                # KDE with label_policy='none' assigns all points to -1 (noise), so no clusters exist
                # for ARI computation. This is expected - Jaccard overlap is the primary stability metric for KDE.
                if method == "kde":
                    print(f"  ✓ Mean ARI: N/A (KDE uses density-based hotspots, not cluster labels)")
                else:
                    print(f"  ✓ Mean ARI: N/A (no valid label comparisons)")
            
            if jaccard_mean is not None:
                print(f"  ✓ Mean Jaccard: {jaccard_mean:.3f}")
            else:
                print(f"  ✓ Mean Jaccard: N/A")
        
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {method}: {e}")
            import traceback
            traceback.print_exc()
            # Store error in results
            results[method] = {
                "error": str(e),
                "jaccard_overlap": {"mean": None, "std": None},
                "ari": {"mean": None, "std": None, "scores": []},
                "silhouette": {"mean": None, "std": None, "scores": []},
                "davies_bouldin": {"mean": None, "std": None, "scores": []},
            }
            continue
    
    return results


def plot_stability_results(results: Dict[str, Any], outdir: str) -> None:
    """Generate bar charts and distribution plots.
    
    Args:
        results: Results dictionary from evaluate_cluster_stability().
        outdir: Output directory for plots.
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    methods = list(results.keys())
    
    # Plot 1: Mean ARI per method (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    ari_means = []
    ari_stds = []
    labels = []
    
    for method in methods:
        ari_data = results[method]["ari"]
        if ari_data["mean"] is not None:
            ari_means.append(ari_data["mean"])
            ari_stds.append(ari_data["std"] or 0.0)
            labels.append(method.upper())
    
    if ari_means:
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, ari_means, yerr=ari_stds, capsize=5, alpha=0.7)
        ax.set_xlabel("Clustering Method")
        ax.set_ylabel("Mean Adjusted Rand Index")
        ax.set_title("Cluster Stability: Mean ARI per Method")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(plots_dir / "cluster_stability_ari.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved ARI plot: {plots_dir / 'cluster_stability_ari.png'}")
    
    # Plot 2: Jaccard overlap per method
    fig, ax = plt.subplots(figsize=(10, 6))
    jaccard_means = []
    labels_j = []
    
    for method in methods:
        jaccard_data = results[method]["jaccard_overlap"]
        if jaccard_data["mean"] is not None:
            jaccard_means.append(jaccard_data["mean"])
            labels_j.append(method.upper())
    
    if jaccard_means:
        x_pos = np.arange(len(labels_j))
        ax.bar(x_pos, jaccard_means, alpha=0.7)
        ax.set_xlabel("Clustering Method")
        ax.set_ylabel("Mean Jaccard Overlap")
        ax.set_title("Cluster Stability: Mean Jaccard Overlap per Method")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_j)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(plots_dir / "cluster_stability_jaccard.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved Jaccard plot: {plots_dir / 'cluster_stability_jaccard.png'}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate cluster stability for multiple clustering methods"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="data/synthetic_cases",
        help="Path to directory containing case JSON files (default: data/synthetic_cases)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="eda_out",
        help="Output directory for results (default: eda_out)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["kmeans", "dbscan", "kde"],
        default=["kmeans", "dbscan", "kde"],
        help="Clustering methods to evaluate (default: all)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of bootstrap iterations (default: 10)"
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.85,
        help="Fraction of points to sample in each iteration (default: 0.85)"
    )
    
    args = parser.parse_args()
    
    # Load case coordinates
    print(f"[INFO] Loading case coordinates from {args.cases}...")
    df_coords = load_case_coordinates(args.cases)
    
    # Evaluate stability
    results = evaluate_cluster_stability(
        df_coords,
        methods=args.methods,
        n_iter=args.n_iter,
        sample_ratio=args.sample_ratio
    )
    
    # Add metadata
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_cases": len(df_coords),
        "methods": results,
    }
    
    # Save JSON results
    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "cluster_stability.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    output_serializable = convert_to_json_serializable(output)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Saved results to {json_path}")
    
    # Generate plots
    plot_stability_results(results, args.outdir)
    
    print("\n[SUMMARY] Cluster Stability Evaluation Complete")
    print(f"  Methods evaluated: {', '.join(args.methods)}")
    print(f"  Cases processed: {len(df_coords)}")
    print(f"  Bootstrap iterations: {args.n_iter}")
    print(f"  Results saved to: {json_path}")


if __name__ == "__main__":
    main()

