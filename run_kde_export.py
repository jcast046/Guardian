#!/usr/bin/env python3
"""Export KDE hotspots using the new unified clustering interface.

Converts KDE hotspots to the new array-of-objects JSON format.

Usage:
    # Run with defaults (no arguments needed)
    python run_kde_export.py
    
    # Or with custom arguments
    python run_kde_export.py --input data.jsonl --bandwidth-meters 25000.0
"""

import argparse
import os
import sys

from clustering import KDEClustering, gdf_to_points_json, load_points_df


def main() -> None:
    """Export KDE hotspots using unified clustering interface.
    
    Loads point data, fits KDE clustering model, extracts hotspots,
    and exports them to JSON format.
    
    Raises:
        SystemExit: If data loading fails.
    """
    parser = argparse.ArgumentParser(
        description="Export KDE hotspots using unified clustering interface"
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
        "--bandwidth-meters",
        type=float,
        default=30000.0,
        help="KDE bandwidth in meters (default: 30000.0)"
    )
    parser.add_argument(
        "--iso-mass",
        type=float,
        default=0.90,
        help="Isodensity mass threshold (default: 0.90 = top 90%%)"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load data
    print(f"[INFO] Loading data from {args.input}...")
    try:
        df = load_points_df(args.input)
        print(f"[INFO] Loaded {len(df)} points")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)
    
    # Fit KDE model
    print(f"[INFO] Fitting KDE with bandwidth={args.bandwidth_meters:.0f}m, iso_mass={args.iso_mass}...")
    kde = KDEClustering(
        bandwidth_meters=args.bandwidth_meters,
        hotspot_mode="iso_polygons",
        iso_mass=args.iso_mass
    )
    kde.fit(df, x_col="lon", y_col="lat")
    
    # Get hotspots
    hotspots = kde.hotspots()
    print(f"[INFO] Found {len(hotspots)} hotspots")
    
    # Export to JSON
    json_path = os.path.join(args.out, "kde_hotspots.json")
    print(f"[INFO] Exporting hotspots to {json_path}...")
    gdf_to_points_json(hotspots, json_path)
    print(f"[OK] Exported {len(hotspots)} hotspots")
    
    print("[DONE] KDE export complete.")


if __name__ == "__main__":
    main()

