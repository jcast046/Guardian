#!/usr/bin/env python3
"""CLI wrapper for search plan generation.

Generates complete search plans with sectors, hotspots, containment rings,
and visualization maps for SAR teams.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
# Works whether running from project root or reinforcement_learning directory
_here = Path(__file__).resolve()
# File is in reinforcement_learning/, so project root is one level up
_proj_root = _here.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import argparse
import json
import pandas as pd
from typing import Dict, Any

# Import from reinforcement_learning module
try:
    from reinforcement_learning.forecast_api import forecast_search_plan
    from reinforcement_learning.visualize_forecast import plot_search_plan
except ImportError:
    # Fallback: if running directly from reinforcement_learning directory
    # Import directly from local modules
    from forecast_api import forecast_search_plan
    from visualize_forecast import plot_search_plan


def serialize_search_plan(search_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Convert search plan to JSON-serializable format (omit geometry).
    
    Args:
        search_plan: Output from forecast_search_plan().
        
    Returns:
        Dictionary with serializable data (no GeoDataFrame geometry).
    """
    serializable = {
        "grid_xy": search_plan["grid_xy"].tolist() if hasattr(search_plan["grid_xy"], "tolist") else search_plan["grid_xy"],
        "p": search_plan["p"].tolist() if hasattr(search_plan["p"], "tolist") else search_plan["p"],
        "sector_idx": search_plan["sector_idx"].tolist() if hasattr(search_plan["sector_idx"], "tolist") else search_plan["sector_idx"],
        "sectors_ranked": search_plan["sectors_ranked"],
        "sectors_ranked_by_horizon": {
            str(h): sectors_ranked_list
            for h, sectors_ranked_list in search_plan.get("sectors_ranked_by_horizon", {}).items()
        },
        "sector_hotspots": search_plan["sector_hotspots"],
        "sector_hotspots_by_horizon": {
            str(h): hotspots_list
            for h, hotspots_list in search_plan.get("sector_hotspots_by_horizon", {}).items()
        } if "sector_hotspots_by_horizon" in search_plan else {},
        "rings": search_plan.get("rings", []),
        "rings_by_horizon": {
            str(h): rings_list
            for h, rings_list in search_plan.get("rings_by_horizon", {}).items()
        } if "rings_by_horizon" in search_plan else {},
        "ipp": search_plan.get("ipp"),
        "sectors_metadata": search_plan.get("sectors_metadata", []),
        "sector_ids": search_plan.get("sector_ids", []),
        # Note: forecasts_by_horizon excluded from JSON due to large array size
    }
    
    return serializable


def export_sectors_csv(
    sectors_ranked: list,
    outpath: str
):
    """Export sector table to CSV.
    
    Args:
        sectors_ranked: List of ranked sector dictionaries.
        outpath: Output CSV file path.
    """
    # Create DataFrame from sectors_ranked
    df = pd.DataFrame.from_records(sectors_ranked)
    
    # Add rank column (1-based index)
    df.insert(0, "rank", range(1, len(df) + 1))
    
    # Select columns for CSV
    csv_columns = ["rank", "sector_id", "name", "region_tag", "mass", "mass_pct"]
    available_columns = [col for col in csv_columns if col in df.columns]
    
    df[available_columns].to_csv(outpath, index=False)


def export_sectors_by_horizon_csv(
    sectors_ranked_by_horizon: Dict[int, list],
    sectors_gdf,
    horizons: list,
    outpath: str
):
    """Export sector rankings for each horizon side-by-side to CSV.
    
    Args:
        sectors_ranked_by_horizon: Dictionary mapping horizon hours to ranked sector lists.
        sectors_gdf: GeoDataFrame of sectors (for metadata lookup).
        horizons: List of horizon hours (e.g., [24, 48, 72]).
        outpath: Output CSV file path.
    """
    # Get all unique sector IDs across all horizons
    all_sector_ids = set()
    for horizon in horizons:
        if horizon in sectors_ranked_by_horizon:
            for sector in sectors_ranked_by_horizon[horizon]:
                all_sector_ids.add(sector["sector_id"])
    
    # Create a mapping of sector_id to metadata (name, region_tag)
    # Use sectors_gdf if available, otherwise use first occurrence in ranked lists
    sector_metadata = {}
    
    # Try to get metadata from sectors_gdf first
    if hasattr(sectors_gdf, 'iloc'):
        for idx in range(len(sectors_gdf)):
            sector_row = sectors_gdf.iloc[idx]
            sector_id = str(sector_row.get("sector_id", f"R{idx}"))
            if sector_id in all_sector_ids and sector_id not in sector_metadata:
                sector_metadata[sector_id] = {
                    "name": str(sector_row.get("region", sector_row.get("name", ""))),
                    "region_tag": str(sector_row.get("region_tag", "")),
                }
    
    # Fill in any missing metadata from ranked lists
    for horizon in horizons:
        if horizon in sectors_ranked_by_horizon:
            for sector in sectors_ranked_by_horizon[horizon]:
                sector_id = sector["sector_id"]
                if sector_id not in sector_metadata:
                    sector_metadata[sector_id] = {
                        "name": sector.get("name", ""),
                        "region_tag": sector.get("region_tag", ""),
                    }
    
    # Build rows for DataFrame
    rows = []
    sorted_sector_ids = sorted(all_sector_ids)
    
    for sector_id in sorted_sector_ids:
        row = {
            "sector_id": sector_id,
            "name": sector_metadata.get(sector_id, {}).get("name", ""),
            "region_tag": sector_metadata.get(sector_id, {}).get("region_tag", ""),
        }
        
        # Add rank and mass_pct for each horizon
        for horizon in sorted(horizons):
            rank_key = f"rank_{horizon}h"
            mass_pct_key = f"mass_pct_{horizon}h"
            
            # Find this sector in the horizon's ranked list
            rank = None
            mass_pct = 0.0
            
            if horizon in sectors_ranked_by_horizon:
                for rank_idx, sector in enumerate(sectors_ranked_by_horizon[horizon], start=1):
                    if sector["sector_id"] == sector_id:
                        rank = rank_idx
                        mass_pct = sector.get("mass_pct", 0.0)
                        break
            
            row[rank_key] = rank
            row[mass_pct_key] = mass_pct
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Export to CSV
    df.to_csv(outpath, index=False)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate search plan with sectors, hotspots, and containment rings"
    )
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        help="Path to case JSON file"
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[24, 48, 72],
        help="Time horizons in hours (default: 24 48 72)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="eda_out/forecast_plots",
        help="Output directory (default: eda_out/forecast_plots)"
    )
    parser.add_argument(
        "--use-cumulative",
        action="store_true",
        default=True,
        help="Use cumulative forecast across horizons (default: True)"
    )
    parser.add_argument(
        "--no-cumulative",
        dest="use_cumulative",
        action="store_false",
        help="Don't use cumulative forecast"
    )
    parser.add_argument(
        "--hotspot-pct",
        type=float,
        default=0.9,
        help="Percentile for sector hotspots (default: 0.9)"
    )
    parser.add_argument(
        "--sector-path",
        type=str,
        default=None,
        help="Path to sectors GeoJSON (default: data/geo/va_rl_regions.geojson relative to project root)"
    )
    parser.add_argument(
        "--alpha-prior",
        type=float,
        default=0.5,
        help="Mixing weight for KDE prior (default: 0.5)"
    )
    parser.add_argument(
        "--steps-per-24h",
        type=int,
        default=3,
        help="Markov steps per 24 hours (default: 3)"
    )
    parser.add_argument(
        "--beta-corr-day",
        type=float,
        default=0.3,
        help="Corridor bias coefficient for day (default: 0.3)"
    )
    parser.add_argument(
        "--beta-corr-night",
        type=float,
        default=0.1,
        help="Corridor bias coefficient for night (default: 0.1)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "runaway", "abduction"],
        help="Survival profile type (default: default)"
    )
    parser.add_argument(
        "--max-hotspots-per-sector",
        type=int,
        default=None,
        help="Maximum hotspots per sector (default: None = no limit)"
    )
    
    args = parser.parse_args()
    
    # Load case - resolve relative paths relative to project root
    case_path = Path(args.case)
    if not case_path.is_absolute():
        # Try relative to project root first
        case_path_from_root = _proj_root / case_path
        if case_path_from_root.exists():
            case_path = case_path_from_root
        elif not case_path.exists():
            # Try relative to current working directory
            case_path_cwd = Path.cwd() / case_path
            if case_path_cwd.exists():
                case_path = case_path_cwd
            else:
                print(f"[ERROR] Case file not found: {args.case}")
                print(f"  Tried: {case_path_from_root}")
                print(f"  Tried: {case_path_cwd}")
                sys.exit(1)
    
    if not case_path.exists():
        print(f"[ERROR] Case file not found: {case_path}")
        sys.exit(1)
    
    with open(case_path, "r", encoding="utf-8") as f:
        case = json.load(f)
    
    case_id = case.get("case_id", case_path.stem)
    
    # Create output directory - resolve relative to project root if needed
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = _proj_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Resolve sector path relative to project root if not absolute
    sector_path = args.sector_path
    if sector_path is None:
        sector_path = str(_proj_root / "data/geo/va_rl_regions.geojson")
    else:
        sector_path_obj = Path(sector_path)
        if not sector_path_obj.is_absolute():
            sector_path_from_root = _proj_root / sector_path_obj
            if sector_path_from_root.exists():
                sector_path = str(sector_path_from_root)
            else:
                sector_path = str(sector_path_obj)
    
    print(f"[INFO] Generating search plan for case: {case_id}")
    print(f"[INFO] Horizons: {args.horizons} hours")
    print(f"[INFO] Use cumulative: {args.use_cumulative}")
    print(f"[INFO] Sector path: {sector_path}")
    
    # Generate search plan
    try:
        search_plan = forecast_search_plan(
            case=case,
            horizons=tuple(args.horizons),
            use_cumulative=args.use_cumulative,
            sector_path=sector_path,
            hotspot_pct=args.hotspot_pct,
            alpha_prior=args.alpha_prior,
            steps_per_24h=args.steps_per_24h,
            beta_corr_day=args.beta_corr_day,
            beta_corr_night=args.beta_corr_night,
            profile=args.profile,
            max_hotspots_per_sector=args.max_hotspots_per_sector,
        )
    except Exception as e:
        print(f"[ERROR] Failed to generate search plan: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generate main visualization (cumulative or single-horizon)
    print("[INFO] Generating main visualization map...")
    viz_path = outdir / f"{case_id}_search_plan.png"
    try:
        plot_search_plan(
            case=case,
            search_plan=search_plan,
            outpath=str(viz_path),
            title_suffix=f" (0–{max(args.horizons)}h Search Plan)",
        )
        print(f"[OK] Saved visualization: {viz_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate per-horizon visualizations (if multiple horizons and data available)
    if len(args.horizons) > 1 and "forecasts_by_horizon" in search_plan:
        print("[INFO] Generating per-horizon visualization maps...")
        for horizon in sorted(args.horizons):
            if horizon not in search_plan.get("forecasts_by_horizon", {}):
                continue
            
            try:
                # Create a modified search_plan dict for this horizon
                horizon_search_plan = {
                    "grid_xy": search_plan["grid_xy"],
                    "p": search_plan["forecasts_by_horizon"][horizon],
                    "sectors_gdf": search_plan["sectors_gdf"],
                    "sector_idx": search_plan["sector_idx"],
                    "sectors_ranked": search_plan["sectors_ranked_by_horizon"].get(horizon, []),
                    "sector_hotspots": search_plan.get("sector_hotspots_by_horizon", {}).get(horizon, []),
                    "rings": search_plan.get("rings_by_horizon", {}).get(horizon, []),
                    "ipp": search_plan.get("ipp"),
                }
                
                viz_path_h = outdir / f"{case_id}_search_plan_t{horizon}h.png"
                plot_search_plan(
                    case=case,
                    search_plan=horizon_search_plan,
                    outpath=str(viz_path_h),
                    title_suffix=f" ({horizon}h Forecast)",
                )
                print(f"[OK] Saved {horizon}h visualization: {viz_path_h}")
            except Exception as e:
                print(f"[ERROR] Failed to generate {horizon}h visualization: {e}")
                import traceback
                traceback.print_exc()
    
    # Export JSON (serializable format)
    print("[INFO] Exporting search plan JSON...")
    json_path = outdir / f"{case_id}_search_plan.json"
    try:
        serializable_plan = serialize_search_plan(search_plan)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_plan, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved search plan JSON: {json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to export JSON: {e}")
        import traceback
        traceback.print_exc()
    
    # Export sectors CSV
    print("[INFO] Exporting sectors CSV...")
    csv_path = outdir / f"{case_id}_search_plan_sectors.csv"
    try:
        export_sectors_csv(search_plan["sectors_ranked"], str(csv_path))
        print(f"[OK] Saved sectors CSV: {csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to export CSV: {e}")
        import traceback
        traceback.print_exc()
    
    # Export per-horizon sectors CSV (if multiple horizons)
    if len(args.horizons) > 1 and "sectors_ranked_by_horizon" in search_plan:
        print("[INFO] Exporting per-horizon sectors CSV...")
        csv_by_horizon_path = outdir / f"{case_id}_search_plan_sectors_by_horizon.csv"
        try:
            export_sectors_by_horizon_csv(
                search_plan["sectors_ranked_by_horizon"],
                search_plan["sectors_gdf"],
                sorted(args.horizons),
                str(csv_by_horizon_path)
            )
            print(f"[OK] Saved per-horizon sectors CSV: {csv_by_horizon_path}")
        except Exception as e:
            print(f"[ERROR] Failed to export per-horizon CSV: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n[SUMMARY] Search Plan Generation Complete:")
    print(f"  Case ID: {case_id}")
    print(f"  Top sectors: {len(search_plan['sectors_ranked'])}")
    if search_plan["sectors_ranked"]:
        top_sector = search_plan["sectors_ranked"][0]
        print(f"  Top sector: {top_sector['sector_id']} ({100*top_sector['mass_pct']:.1f}%)")
    print(f"  Rings: {len(search_plan.get('rings', []))}")
    print(f"  Output files:")
    print(f"    - {viz_path}")
    if len(args.horizons) > 1 and "forecasts_by_horizon" in search_plan:
        for horizon in sorted(args.horizons):
            if horizon in search_plan.get("forecasts_by_horizon", {}):
                viz_path_h = outdir / f"{case_id}_search_plan_t{horizon}h.png"
                print(f"    - {viz_path_h}")
    print(f"    - {json_path}")
    print(f"    - {csv_path}")
    if len(args.horizons) > 1 and "sectors_ranked_by_horizon" in search_plan:
        csv_by_horizon_path = outdir / f"{case_id}_search_plan_sectors_by_horizon.csv"
        print(f"    - {csv_by_horizon_path}")


if __name__ == "__main__":
    main()

