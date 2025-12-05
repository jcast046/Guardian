"""Sector overlay and hotspot functions for search plan generation.

Converts grid-based probability distributions into sector-level probabilities
using Virginia regional boundaries from va_rl_regions.geojson.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_here = Path(__file__).resolve()
_proj_root = _here.parents[1]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict, Any, Optional


def load_sectors(path: str = "data/geo/va_rl_regions.geojson") -> gpd.GeoDataFrame:
    """Load sectors from GeoJSON file and ensure sector_id column.
    
    Args:
        path: Path to sectors GeoJSON file.
        
    Returns:
        GeoDataFrame with columns: sector_id (string), region, region_tag, geometry.
    """
    sectors_gdf = gpd.read_file(path)
    
    # Ensure/derive sector_id column
    if "sector_id" not in sectors_gdf.columns:
        if "region_tag" in sectors_gdf.columns:
            # Use region_tag if unique
            if sectors_gdf["region_tag"].nunique() == len(sectors_gdf):
                sectors_gdf["sector_id"] = sectors_gdf["region_tag"]
            else:
                # Generate sequential IDs
                sectors_gdf["sector_id"] = ["R" + str(i) for i in range(len(sectors_gdf))]
        else:
            # Generate sequential IDs from index
            sectors_gdf["sector_id"] = ["R" + str(i) for i in range(len(sectors_gdf))]
    
    # Ensure sector_id is string type
    sectors_gdf["sector_id"] = sectors_gdf["sector_id"].astype(str)
    
    return sectors_gdf


def assign_grid_to_sectors(grid_xy: np.ndarray, sectors_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Assign each grid point to a sector using vectorized spatial join.
    
    Uses GeoPandas spatial join for performance on large grids instead of
    per-point contains() checks.
    
    Args:
        grid_xy: Grid coordinates array (N, 2) with (lon, lat).
        sectors_gdf: GeoDataFrame of sector polygons.
        
    Returns:
        sector_idx array of shape (N,):
        - sector_idx[i] = j where j is the index in sectors_gdf
        - -1 if grid cell is outside all sectors
    """
    # Create GeoDataFrame from grid points with explicit integer index
    points_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in grid_xy],
        crs="EPSG:4326"
    )
    # Reset index to ensure integer index matching grid positions
    points_gdf = points_gdf.reset_index(drop=True)
    
    # Ensure sectors are in same CRS
    if sectors_gdf.crs is None:
        sectors_gdf.set_crs("EPSG:4326", inplace=True)
    
    # Perform spatial join (within predicate)
    # Note: If sectors overlap, a point may match multiple sectors (creates duplicate rows)
    joined = gpd.sjoin(points_gdf, sectors_gdf, how="left", predicate="within")
    
    # Map result to sector indices
    sector_idx = np.full(len(grid_xy), -1, dtype=np.int32)
    
    if "index_right" in joined.columns:
        # Handle potential duplicates from overlapping sectors
        # Group by original point index and take first match
        for orig_idx in range(len(grid_xy)):
            # Find all matches for this point index
            matches = joined[joined.index == orig_idx]
            if len(matches) > 0:
                first_match = matches.iloc[0]
                if pd.notna(first_match["index_right"]):
                    sector_idx[orig_idx] = int(first_match["index_right"])
    
    return sector_idx


def sector_probabilities(
    p: np.ndarray,
    sector_idx: np.ndarray,
    sectors_gdf: gpd.GeoDataFrame
) -> List[Dict[str, Any]]:
    """Compute probability mass and percentage for each sector.
    
    Args:
        p: Probability distribution array (N,).
        sector_idx: Sector assignments array (N,) mapping grid points to sector indices.
        sectors_gdf: GeoDataFrame of sectors.
        
    Returns:
        List of dicts with keys: sector_id, name (or region), region_tag, mass, mass_pct.
        Sorted by mass descending.
    """
    results = []
    total_mass = p.sum()
    
    if total_mass == 0:
        return results
    
    for j in range(len(sectors_gdf)):
        # Find grid points in this sector
        mask = sector_idx == j
        mass = p[mask].sum()
        mass_pct = mass / total_mass if total_mass > 0 else 0.0
        
        # Extract sector info
        sector_row = sectors_gdf.iloc[j]
        sector_info = {
            "sector_id": str(sector_row.get("sector_id", f"R{j}")),
            "name": str(sector_row.get("region", sector_row.get("name", ""))),
            "region_tag": str(sector_row.get("region_tag", "")),
            "mass": float(mass),
            "mass_pct": float(mass_pct),
        }
        results.append(sector_info)
    
    # Sort by mass descending
    results.sort(key=lambda x: x["mass"], reverse=True)
    
    return results


def rank_sectors(
    p: np.ndarray,
    sector_idx: np.ndarray,
    sectors_gdf: gpd.GeoDataFrame,
    min_mass: float = 0.01
) -> List[Dict[str, Any]]:
    """Rank sectors by probability mass, filtering low-mass sectors.
    
    Wrapper around sector_probabilities() that filters out sectors below threshold.
    
    Args:
        p: Probability distribution array (N,).
        sector_idx: Sector assignments array (N,).
        sectors_gdf: GeoDataFrame of sectors.
        min_mass: Minimum mass percentage to include (default: 0.01 = 1%).
        
    Returns:
        Filtered and sorted list of sector dictionaries (by mass descending).
    """
    all_sectors = sector_probabilities(p, sector_idx, sectors_gdf)
    
    # Filter by min_mass
    filtered = [s for s in all_sectors if s["mass_pct"] >= min_mass]
    
    return filtered


def sector_hotspots(
    grid_xy: np.ndarray,
    p: np.ndarray,
    sector_idx: np.ndarray,
    sectors_ranked: List[Dict[str, Any]],
    sectors_gdf: gpd.GeoDataFrame,
    local_pct: float = 0.9,
    max_hotspots_per_sector: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Find percentile hotspots within each ranked sector.
    
    For each sector, identifies grid cells that exceed the local percentile threshold
    within that sector's probability distribution.
    
    Args:
        grid_xy: Grid coordinates (N, 2) with (lon, lat).
        p: Probability distribution (N,).
        sector_idx: Sector assignments (N,).
        sectors_ranked: List of ranked sector dicts (must include sector_id).
        sectors_gdf: GeoDataFrame of sectors (for sector_id lookup).
        local_pct: Percentile threshold (default 0.9 = top 10%).
        max_hotspots_per_sector: Optional limit on hotspots per sector.
        
    Returns:
        List of dicts, one per sector:
        [
          {
            "sector_id": "R1",
            "hotspots": [
              {"lon": float, "lat": float, "p": float, "local_pct": float},
              ...
            ]
          },
          ...
        ]
    """
    results = []
    
    # Create mapping from sector_id to sector index in sectors_gdf
    sector_id_to_idx = {}
    for idx, row in sectors_gdf.iterrows():
        sector_id_to_idx[str(row.get("sector_id", f"R{idx}"))] = idx
    
    for sector_info in sectors_ranked:
        sector_id = sector_info["sector_id"]
        
        # Find the sector index in sectors_gdf
        if sector_id not in sector_id_to_idx:
            # Skip if sector_id not found
            results.append({
                "sector_id": sector_id,
                "hotspots": []
            })
            continue
        
        sector_j = sector_id_to_idx[sector_id]
        
        # Find grid indices in this sector
        I = np.where(sector_idx == sector_j)[0]
        
        if len(I) == 0:
            results.append({
                "sector_id": sector_id,
                "hotspots": []
            })
            continue
        
        # Get probabilities for this sector
        p_sector = p[I]
        p_sector_sum = p_sector.sum()
        
        # Edge case: skip if sector has zero probability
        if p_sector_sum == 0:
            results.append({
                "sector_id": sector_id,
                "hotspots": []
            })
            continue
        
        # Normalize to local distribution
        p_local = p_sector / p_sector_sum
        
        # Compute threshold
        threshold = np.quantile(p_local, local_pct)
        
        # Find hotspots (cells above threshold)
        hotspot_mask = p_local >= threshold
        hotspot_indices = I[hotspot_mask]
        hotspot_probs = p_local[hotspot_mask]
        
        # Create hotspot list with coordinates and probabilities
        hotspots = []
        for idx, prob in zip(hotspot_indices, hotspot_probs):
            lon, lat = grid_xy[idx]
            hotspots.append({
                "lon": float(lon),
                "lat": float(lat),
                "p": float(p[idx]),  # Original probability, not normalized
                "local_pct": float(prob)
            })
        
        # Sort by local probability descending
        hotspots.sort(key=lambda x: x["local_pct"], reverse=True)
        
        # Optionally limit to top N
        if max_hotspots_per_sector is not None:
            hotspots = hotspots[:max_hotspots_per_sector]
        
        results.append({
            "sector_id": sector_id,
            "hotspots": hotspots
        })
    
    return results

