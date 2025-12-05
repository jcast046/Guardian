"""Probability containment ring computation for search plan generation.

Computes probability-based containment radii from Initial Planning Point (IPP)
to provide SAR-style containment zones (e.g., 50%, 75%, 90% probability).
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_here = Path(__file__).resolve()
_proj_root = _here.parents[1]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.geography.distance import haversine_distance


def compute_distance_miles(grid_xy: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Compute haversine distance from each grid cell to IPP.
    
    Args:
        grid_xy: Grid coordinates array (N, 2) with (lon, lat).
        lon0: IPP longitude.
        lat0: IPP latitude.
        
    Returns:
        Array of distances in miles, shape (N,).
    """
    distances = np.zeros(len(grid_xy))
    
    for i, (lon, lat) in enumerate(grid_xy):
        distances[i] = haversine_distance(lat0, lon0, lat, lon)
    
    return distances


def probability_radii(
    grid_xy: np.ndarray,
    p: np.ndarray,
    lon0: float,
    lat0: float,
    quantiles: Tuple[float, ...] = (0.5, 0.75, 0.9)
) -> List[Dict[str, float]]:
    """Compute probability containment radii from IPP.
    
    Sorts grid cells by distance from IPP and computes cumulative probability,
    then finds radii where cumulative probability reaches each quantile.
    
    Args:
        grid_xy: Grid coordinates array (N, 2) with (lon, lat).
        p: Probability distribution array (N,).
        lon0: IPP longitude.
        lat0: IPP latitude.
        quantiles: Quantile levels to compute (default: 0.5, 0.75, 0.9).
        
    Returns:
        List of dicts with keys "q" and "radius_mi":
        [
          {"q": 0.5, "radius_mi": float},
          {"q": 0.75, "radius_mi": float},
          {"q": 0.9, "radius_mi": float},
        ]
        Sorted by quantile (ascending).
    """
    # Compute distances
    dist = compute_distance_miles(grid_xy, lon0, lat0)
    
    # Sort distance and probabilities together
    idx = np.argsort(dist)
    dist_sorted = dist[idx]
    p_sorted = p[idx]
    
    # Compute cumulative probability
    cumsum = np.cumsum(p_sorted)
    
    # Find radii for each quantile
    rings = []
    for q in quantiles:
        # Find first index where cumulative >= quantile
        idx_at_q = np.searchsorted(cumsum, q, side='left')
        
        if idx_at_q >= len(dist_sorted):
            # If quantile not reached, use max distance
            radius_mi = float(dist_sorted[-1]) if len(dist_sorted) > 0 else 0.0
        else:
            radius_mi = float(dist_sorted[idx_at_q])
        
        rings.append({
            "q": q,
            "radius_mi": radius_mi
        })
    
    return rings


def radial_profile(
    grid_xy: np.ndarray,
    p: np.ndarray,
    lon0: float,
    lat0: float,
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial probability profile (distance vs cumulative probability).
    
    Args:
        grid_xy: Grid coordinates array (N, 2) with (lon, lat).
        p: Probability distribution array (N,).
        lon0: IPP longitude.
        lat0: IPP latitude.
        n_bins: Number of distance bins for profile (default: 100).
        
    Returns:
        Tuple of (distances, cumulative_probs) arrays for plotting.
    """
    # Compute distances
    dist = compute_distance_miles(grid_xy, lon0, lat0)
    
    # Sort by distance
    idx = np.argsort(dist)
    dist_sorted = dist[idx]
    p_sorted = p[idx]
    
    # Compute cumulative probability
    cumsum = np.cumsum(p_sorted)
    
    # Bin distances
    max_dist = dist_sorted[-1] if len(dist_sorted) > 0 else 1.0
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find cumulative probability at each bin edge
    cumulative_at_bins = np.zeros(n_bins)
    for i, bin_edge in enumerate(bin_edges[1:]):  # Skip first bin edge (0)
        idx_at_edge = np.searchsorted(dist_sorted, bin_edge, side='left')
        if idx_at_edge >= len(cumsum):
            cumulative_at_bins[i] = 1.0
        else:
            cumulative_at_bins[i] = cumsum[idx_at_edge]
    
    return bin_centers, cumulative_at_bins

