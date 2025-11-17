"""Visualization tools for mobility forecasting results.

Provides functions to visualize probability distributions from mobility
forecasting as maps with optional boundary overlays.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_here = Path(__file__).resolve()
_proj_root = _here.parents[1]
if str(_proj_root) not in sys.path:
	sys.path.insert(0, str(_proj_root))

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
import json

# Import forecasting modules (after path setup)
from reinforcement_learning.forecast_api import forecast_timeline
from reinforcement_learning.build_rl_zones import load_grid_and_layers

try:
	import geopandas as gpd
	from shapely.geometry import Point
	HAS_GEO = True
except ImportError:
	HAS_GEO = False


def load_boundary(boundary_path: str = "data/geo/va_boundary.geojson"):
	"""Load Virginia boundary polygon for overlay.
	
	Args:
		boundary_path: Path to boundary GeoJSON file.
		
	Returns:
		GeoDataFrame with boundary geometry, or None if not available.
	"""
	if not HAS_GEO:
		return None
	
	try:
		gdf = gpd.read_file(boundary_path)
		if gdf.crs is None:
			gdf = gdf.set_crs(4326)
		return gdf
	except Exception as e:
		print(f"[WARN] Could not load boundary: {e}")
		return None


def plot_risk_map(
	grid_xy: np.ndarray,
	p: np.ndarray,
	outpath: str,
	title: str = "",
	boundary_gdf=None,
	vmin: Optional[float] = None,
	vmax: Optional[float] = None,
):
	"""Plot probability distribution as a scatter map.
	
	Creates a scatter plot of grid points colored by probability, optionally
	overlaying Virginia boundary.
	
	Args:
		grid_xy: Array of shape (N, 2) with (longitude, latitude) coordinates.
		p: Probability distribution array of shape (N,) summing to 1.0.
		outpath: Output file path for PNG image.
		title: Title text for the plot.
		boundary_gdf: Optional GeoDataFrame with boundary geometry.
		vmin: Minimum value for color scale (default: None, uses data min).
		vmax: Maximum value for color scale (default: None, uses data max).
	"""
	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Plot boundary if available
	if boundary_gdf is not None and HAS_GEO:
		try:
			boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="#2b8cbe",
			                 linewidth=1.5, alpha=0.8, zorder=0)
		except Exception:
			pass
	
	# Plot grid points colored by probability using hexbin for better visualization
	lon = grid_xy[:, 0]
	lat = grid_xy[:, 1]
	
	# Use hexbin plot for smoother visualization 
	hexbin = ax.hexbin(lon, lat, C=p, gridsize=120, cmap="YlOrRd", 
	                   bins='log', alpha=0.8, mincnt=1, vmin=vmin, vmax=vmax, zorder=1)
	
	# Add colorbar
	cbar = plt.colorbar(hexbin, ax=ax)
	cbar.set_label("Probability", rotation=270, labelpad=20)
	
	# Set labels and title
	ax.set_xlabel("Longitude")
	ax.set_ylabel("Latitude")
	if title:
		ax.set_title(title)
	else:
		ax.set_title("Risk Map")
	
	# Set aspect ratio
	ax.set_aspect("equal", adjustable="box")
	
	# Save figure
	plt.tight_layout()
	plt.savefig(outpath, dpi=150, bbox_inches="tight")
	plt.close()


def visualize_forecast_timeline(
	case_path: Optional[str] = None,
	case: Optional[Dict] = None,
	horizons: Tuple[int, ...] = (24, 48, 72),
	output_dir: str = "eda_out/forecast_plots",
	**kwargs
):
	"""Visualize forecast timeline for a case.
	
	Loads a case (from file path or dict), computes forecasts for multiple horizons, 
	and saves PNG images for each horizon. Can be used programmatically without CLI.
	
	Args:
		case_path: Optional path to case JSON file. If None, case dict must be provided.
		case: Optional case dictionary. If None, case_path must be provided.
		horizons: Tuple of time horizons in hours (default: (24, 48, 72)).
		output_dir: Output directory for PNG files.
		**kwargs: Additional arguments passed to forecast_timeline().
		
	Raises:
		ValueError: If neither case_path nor case is provided.
	"""
	# Load case from file or use provided dict
	if case is None:
		if case_path is None:
			raise ValueError("Either case_path or case dict must be provided")
		with open(case_path, "r", encoding="utf-8") as f:
			case = json.load(f)
	elif case_path is not None:
		# If both provided, prefer the dict
		pass
	
	case_id = case.get("case_id", "unknown")
	
	# Load grid and boundary
	grid_xy, _, _, _, _, _ = load_grid_and_layers()
	boundary_gdf = load_boundary()
	
	# Compute forecasts
	forecasts = forecast_timeline(case, horizons=horizons, **kwargs)
	
	# Create output directory
	Path(output_dir).mkdir(parents=True, exist_ok=True)
	
	# Plot each horizon
	for horizon, p in forecasts.items():
		title = f"{case_id} - Forecast at {horizon}h"
		outpath = Path(output_dir) / f"{case_id}_t{horizon}.png"
		
		plot_risk_map(
			grid_xy=grid_xy,
			p=p,
			outpath=str(outpath),
			title=title,
			boundary_gdf=boundary_gdf,
		)
		
		print(f"Saved forecast plot: {outpath}")
	
	# Generate cumulative map if requested
	cumulative = kwargs.get("cumulative", False)
	if cumulative:
		from reinforcement_learning.forecast_api import forecast_cumulative
		
		cumulative_weights = kwargs.get("cumulative_weights", None)
		cumulative_mode = kwargs.get("cumulative_mode", "avg")
		
		p_cum = forecast_cumulative(
			case, horizons=horizons,
			weights=cumulative_weights,
			mode=cumulative_mode,
			**{k: v for k, v in kwargs.items() 
			   if k not in ["cumulative", "cumulative_weights", "cumulative_mode"]}
		)
		
		# Determine horizon range string for title
		horizon_str = f"{min(horizons)}-{max(horizons)}h"
		title_cum = f"{case_id} - Cumulative Forecast ({horizon_str})"
		outpath_cum = Path(output_dir) / f"{case_id}_cumulative_{horizon_str.replace('-', '_')}.png"
		
		plot_risk_map(
			grid_xy=grid_xy,
			p=p_cum,
			outpath=str(outpath_cum),
			title=title_cum,
			boundary_gdf=boundary_gdf,
		)
		
		print(f"Saved cumulative forecast plot: {outpath_cum}")


# CLI entry point (optional - module can be used programmatically without CLI)
if __name__ == "__main__":
	import argparse
	import glob
	import os
	
	parser = argparse.ArgumentParser(description="Visualize mobility forecasts")
	parser.add_argument("--case", type=str, default=None,
	                   help="Path to case JSON file (if not provided, uses first available case)")
	parser.add_argument("--horizons", type=int, nargs="+", default=[24, 48, 72],
	                   help="Time horizons in hours")
	parser.add_argument("--output-dir", type=str, default="eda_out/forecast_plots",
	                   help="Output directory for plots")
	parser.add_argument("--alpha-prior", type=float, default=0.5,
	                   help="Mixing weight for KDE prior")
	parser.add_argument("--steps-per-24h", type=int, default=3,
	                   help="Markov steps per 24 hours")
	parser.add_argument("--cumulative", action="store_true",
	                   help="Generate cumulative map (0-72h combined)")
	parser.add_argument("--cumulative-weights", type=float, nargs="+", default=None,
	                   help="Weights for cumulative map (default: [0.5, 0.3, 0.2] for 3 horizons)")
	parser.add_argument("--cumulative-mode", type=str, default="avg",
	                   choices=["avg", "max"],
	                   help="Cumulative combination mode: avg (weighted average) or max (element-wise maximum)")
	
	args = parser.parse_args()
	
	# If no case provided, try to find one automatically
	case_path = args.case
	if case_path is None:
		# Look for case files in common locations
		possible_paths = [
			"data/synthetic_cases/*.json",
			"data/cases/*.json",
			"*.json"
		]
		
		for pattern in possible_paths:
			matches = glob.glob(pattern)
			if matches:
				case_path = matches[0]
				print(f"[INFO] Using first available case: {case_path}")
				break
		
		if case_path is None:
			print("[ERROR] No case file provided and none found automatically.")
			print("Usage: python visualize_forecast.py --case <path_to_case.json>")
			print("Or place a case JSON file in data/synthetic_cases/ or data/cases/")
			exit(1)
	
	# Check if case file exists
	if not os.path.exists(case_path):
		print(f"[ERROR] Case file not found: {case_path}")
		exit(1)
	
	visualize_forecast_timeline(
		case_path=case_path,
		horizons=tuple(args.horizons),
		output_dir=args.output_dir,
		alpha_prior=args.alpha_prior,
		steps_per_24h=args.steps_per_24h,
		cumulative=args.cumulative,
		cumulative_weights=args.cumulative_weights,
		cumulative_mode=args.cumulative_mode,
	)

