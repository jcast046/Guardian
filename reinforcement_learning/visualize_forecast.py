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


def miles_to_deg(miles: float, lat: float) -> float:
	"""Convert miles to degrees longitude/latitude (approximate).
	
	Args:
		miles: Distance in miles.
		lat: Reference latitude for conversion.
		
	Returns:
		Distance in degrees (approximate).
	"""
	# Approximate conversion factors
	lat_to_miles = 69.0  # 1 degree latitude ≈ 69 miles
	lon_to_miles = 54.6 * np.cos(np.radians(lat))  # Longitude varies with latitude
	
	# Use average of lat/lon conversion for radius
	deg_per_mile = 1.0 / ((lat_to_miles + lon_to_miles) / 2.0)
	
	return miles * deg_per_mile


def plot_search_plan(
	case: Dict,
	search_plan: Dict,
	outpath: str,
	title_suffix: str = " (0–72h Search Plan)",
	top_k_labels: int = 5,
):
	"""Generate a SAR-ready search plan visualization map.
	
	Map includes:
	- Hexbin probability distribution (base layer)
	- Sector boundaries (polygon outlines)
	- Sector labels with probability percentages
	- Sector hotspots (marked with X markers)
	- Probability containment rings (50%, 75%, 90%)
	- IPP marker (last seen location)
	
	Args:
		case: Case dictionary.
		search_plan: Output from forecast_search_plan().
		outpath: Output file path for PNG.
		title_suffix: Title suffix for plot.
		top_k_labels: Number of top sectors to label (default: 5).
	"""
	fig, ax = plt.subplots(figsize=(14, 10))
	
	grid_xy = search_plan["grid_xy"]
	p = search_plan["p"]
	sectors_gdf = search_plan["sectors_gdf"]
	sectors_ranked = search_plan["sectors_ranked"]
	sector_hotspots_list = search_plan.get("sector_hotspots", [])
	rings = search_plan.get("rings", [])
	ipp = search_plan.get("ipp", None)
	
	lon = grid_xy[:, 0]
	lat = grid_xy[:, 1]
	
	# 1) Base hexbin layer
	hexbin = ax.hexbin(lon, lat, C=p, gridsize=120, cmap="YlOrRd",
	                  bins='log', alpha=0.8, mincnt=1, zorder=1)
	
	# Add colorbar
	cbar = plt.colorbar(hexbin, ax=ax)
	cbar.set_label("Probability", rotation=270, labelpad=20)
	
	# 1.5) Add Virginia state boundary outline
	if HAS_GEO:
		try:
			# Resolve boundary path relative to project root
			boundary_path = _proj_root / "data/geo/va_boundary.geojson"
			va_boundary = load_boundary(str(boundary_path))
			if va_boundary is not None:
				# Ensure consistent CRS
				if va_boundary.crs is None:
					va_boundary = va_boundary.set_crs("EPSG:4326")
				elif va_boundary.crs.to_string() != "EPSG:4326":
					va_boundary = va_boundary.to_crs("EPSG:4326")
				
				# Plot boundary outline (behind sector polygons, on top of hexbin)
				va_boundary.boundary.plot(ax=ax, color="black", linewidth=1.5, 
				                         alpha=0.8, zorder=2)
		except Exception as e:
			print(f"[WARN] Could not load VA boundary: {e}")
	
	# 2) Sector boundaries
	if HAS_GEO and sectors_gdf is not None:
		try:
			sectors_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black",
			                         alpha=0.6, zorder=3)
		except Exception:
			pass
	
	# 3) Sector annotations with label clutter reduction
	if sectors_ranked and HAS_GEO:
		for rank, sector in enumerate(sectors_ranked[:top_k_labels]):
			sector_id = sector["sector_id"]
			
			# Find sector geometry in sectors_gdf
			try:
				sector_mask = sectors_gdf["sector_id"] == sector_id
				if sector_mask.any():
					geom = sectors_gdf.loc[sector_mask, "geometry"].iloc[0]
					cx, cy = geom.centroid.x, geom.centroid.y
					
					# Use bold labels for top 3, lighter for others
					if rank < 3:
						fontweight = "bold"
						fontsize = 12
					else:
						fontweight = "normal"
						fontsize = 10
					
					label_text = f'{sector_id}\n{100*sector["mass_pct"]:.1f}%'
					ax.text(cx, cy, label_text, ha="center", va="center",
					       fontweight=fontweight, fontsize=fontsize,
					       bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
					                alpha=0.7, edgecolor="black", linewidth=1),
					       zorder=6)
			except Exception:
				pass
	
	# 4) Sector hotspots
	for sh in sector_hotspots_list:
		hotspots = sh.get("hotspots", [])
		if hotspots:
			xs = [h["lon"] for h in hotspots]
			ys = [h["lat"] for h in hotspots]
			ax.scatter(xs, ys, marker="x", s=30, color="black",
			          linewidths=1.5, zorder=5, label="Hotspots" if sh == sector_hotspots_list[0] else "")
	
	# 5) Probability rings
	if ipp and rings:
		for r in rings:
			radius_mi = r["radius_mi"]
			quantile = r["q"]
			
			# Convert radius to degrees
			radius_deg = miles_to_deg(radius_mi, ipp["lat"])
			
			# Create circle patch
			from matplotlib.patches import Circle
			circ = Circle(
				(ipp["lon"], ipp["lat"]),
				radius_deg,
				edgecolor="blue",
				facecolor="none",
				linestyle="--",
				linewidth=1.5,
				alpha=0.7,
				zorder=4
			)
			ax.add_patch(circ)
			
			# Add label for quantile percentage
			label_lat = ipp["lat"] + radius_deg * 0.1  # Offset slightly
			ax.text(ipp["lon"], label_lat, f'{int(quantile*100)}%',
			       color="blue", fontsize=9, fontweight="bold",
			       bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
			               alpha=0.8, edgecolor="blue", linewidth=1),
			       zorder=6, ha="center")
	
	# 6) IPP marker
	if ipp:
		ax.scatter(ipp["lon"], ipp["lat"], marker="*", s=400, color="red",
		          edgecolor="black", linewidths=1.5, zorder=10,
		          label="IPP (Last Seen)")
	
	# 7) Final touches
	case_id = case.get("case_id", "unknown")
	ax.set_title(f'{case_id}{title_suffix}', fontsize=14, fontweight="bold")
	ax.set_xlabel("Longitude")
	ax.set_ylabel("Latitude")
	ax.set_aspect("equal", adjustable="box")
	
	# Add legend
	ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
	
	# Set axis limits based on data bounds
	if len(grid_xy) > 0:
		ax.set_xlim(lon.min() - 0.1, lon.max() + 0.1)
		ax.set_ylim(lat.min() - 0.1, lat.max() + 0.1)
	
	plt.tight_layout()
	plt.savefig(outpath, dpi=200, bbox_inches="tight")
	plt.close()


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

