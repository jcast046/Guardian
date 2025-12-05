"""Forecasting API for mobility prediction using Markov chain propagation.

Provides a standalone API for forecasting probability distributions over
geographic space for missing-person cases. Combines multi-source hotspots,
Markov chain movement models, and survival analysis to predict subject
location at future time horizons.

Examples:
    >>> from reinforcement_learning.forecast_api import forecast_distribution
    >>> import json
    >>> 
    >>> with open("data/synthetic_cases/GRD-2025-000001.json", "r") as f:
    ...     case = json.load(f)
    >>> 
    >>> p, top_idx = forecast_distribution(case, t_hours=24.0)
    >>> forecasts = forecast_timeline(case, horizons=(24, 48, 72))
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_here = Path(__file__).resolve()
_proj_root = _here.parents[1]
if str(_proj_root) not in sys.path:
	sys.path.insert(0, str(_proj_root))

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from reinforcement_learning.build_rl_zones import (
	load_grid_and_layers,
	MovementForecaster,
	_normalize,
	assert_seed_present,
	_seed_from_disappearance,
)
from reinforcement_learning.movement_model import (
	kde_prior,
	survival_factor,
)


def _get_time_bucket(ts_iso: str, hours_offset: float = 0.0) -> str:
	"""Determine time bucket (day/night) based on timestamp and offset.
	
	Classifies time as day (6 AM - 8 PM) or night (8 PM - 6 AM).
	
	Args:
		ts_iso: ISO format timestamp string.
		hours_offset: Hours to add to timestamp for future time buckets.
		
	Returns:
		Time bucket string: "day" or "night".
	"""
	try:
		dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
	except Exception:
		# Fallback: assume day
		return "day"
	
	# Add offset
	from datetime import timedelta
	dt_offset = dt + timedelta(hours=hours_offset)
	
	hour = dt_offset.hour
	# Day: 6 AM - 8 PM (6-20), Night: 8 PM - 6 AM (20-6)
	if 6 <= hour < 20:
		return "day"
	else:
		return "night"


def forecast_distribution(
	case: Dict[str, Any],
	t_hours: float,
	alpha_prior: float = 0.5,
	steps_per_24h: int = 3,
	method_weights: Optional[Dict[str, float]] = None,
	profile: str = "default",
	beta_corr_day: float = 0.3,
	beta_corr_night: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Forecast probability distribution over grid for specified time horizon.
	
	Given a case and time horizon, computes the probability distribution
	over grid locations using Markov chain propagation with day/night transitions
	and survival decay.
	
	Args:
		case: Case dictionary containing temporal and spatial data.
		t_hours: Time horizon in hours since last seen.
		alpha_prior: Mixing weight for KDE prior vs case-specific seed (default: 0.5).
		steps_per_24h: Number of Markov steps per 24 hours (default: 3).
		method_weights: Optional dict mapping hotspot method names to weights.
		profile: Survival profile type (default: "default").
		beta_corr_day: Corridor bias coefficient for day transitions (default: 0.3).
		beta_corr_night: Corridor bias coefficient for night transitions (default: 0.1).
		
	Returns:
		Tuple containing:
		- p: Probability distribution array of shape (N,) summing to 1.0
		- top_idx: Array of top-K grid cell indices sorted by probability
	"""
	# 1) Load grid and layers
	grid_xy, hotspots, road_cost, seclusion, va_mask, corridor_score = load_grid_and_layers(method_weights=method_weights)
	
	# 2) Build forecaster with corridor support
	forecaster = MovementForecaster(
		grid_xy, road_cost, seclusion,
		corridor_score=corridor_score,
		beta_corr_day=beta_corr_day,
		beta_corr_night=beta_corr_night
	)
	
	# 3) Extract last-seen coordinates
	try:
		lat, lon = assert_seed_present(case)
	except (KeyError, Exception) as e:
		raise ValueError(f"Case missing last_seen coordinates: {e}")
	
	# 4) Build seed distribution
	seed_dist = _seed_from_disappearance(grid_xy, lon, lat, sigma_mi=5.0)
	
	# 5) Build KDE prior
	prior = kde_prior(grid_xy, hotspots)
	
	# 6) Mix prior and seed
	p = alpha_prior * prior + (1.0 - alpha_prior) * seed_dist
	p = _normalize(p)
	
	# 7) Calculate number of Markov steps
	steps = max(1, int(steps_per_24h * (t_hours / 24.0)))
	
	# 8) Get last-seen timestamp for day/night bucket logic
	last_seen_ts = case.get("temporal", {}).get("last_seen_ts")
	if not last_seen_ts:
		# Fallback: assume day
		last_seen_ts = case.get("temporal", {}).get("reported_missing_ts")
	
	for step_idx in range(steps):
		# Calculate hours offset for this step
		hours_into_forecast = (step_idx + 1) * (t_hours / steps)
		
		# Determine time bucket
		if last_seen_ts:
			bucket = _get_time_bucket(last_seen_ts, hours_offset=hours_into_forecast)
		else:
			# Fallback: alternate day/night if no timestamp
			bucket = "day" if (step_idx % 2 == 0) else "night"
		
		# Propagate one step
		p = forecaster.step(p, bucket=bucket, steps=1)
	
	# 10) Apply survival decay
	survival = survival_factor(t_hours, profile=profile)
	p = p * survival
	p = _normalize(p)
	
	# 11) Apply boundary mask (remove probability outside VA)
	if va_mask is not None:
		p = p * va_mask
		p = p / (p.sum() + 1e-9)
	
	# 12) Get top indices
	top_idx = np.argsort(-p)[:10]  # Top 10 by default
	
	return p, top_idx


def forecast_timeline(
	case: Dict[str, Any],
	horizons: Tuple[int, ...] = (24, 48, 72),
	**kwargs
) -> Dict[int, np.ndarray]:
	"""Forecast probability distributions for multiple time horizons.
	
	Computes forecasts for multiple time horizons sequentially, reusing
	the same forecaster instance. Each horizon builds on the previous one
	rather than restarting from t=0.
	
	Args:
		case: Case dictionary containing temporal and spatial data.
		horizons: Tuple of time horizons in hours (default: (24, 48, 72)).
		**kwargs: Additional arguments passed to forecast_distribution().
		
	Returns:
		Dictionary mapping horizon hours to probability distributions:
		{24: p24, 48: p48, 72: p72}
	"""
	# Load grid and layers once
	method_weights = kwargs.get("method_weights")
	grid_xy, hotspots, road_cost, seclusion, va_mask, corridor_score = load_grid_and_layers(method_weights=method_weights)
	
	# Get beta_corr parameters
	beta_corr_day = kwargs.get("beta_corr_day", 0.3)
	beta_corr_night = kwargs.get("beta_corr_night", 0.1)
	
	# Build forecaster once with corridor support
	forecaster = MovementForecaster(
		grid_xy, road_cost, seclusion,
		corridor_score=corridor_score,
		beta_corr_day=beta_corr_day,
		beta_corr_night=beta_corr_night
	)
	
	# Extract last-seen coordinates
	try:
		lat, lon = assert_seed_present(case)
	except (KeyError, Exception) as e:
		raise ValueError(f"Case missing last_seen coordinates: {e}")
	
	# Build initial seed and prior
	alpha_prior = kwargs.get("alpha_prior", 0.5)
	seed_dist = _seed_from_disappearance(grid_xy, lon, lat, sigma_mi=5.0)
	prior = kde_prior(grid_xy, hotspots)
	p = alpha_prior * prior + (1.0 - alpha_prior) * seed_dist
	p = _normalize(p)
	
	# Get parameters
	steps_per_24h = kwargs.get("steps_per_24h", 3)
	profile = kwargs.get("profile", "default")
	last_seen_ts = case.get("temporal", {}).get("last_seen_ts")
	if not last_seen_ts:
		last_seen_ts = case.get("temporal", {}).get("reported_missing_ts")
	
	results = {}
	prev_horizon = 0
	
	# Process each horizon sequentially
	for horizon in sorted(horizons):
		# Calculate steps for this horizon segment
		horizon_delta = horizon - prev_horizon
		steps = max(1, int(steps_per_24h * (horizon_delta / 24.0)))
		
		# Propagate through Markov chain
		for step_idx in range(steps):
			hours_into_forecast = prev_horizon + (step_idx + 1) * (horizon_delta / steps)
			
			# Determine time bucket
			if last_seen_ts:
				bucket = _get_time_bucket(last_seen_ts, hours_offset=hours_into_forecast)
			else:
				bucket = "day" if (step_idx % 2 == 0) else "night"
			
			p = forecaster.step(p, bucket=bucket, steps=1)
		
		# Apply survival decay for this horizon
		survival = survival_factor(horizon, profile=profile)
		p_horizon = p * survival
		p_horizon = _normalize(p_horizon)
		
		# Apply boundary mask (remove probability outside VA)
		if va_mask is not None:
			p_horizon = p_horizon * va_mask
			p_horizon = p_horizon / (p_horizon.sum() + 1e-9)
		
		results[horizon] = p_horizon
		prev_horizon = horizon
	
	return results


def forecast_cumulative(
	case: Dict[str, Any],
	horizons: Tuple[int, ...] = (24, 48, 72),
	weights: Optional[List[float]] = None,
	mode: str = "avg",
	**kwargs
) -> np.ndarray:
	"""Forecast cumulative probability distribution across multiple time horizons.
	
	Computes a combined probability distribution representing where the subject
	is likely to be at some time within the given horizons. Useful for SAR teams
	planning searches across extended time windows.
	
	Args:
		case: Case dictionary containing temporal and spatial data.
		horizons: Tuple of time horizons in hours (default: (24, 48, 72)).
		weights: Optional list of weights for each horizon. If None, uses
			time-biased default [0.5, 0.3, 0.2] for 3 horizons, or uniform
			average if different number of horizons.
		mode: Combination mode (default: "avg"):
			- "avg": Weighted average of horizon distributions
			- "max": Element-wise maximum across horizons (diagnostic)
		**kwargs: Additional arguments passed to forecast_timeline().
		
	Returns:
		Probability distribution array of shape (N,) summing to 1.0 representing
		cumulative probability across all horizons.
	"""
	# Get forecasts for all horizons
	forecasts = forecast_timeline(case, horizons=horizons, **kwargs)
	
	# Determine weights
	if weights is None:
		if len(horizons) == 3:
			# Time-biased default: earlier horizons weighted more heavily
			weights = [0.5, 0.3, 0.2]
		else:
			# Uniform average for other numbers of horizons
			weights = [1.0 / len(horizons)] * len(horizons)
	
	# Normalize weights
	total_weight = sum(weights)
	if total_weight > 0:
		weights = [w / total_weight for w in weights]
	else:
		weights = [1.0 / len(horizons)] * len(horizons)
	
	# Combine distributions based on mode
	sorted_horizons = sorted(horizons)
	if mode == "max":
		# Element-wise maximum (diagnostic mode)
		p_cum = np.maximum.reduce([forecasts[h] for h in sorted_horizons])
	else:
		# Weighted average (default)
		p_cum = np.zeros_like(forecasts[sorted_horizons[0]])
		for h, w in zip(sorted_horizons, weights):
			p_cum += w * forecasts[h]
	
	# Normalize result
	p_cum = p_cum / (p_cum.sum() + 1e-9)
	
	return p_cum


def attach_sector_probs(
	grid_xy: np.ndarray,
	p: np.ndarray,
	sector_path: str = "data/geo/va_rl_regions.geojson"
) -> Dict[str, Any]:
	"""Attach sector-level probabilities to a probability distribution.
	
	Args:
		grid_xy: Grid coordinates array (N, 2) with (lon, lat).
		p: Probability distribution array (N,).
		sector_path: Path to sectors GeoJSON.
		
	Returns:
		Dictionary with keys:
		- sectors_gdf: GeoDataFrame of sectors
		- sector_idx: Array mapping grid points to sector indices
		- sectors_ranked: List of ranked sector dictionaries
	"""
	from reinforcement_learning.sectors import (
		load_sectors,
		assign_grid_to_sectors,
		rank_sectors,
	)
	
	# Load sectors
	sectors_gdf = load_sectors(sector_path)
	
	# Assign grid points to sectors
	sector_idx = assign_grid_to_sectors(grid_xy, sectors_gdf)
	
	# Rank sectors by probability mass
	sectors_ranked = rank_sectors(p, sector_idx, sectors_gdf)
	
	return {
		"sectors_gdf": sectors_gdf,
		"sector_idx": sector_idx,
		"sectors_ranked": sectors_ranked,
	}


def forecast_search_plan(
	case: Dict[str, Any],
	horizons: Tuple[int, ...] = (24, 48, 72),
	use_cumulative: bool = True,
	sector_path: str = "data/geo/va_rl_regions.geojson",
	hotspot_pct: float = 0.9,
	**kwargs
) -> Dict[str, Any]:
	"""Generate a complete search plan with sectors, hotspots, and rings.
	
	Args:
		case: Case dictionary with spatial info.
		horizons: Time horizons for forecast (hours).
		use_cumulative: If True, use cumulative forecast across horizons.
		sector_path: Path to sectors GeoJSON.
		hotspot_pct: Percentile for sector hotspots (default: 0.9).
		**kwargs: Additional arguments passed to forecast functions.
		
	Returns:
		Dictionary with keys:
		- grid_xy: Grid coordinates
		- p: Final probability distribution
		- sectors_gdf: Sector GeoDataFrame
		- sector_idx: Sector assignments
		- sectors_ranked: Ranked sector list
		- sectors_ranked_by_horizon: Dictionary mapping horizon hours to ranked sector lists
		- forecasts_by_horizon: Dictionary mapping horizon hours to probability distributions
		- sector_hotspots: Hotspot list per sector (for main distribution)
		- sector_hotspots_by_horizon: Dictionary mapping horizon hours to hotspot lists
		- rings: Probability containment rings (for main distribution)
		- rings_by_horizon: Dictionary mapping horizon hours to rings
		- ipp: Initial Planning Point (lon, lat)
		- sectors_metadata: Serializable sector metadata (no geometry)
		- sector_ids: List of sector IDs for geometry lookup
	"""
	from reinforcement_learning.sectors import sector_hotspots
	from reinforcement_learning.rings import probability_radii
	
	# 1) Get probability distribution for main visualization
	if use_cumulative:
		p = forecast_cumulative(case, horizons=horizons, **kwargs)
	else:
		max_horizon = max(horizons)
		p, _ = forecast_distribution(case, t_hours=float(max_horizon), **kwargs)
	
	# 1a) Always get forecasts for all horizons to compute per-horizon rankings
	forecasts_by_horizon = forecast_timeline(case, horizons=horizons, **kwargs)
	
	# 2) Load grid and layers
	method_weights = kwargs.get("method_weights")
	grid_xy, hotspots, road_cost, seclusion, va_mask, corridor_score = load_grid_and_layers(method_weights=method_weights)
	
	# 3) Sector probabilities for main distribution
	sector_info = attach_sector_probs(grid_xy, p, sector_path=sector_path)
	sectors_gdf = sector_info["sectors_gdf"]
	sector_idx = sector_info["sector_idx"]
	sectors_ranked = sector_info["sectors_ranked"]
	
	# 3a) Compute sector rankings for each horizon separately
	sectors_ranked_by_horizon = {}
	for horizon in sorted(horizons):
		p_horizon = forecasts_by_horizon[horizon]
		sector_info_h = attach_sector_probs(grid_xy, p_horizon, sector_path=sector_path)
		sectors_ranked_by_horizon[horizon] = sector_info_h["sectors_ranked"]
	
	# 4) Sector hotspots for main distribution
	sector_hotspots_list = sector_hotspots(
		grid_xy, p, sector_idx, sectors_ranked, sectors_gdf,
		local_pct=hotspot_pct,
		max_hotspots_per_sector=kwargs.get("max_hotspots_per_sector")
	)
	
	# 4a) Compute sector hotspots for each horizon
	sector_hotspots_by_horizon = {}
	for horizon in sorted(horizons):
		p_horizon = forecasts_by_horizon[horizon]
		sector_info_h = attach_sector_probs(grid_xy, p_horizon, sector_path=sector_path)
		sector_idx_h = sector_info_h["sector_idx"]
		sectors_ranked_h = sector_info_h["sectors_ranked"]
		sector_hotspots_by_horizon[horizon] = sector_hotspots(
			grid_xy, p_horizon, sector_idx_h, sectors_ranked_h, sectors_gdf,
			local_pct=hotspot_pct,
			max_hotspots_per_sector=kwargs.get("max_hotspots_per_sector")
		)
	
	# 5) Extract IPP and compute rings
	ipp = None
	rings = []
	rings_by_horizon = {}
	
	try:
		lat, lon = assert_seed_present(case)
		ipp = {"lon": float(lon), "lat": float(lat)}
		
		# Compute probability containment rings for main distribution
		rings = probability_radii(grid_xy, p, lon, lat)
		
		# Compute rings for each horizon
		for horizon in sorted(horizons):
			p_horizon = forecasts_by_horizon[horizon]
			rings_by_horizon[horizon] = probability_radii(grid_xy, p_horizon, lon, lat)
	except (KeyError, Exception):
		# IPP not available, skip rings
		pass
	
	# 6) Create serializable sector metadata (no geometry)
	sectors_metadata = []
	sector_ids = []
	for sector in sectors_ranked:
		sectors_metadata.append({
			"sector_id": sector["sector_id"],
			"name": sector["name"],
			"region_tag": sector["region_tag"],
			"mass": sector["mass"],
			"mass_pct": sector["mass_pct"],
		})
		sector_ids.append(sector["sector_id"])
	
	return {
		"grid_xy": grid_xy,
		"p": p,
		"forecasts_by_horizon": forecasts_by_horizon,
		"sectors_gdf": sectors_gdf,
		"sector_idx": sector_idx,
		"sectors_ranked": sectors_ranked,
		"sectors_ranked_by_horizon": sectors_ranked_by_horizon,
		"sector_hotspots": sector_hotspots_list,
		"sector_hotspots_by_horizon": sector_hotspots_by_horizon,
		"rings": rings,
		"rings_by_horizon": rings_by_horizon,
		"ipp": ipp,
		"sectors_metadata": sectors_metadata,
		"sector_ids": sector_ids,
	}

