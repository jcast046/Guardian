"""Movement model for risk map computation using Markov chain propagation.

This module implements a probabilistic movement model that combines:
- Kernel density estimation (KDE) priors from hotspot data
- Transition matrices based on road costs and seclusion factors
- Temporal decay to model survival probability over time
- Markov chain propagation to predict risk distribution

The model computes spatial risk maps by propagating initial probability
distributions through a transition matrix representing movement patterns.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Miles per degree latitude (approximately constant across latitudes)
MILES_PER_DEG_LAT = 69.13


def load_hotspots_multi(
	kmeans_path: str = "eda_out/kmeans_hotspots.json",
	dbscan_path: str = "eda_out/dbscan_hotspots.json",
	kde_path: str = "eda_out/kde_hotspots.json",
	method_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, float, float, float]]:
	"""Load and combine hotspots from multiple clustering methods.
	
	Loads hotspots from KMeans, DBSCAN, and KDE JSON files and combines them
	into a unified list. Each hotspot's weight is multiplied by its method-level
	coefficient, and final weights are normalized to sum to 1.0.
	
	Args:
		kmeans_path: Path to KMeans hotspots JSON file.
		dbscan_path: Path to DBSCAN hotspots JSON file.
		kde_path: Path to KDE hotspots JSON file.
		method_weights: Optional dict mapping method names to weights.
			Default: {"kmeans": 1.0, "dbscan": 1.0, "kde": 1.0}
			Example: {"kmeans": 1.0, "dbscan": 0.8, "kde": 1.2}
	
	Returns:
		List of tuples (lon, lat, w, s_miles) where:
		- lon: Longitude in degrees
		- lat: Latitude in degrees
		- w: Normalized weight (sum of all weights = 1.0)
		- s_miles: Gaussian standard deviation in miles
	
	Raises:
		FileNotFoundError: If any required hotspot file is missing.
		ValueError: If method_weights contains invalid method names.
	"""
	if method_weights is None:
		method_weights = {"kmeans": 1.0, "dbscan": 1.0, "kde": 1.0}
	
	# Validate method_weights keys
	valid_methods = {"kmeans", "dbscan", "kde"}
	if not valid_methods.issuperset(method_weights.keys()):
		invalid = set(method_weights.keys()) - valid_methods
		raise ValueError(f"Invalid method names in method_weights: {invalid}")
	
	all_hotspots = []
	
	# Helper function to parse hotspot (handles both old list format and new dict format)
	def parse_hotspot(h):
		"""Parse hotspot from either list [lon, lat, weight, radius_miles] or dict format."""
		if isinstance(h, list):
			# Old format: [lon, lat, weight, radius_miles]
			if len(h) >= 4:
				return float(h[0]), float(h[1]), float(h[2]), float(h[3])
			elif len(h) >= 2:
				return float(h[0]), float(h[1]), 1.0, 0.0
			else:
				return 0.0, 0.0, 0.0, 0.0
		elif isinstance(h, dict):
			# New format: {"lon": ..., "lat": ..., "weight": ..., "radius_miles": ...}
			return (
				float(h.get("lon", 0.0)),
				float(h.get("lat", 0.0)),
				float(h.get("weight", 0.0)),
				float(h.get("radius_miles", 0.0))
			)
		else:
			return 0.0, 0.0, 0.0, 0.0
	
	# Load KMeans hotspots
	if Path(kmeans_path).exists():
		with open(kmeans_path, "r", encoding="utf-8") as f:
			kmeans_data = json.load(f)
		
		# Compute median radius for KMeans if needed
		kmeans_radii = []
		for h in kmeans_data:
			_, _, _, radius_miles = parse_hotspot(h)
			if radius_miles > 0.0:
				kmeans_radii.append(radius_miles)
		median_radius = float(np.median(kmeans_radii)) if kmeans_radii else 5.0
		
		for h in kmeans_data:
			lon, lat, weight, radius_miles = parse_hotspot(h)
			
			# Handle zero radius: use median of nonzero radii or default
			if radius_miles <= 0.0:
				s_miles = median_radius if median_radius > 0 else 5.0
			else:
				s_miles = radius_miles
			
			# Apply method weight
			adjusted_weight = weight * method_weights.get("kmeans", 1.0)
			if adjusted_weight > 0:
				all_hotspots.append((lon, lat, adjusted_weight, s_miles))
	else:
		raise FileNotFoundError(f"KMeans hotspots file not found: {kmeans_path}")
	
	# Load DBSCAN hotspots
	if Path(dbscan_path).exists():
		with open(dbscan_path, "r", encoding="utf-8") as f:
			dbscan_data = json.load(f)
		
		# Compute median radius for DBSCAN if needed
		dbscan_radii = []
		for h in dbscan_data:
			_, _, _, radius_miles = parse_hotspot(h)
			if radius_miles > 0.0:
				dbscan_radii.append(radius_miles)
		median_radius = float(np.median(dbscan_radii)) if dbscan_radii else 5.0
		
		for h in dbscan_data:
			lon, lat, weight, radius_miles = parse_hotspot(h)
			
			# Handle zero radius: use median of nonzero radii or default
			if radius_miles <= 0.0:
				s_miles = median_radius if median_radius > 0 else 5.0
			else:
				s_miles = radius_miles
			
			# Apply method weight
			adjusted_weight = weight * method_weights.get("dbscan", 1.0)
			if adjusted_weight > 0:
				all_hotspots.append((lon, lat, adjusted_weight, s_miles))
	else:
		raise FileNotFoundError(f"DBSCAN hotspots file not found: {dbscan_path}")
	
	# Load KDE hotspots
	if Path(kde_path).exists():
		with open(kde_path, "r", encoding="utf-8") as f:
			kde_data = json.load(f)
		
		# Compute median radius for KDE if needed
		kde_radii = []
		for h in kde_data:
			_, _, _, radius_miles = parse_hotspot(h)
			if radius_miles > 0.0:
				kde_radii.append(radius_miles)
		median_radius = float(np.median(kde_radii)) if kde_radii else 5.0
		
		for h in kde_data:
			lon, lat, weight, radius_miles = parse_hotspot(h)
			
			# Handle zero radius: use median of nonzero radii or default
			if radius_miles <= 0.0:
				s_miles = median_radius if median_radius > 0 else 5.0
			else:
				s_miles = radius_miles
			
			# Apply method weight
			adjusted_weight = weight * method_weights.get("kde", 1.0)
			if adjusted_weight > 0:
				all_hotspots.append((lon, lat, adjusted_weight, s_miles))
	else:
		raise FileNotFoundError(f"KDE hotspots file not found: {kde_path}")
	
	# Normalize weights so sum = 1.0
	total_weight = sum(w for _, _, w, _ in all_hotspots)
	if total_weight > 0:
		normalized_hotspots = [
			(lon, lat, w / total_weight, s_miles)
			for lon, lat, w, s_miles in all_hotspots
		]
	else:
		# Fallback: equal weights if all weights are zero
		n = len(all_hotspots)
		normalized_hotspots = [
			(lon, lat, 1.0 / n if n > 0 else 0.0, s_miles)
			for lon, lat, _, s_miles in all_hotspots
		]
	
	return normalized_hotspots


def kde_prior(grid_xy: np.ndarray, hotspots):
	"""Compute KDE-based prior probability distribution from hotspot data.

	Combines multiple Gaussian kernels centered at hotspot locations to create
	a spatial prior distribution. Each hotspot contributes a weighted Gaussian
	with its own scale parameter. Distances are computed in miles accounting
	for latitude-dependent longitude scaling.

	Args:
		grid_xy: Array of shape (N, 2) containing (longitude, latitude) coordinates.
		hotspots: List of tuples (x0, y0, w, s_miles) where:
			- x0: Hotspot longitude (degrees)
			- y0: Hotspot latitude (degrees)
			- w: Hotspot weight
			- s_miles: Gaussian standard deviation in miles

	Returns:
		Array of shape (N,) containing normalized prior probabilities summing to 1.
	"""
	Z = np.zeros(len(grid_xy), float)

	for (x0, y0, w, s_miles) in hotspots:
		dx_deg = grid_xy[:, 0] - x0
		dy_deg = grid_xy[:, 1] - y0

		# Longitude degrees scale with cos(latitude) when converting to miles
		coslat = np.cos(np.radians(grid_xy[:, 1]))
		dx_miles = dx_deg * MILES_PER_DEG_LAT * coslat
		dy_miles = dy_deg * MILES_PER_DEG_LAT

		Z += w * np.exp(-(dx_miles*dx_miles + dy_miles*dy_miles) / (2*s_miles*s_miles))

	Z /= (Z.sum() + 1e-9)

	return Z


def survival_curve_exponential(t_hours: float, half_life: float = 24.0) -> float:
	"""Compute exponential survival decay factor.

	Models the probability that a subject remains active over time using
	exponential decay with a specified half-life.

	Args:
		t_hours: Time elapsed in hours.
		half_life: Half-life in hours (default: 24.0). After this time,
			the survival probability is 0.5.

	Returns:
		Survival probability factor between 0 and 1.
	"""
	lam = np.log(2.0) / max(half_life, 1e-6)
	return np.exp(-lam * t_hours)


def survival_decay(t_hours: float, half_life: float = 24.0) -> float:
	"""Compute exponential survival decay factor (backward-compatible alias).

	This function is a backward-compatible alias for survival_curve_exponential().
	Use survival_factor() for profile-based survival curves.

	Args:
		t_hours: Time elapsed in hours.
		half_life: Half-life in hours (default: 24.0). After this time,
			the survival probability is 0.5.

	Returns:
		Survival probability factor between 0 and 1.
	"""
	return survival_curve_exponential(t_hours, half_life=half_life)


def survival_factor(t_hours: float, profile: str = "default") -> float:
	"""Compute survival probability factor based on case profile.

	Wrapper function that selects appropriate survival curve parameters
	based on case type. Currently uses exponential decay with profile-specific
	half-lives. Can be extended to use more sophisticated survival models.

	Args:
		t_hours: Time elapsed in hours.
		profile: Case profile type (default: "default").
			- "default": Standard case (half_life=24h)
			- "runaway": Runaway case (half_life=48h)
			- "abduction": Abduction case (half_life=12h)

	Returns:
		Survival probability factor between 0 and 1.
	"""
	if profile == "default":
		return survival_curve_exponential(t_hours, half_life=24.0)
	elif profile == "runaway":
		return survival_curve_exponential(t_hours, half_life=48.0)
	elif profile == "abduction":
		return survival_curve_exponential(t_hours, half_life=12.0)
	else:
		# Fallback to default
		return survival_curve_exponential(t_hours, half_life=24.0)


# Module-level cache for transition matrices to avoid recomputation
_P_CACHE = {}


def build_transition(grid_xy: np.ndarray, road_cost: np.ndarray, seclusion: np.ndarray, k: int = 12, beta_cost: float = 1.0, beta_secl: float = 0.5, corridor_score: Optional[np.ndarray] = None, beta_corr: float = 0.0):
	"""Build transition probability matrix for Markov chain movement model.

	Constructs a transition matrix where each row represents probabilities
	of moving from one grid cell to its k nearest neighbors. Transition
	probabilities are weighted by:
	- Low road cost (preferred movement paths)
	- High seclusion (preferred hiding locations)
	- High corridor proximity (preferred highway/major road access) [optional]

	Results are cached to avoid recomputation for identical inputs.

	Args:
		grid_xy: Array of shape (N, 2) containing (longitude, latitude) coordinates.
		road_cost: Array of shape (N,) containing road accessibility costs.
			Lower values indicate easier movement.
		seclusion: Array of shape (N,) containing seclusion scores.
			Higher values indicate better hiding locations.
		k: Number of nearest neighbors to consider for transitions (default: 12).
		beta_cost: Cost penalty coefficient (default: 1.0). Higher values
			penalize high-cost transitions more.
		beta_secl: Seclusion reward coefficient (default: 0.5). Higher values
			reward secluded locations more.
		corridor_score: Optional array of shape (N,) with corridor proximity scores [0, 1].
			Higher values indicate closer proximity to major interstates/highways.
		beta_corr: Corridor reward coefficient (default: 0.0). Higher values
			reward corridor-proximate locations more. Ignored if corridor_score is None.

	Returns:
		Array of shape (N, N) representing row-stochastic transition matrix.
		Each row sums to 1 and represents transition probabilities from
		that grid cell to all others.
	"""
	# Compute cache key including corridor_score sum and beta_corr
	corr_sum = float(np.sum(corridor_score)) if corridor_score is not None else 0.0
	cache_key = (
		grid_xy.shape,
		float(np.sum(road_cost)),
		float(np.sum(seclusion)),
		corr_sum,
		k,
		beta_cost,
		beta_secl,
		beta_corr,
	)
	
	if cache_key in _P_CACHE:
		return _P_CACHE[cache_key]

	from sklearn.neighbors import NearestNeighbors

	nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(grid_xy)
	idx = nbrs.kneighbors(return_distance=False)

	N = len(grid_xy)
	P = np.zeros((N, N), float)

	rc = (road_cost - float(np.min(road_cost))) / (float(np.ptp(road_cost)) + 1e-9)
	sc = (seclusion - float(np.min(seclusion))) / (float(np.ptp(seclusion)) + 1e-9)
	
	# Normalize corridor score if provided
	if corridor_score is not None:
		cc = (corridor_score - float(np.min(corridor_score))) / (float(np.ptp(corridor_score)) + 1e-9)
	else:
		cc = np.zeros_like(road_cost)

	for i in range(N):
		js = idx[i]
		w = np.exp(-beta_cost * rc[js] + beta_secl * sc[js] + beta_corr * cc[js])

		P[i, js] = w
		P[i, i] += 1e-6
		P[i] /= (P[i].sum() + 1e-12)

	# Validate transition matrix
	row_sums = P.sum(axis=1)
	if not np.all(np.isfinite(row_sums)):
		raise ValueError("Transition matrix has non-finite row sums")
	if not np.allclose(row_sums, 1.0, atol=1e-6):
		# Re-normalize strictly
		P = P / (row_sums[:, None] + 1e-12)
		# Log warning if significantly off
		max_deviation = np.max(np.abs(row_sums - 1.0))
		if max_deviation > 1e-3:
			import warnings
			warnings.warn(f"Transition matrix row sums deviated from 1.0 (max: {max_deviation:.6f}), re-normalized")

	# Ensure non-negativity
	if not np.all(P >= 0):
		raise ValueError("Transition matrix contains negative values")

	_P_CACHE[cache_key] = P

	return P


def risk_map(grid_xy: np.ndarray, hotspots, road_cost: np.ndarray, seclusion: np.ndarray, t_hours: float, steps: int = 3, init: np.ndarray | None = None, alpha_prior: float = 0.5):
	"""Compute spatial risk map using Markov chain propagation.

	Combines hotspot priors, movement transitions, and temporal decay to
	produce a probability distribution over grid locations. The distribution
	is propagated through a Markov chain for the specified number of steps,
	then scaled by survival decay based on elapsed time.

	Args:
		grid_xy: Array of shape (N, 2) containing (longitude, latitude) coordinates.
		hotspots: List of hotspot tuples for KDE prior computation.
		road_cost: Array of shape (N,) containing road accessibility costs.
		seclusion: Array of shape (N,) containing seclusion scores.
		t_hours: Time elapsed in hours since last sighting.
		steps: Number of Markov chain propagation steps (default: 3).
		init: Optional initial probability distribution. If None, uses KDE prior.
		alpha_prior: Mixing weight for KDE prior when init is provided (default: 0.5).
			Value of 1.0 uses only prior, 0.0 uses only init.

	Returns:
		Array of shape (N,) containing normalized risk probabilities summing to 1.
	"""
	prior = kde_prior(grid_xy, hotspots)
	P = build_transition(grid_xy, road_cost, seclusion)

	if init is None:
		p = prior
	else:
		# Mix KDE prior and case-specific seed
		p = alpha_prior * prior + (1.0 - alpha_prior) * init
		p = p / (p.sum() + 1e-9)

	for _ in range(steps):
		p = P.T @ p

	# Use survival_factor for profile-based decay (defaults to standard exponential)
	r = p * survival_factor(t_hours, profile="default")

	return r / (r.sum() + 1e-9)


