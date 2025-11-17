"""Script for building RL search zones from movement model predictions.

Loads preprocessed geographic and transportation data, generates search zones
using the movement model, and evaluates them against ground truth. Processes
synthetic case files and produces zone recommendations with reward scoring.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_here = Path(__file__).resolve()
_proj_root = _here.parents[1]
if str(_proj_root) not in sys.path:
	sys.path.insert(0, str(_proj_root))

import json, glob, numpy as np  
from reinforcement_learning.zone_rl import run_one

# Geographic geometry imports
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.ops import unary_union

# Conversion constant: miles to meters
MI_TO_M = 1609.344
# Miles per degree latitude (approximately constant across latitudes)
MILES_PER_DEG_LAT = 69.13  


def get_last_seen(case):
	"""Extract last_seen coordinates with fallback support for multiple case schemas.
	
	Supports multiple case structure formats:
	- Primary: case["spatial"]["last_seen_lat/lon"]
	- Fallback 1: case["last_seen"]["lat/lon"]
	- Fallback 2: case["provenance"]["last_seen"]["lat/lon"]
	
	Args:
		case: Case dictionary containing location data.
		
	Returns:
		Tuple (lat, lon) as floats.
		
	Raises:
		KeyError: If coordinates cannot be found in any supported schema.
	"""
	sp = case.get("spatial", {})
	lat, lon = sp.get("last_seen_lat"), sp.get("last_seen_lon")
	if lat is not None and lon is not None:
		return float(lat), float(lon)
	
	ls = case.get("last_seen")
	if isinstance(ls, dict) and "lat" in ls and "lon" in ls:
		return float(ls["lat"]), float(ls["lon"])
	
	prov = case.get("provenance", {}).get("last_seen")
	if isinstance(prov, dict) and "lat" in prov and "lon" in prov:
		return float(prov["lat"]), float(prov["lon"])
	
	raise KeyError(f"[{case.get('case_id')}] missing last_seen coords")


def assert_seed_present(case):
	"""Assert and return last_seen coordinates. Raises KeyError if missing.
	
	Args:
		case: Case dictionary containing location data.
		
	Returns:
		Tuple (lat, lon) as floats.
		
	Raises:
		KeyError: If coordinates cannot be found.
	"""
	lat, lon = get_last_seen(case)
	return lat, lon


def _seed_from_disappearance(grid_xy, disp_lon, disp_lat, sigma_mi=5.0):
	"""Create Gaussian seed around the last-seen position (disappearance point).
	
	Creates a normalized probability distribution centered at the disappearance
	location with a Gaussian kernel. All distances are computed in miles.
	
	Args:
		grid_xy: Array of shape (N, 2) with (longitude, latitude) coordinates.
		disp_lon: Disappearance longitude in degrees.
		disp_lat: Disappearance latitude in degrees.
		sigma_mi: Gaussian standard deviation in miles (default: 5.0).
		
	Returns:
		Array of shape (N,) containing normalized probabilities summing to 1.
	"""
	x = grid_xy[:, 0]
	y = grid_xy[:, 1]
	
	# Convert lon delta to miles using local cos(lat)
	coslat = np.cos(np.radians(disp_lat))
	dx_mi = (x - disp_lon) * MILES_PER_DEG_LAT * coslat
	dy_mi = (y - disp_lat) * MILES_PER_DEG_LAT
	
	Z = np.exp(-(dx_mi*dx_mi + dy_mi*dy_mi) / (2.0 * sigma_mi * sigma_mi))
	Z /= (Z.sum() + 1e-12)
	
	return Z


def _normalize(vec):
	"""Normalize a probability vector to sum to 1.
	
	Args:
		vec: Array-like probability vector.
		
	Returns:
		Normalized array with same shape, summing to 1.
	"""
	s = float(vec.sum())
	return vec / s if s > 0 else vec


def load_grid_and_layers(method_weights=None):
	"""Load preprocessed geographic and transportation data from cache files.

	Loads the grid coordinates, road cost matrix, seclusion scores, hotspot data,
	Virginia boundary mask, and corridor scores required for movement model computation.
	Validates that all required files exist and provides informative error messages if
	they are missing. Uses multi-source hotspot loading to combine KMeans, DBSCAN,
	and KDE hotspots.

	Args:
		method_weights: Optional dict mapping method names to weights for
			hotspot combination. Default: {"kmeans": 1.0, "dbscan": 1.0, "kde": 1.0}

	Returns:
		Tuple containing:
		- grid_xy: Array of shape (N, 2) with (longitude, latitude) coordinates.
		- hotspots: List of hotspot tuples (lon, lat, w, s_miles) for KDE prior computation.
		- road_cost: Array of shape (N,) with road accessibility costs.
		- seclusion: Array of shape (N,) with seclusion scores.
		- va_mask: Boolean array of shape (N,) indicating VA boundary containment, or None.
		- corridor_score: Array of shape (N,) with corridor proximity scores [0, 1], or None.

	Raises:
		FileNotFoundError: If any required input files are missing from
			the eda_out/ directory.
	"""
	import os
	from reinforcement_learning.movement_model import load_hotspots_multi
	
	missing = []
	if not os.path.exists("eda_out/grid_xy.npy"):
		missing.append("eda_out/grid_xy.npy")
	if not os.path.exists("eda_out/road_cost.npy"):
		missing.append("eda_out/road_cost.npy")
	if not os.path.exists("eda_out/seclusion.npy"):
		missing.append("eda_out/seclusion.npy")
	if not os.path.exists("eda_out/kmeans_hotspots.json"):
		missing.append("eda_out/kmeans_hotspots.json")
	if not os.path.exists("eda_out/dbscan_hotspots.json"):
		missing.append("eda_out/dbscan_hotspots.json")
	if not os.path.exists("eda_out/kde_hotspots.json"):
		missing.append("eda_out/kde_hotspots.json")
	
	if missing:
		print("[ERROR] Missing required input files:")
		for f in missing:
			print(f"  - {f}")
		print("\n[INFO] These files need to be generated first:")
		print("  - grid_xy.npy: Tile centers or road nodes (lon, lat) from the grid/network")
		print("  - road_cost.npy: Minutes-to-reach from bounded Dijkstra (src/transportation/networks.py)")
		print("  - seclusion.npy: Composite score (low pop density + wooded/industrial POIs)")
		print("  - kmeans_hotspots.json, dbscan_hotspots.json, kde_hotspots.json: Hotspot data from eda_hotspot.py")
		print("\n[INFO] Run the preprocessing pipeline first, or create these files manually.")
		raise FileNotFoundError(f"Missing required input files: {', '.join(missing)}")
	
	grid_xy = np.load("eda_out/grid_xy.npy")
	road_cost = np.load("eda_out/road_cost.npy")
	seclusion = np.load("eda_out/seclusion.npy")
	hotspots = load_hotspots_multi(method_weights=method_weights)
	
	# Load Virginia boundary mask (optional, lazy compute if missing)
	va_mask = None
	if os.path.exists("eda_out/va_mask.npy"):
		va_mask = np.load("eda_out/va_mask.npy")
	elif os.path.exists("data/geo/va_boundary.geojson"):
		# Lazy compute if boundary file exists but mask doesn't
		print("[INFO] Computing VA boundary mask...")
		va_mask = compute_va_mask(grid_xy)
		os.makedirs("eda_out", exist_ok=True)
		np.save("eda_out/va_mask.npy", va_mask)
		print("[OK] Saved VA boundary mask to eda_out/va_mask.npy")
	
	# Load corridor score (optional, lazy compute if missing)
	corridor_score = None
	if os.path.exists("eda_out/corridor_score.npy"):
		corridor_score = np.load("eda_out/corridor_score.npy")
	elif os.path.exists("data/geo/va_corridors.geojson") or os.path.exists("data/transportation/va_road_segments.json"):
		# Lazy compute if corridor data exists but score file doesn't
		print("[INFO] Computing corridor proximity scores...")
		try:
			corridor_index = GeoCorridorIndex("data/geo/va_corridors.geojson", id_field="id")
			corridor_score = compute_corridor_score(grid_xy, corridor_index, max_radius_mi=5.0)
			os.makedirs("eda_out", exist_ok=True)
			np.save("eda_out/corridor_score.npy", corridor_score)
			print("[OK] Saved corridor proximity scores to eda_out/corridor_score.npy")
		except Exception as e:
			print(f"[WARN] Could not compute corridor scores: {e}. Using None (no corridor bias).")
			corridor_score = None
	
	# Validate shapes (handle None gracefully)
	assert va_mask is None or va_mask.shape == road_cost.shape == (len(grid_xy),), \
		f"Shape mismatch: va_mask={va_mask.shape if va_mask is not None else None}, road_cost={road_cost.shape}, grid_xy={len(grid_xy)}"
	assert corridor_score is None or corridor_score.shape == road_cost.shape == (len(grid_xy),), \
		f"Shape mismatch: corridor_score={corridor_score.shape if corridor_score is not None else None}, road_cost={road_cost.shape}, grid_xy={len(grid_xy)}"
	
	return grid_xy, hotspots, road_cost, seclusion, va_mask, corridor_score


def truth_adapter(case, actions_by_window):
	"""Extract ground truth points and compute zone evaluation metrics.

	Extracts temporal truth points from case data (last seen, follow-up
	sightings, recovery) and computes per-window metrics for each zone:
	- Minimum distance from zone center to any truth point
	- Earliest time when truth point enters zone radius
	- Whether zone matches assigned corridor

	Time windows are aligned to hours since last_seen_ts.

	Args:
		case: Case dictionary containing temporal, spatial, and outcome data.
		actions_by_window: Dictionary mapping window IDs to lists of zone
			dictionaries. Each zone must have center_lon, center_lat, and
			radius_miles keys.

	Returns:
		Dictionary mapping window IDs to dictionaries containing:
		- d_center_true_miles: List of minimum distances (miles) from zone
			centers to truth points.
		- t_hit_hr: List of earliest hit times (hours) or None if no hit.
		- corridor_match: List of booleans indicating corridor matches.

	Raises:
		ValueError: If case is missing required temporal.last_seen_ts or
			if no truth points can be extracted.
	"""
	from datetime import datetime
	from src.geography.distance import haversine_distance

	ls_iso = case.get("temporal", {}).get("last_seen_ts")
	if not ls_iso:
		raise ValueError("Case missing temporal.last_seen_ts")
	
	try:
		last_seen_dt = datetime.fromisoformat(ls_iso)
	except Exception:
		last_seen_dt = datetime.fromisoformat(ls_iso.replace("Z", "+00:00"))

	truth_points = []
	ls_lat = case.get("spatial", {}).get("last_seen_lat")
	ls_lon = case.get("spatial", {}).get("last_seen_lon")
	if ls_lat is not None and ls_lon is not None:
		truth_points.append((0.0, float(ls_lat), float(ls_lon)))

	for s in case.get("temporal", {}).get("follow_up_sightings", []) or []:
		try:
			ts = datetime.fromisoformat(s.get("ts", "").replace("Z", "+00:00"))
			lat = float(s.get("lat"))
			lon = float(s.get("lon"))
			dt_hours = max(0.0, (ts - last_seen_dt).total_seconds() / 3600.0)
			truth_points.append((dt_hours, lat, lon))
		except Exception:
			continue

	rec = case.get("outcome", {})
	if rec and rec.get("recovery_lat") is not None and rec.get("recovery_lon") is not None and rec.get("recovery_time_hours") is not None:
		try:
			truth_points.append((float(rec["recovery_time_hours"]), float(rec["recovery_lat"]), float(rec["recovery_lon"])))
		except Exception:
			pass

	truth_points.sort(key=lambda x: x[0])
	if not truth_points:
		raise ValueError("No truth points available in case")

	windows = {"0-24": (0.0, 24.0), "24-48": (24.0, 48.0), "48-72": (48.0, 72.0)}

	out = {}
	for wid, zones in (actions_by_window or {}).items():
		ws, we = windows.get(wid, (0.0, 24.0))
		pts_in = [p for p in truth_points if ws <= p[0] < we]
		pts_all = truth_points
		
		D = []
		T = []
		C = []
		
		for z in zones:
			clon = float(z.get("center_lon"))
			clat = float(z.get("center_lat"))
			rmi = float(z.get("radius_miles", 10.0))
			cand = pts_in if pts_in else pts_all
			min_d = min(haversine_distance(clat, clon, lat, lon) for _, lat, lon in cand)
			D.append(min_d)
			
			hit_time = None
			for th, lat, lon in pts_in:
				d = haversine_distance(clat, clon, lat, lon)
				if d <= rmi:
					hit_time = th
					break
			T.append(hit_time)
			
			corridor = z.get("corridor")
			C.append(corridor is not None and corridor != "")
		
		out[wid] = {"d_center_true_miles": D, "t_hit_hr": T, "corridor_match": C}

	return out


class MovementForecaster:
	"""Thin wrapper around movement model for sequential belief propagation.
	
	This class provides a clean interface for seeding belief distributions
	from disappearance locations and propagating them through time windows
	using Markov chain transitions. Caches transition matrices for efficiency.
	
	Args:
		grid_xy: Array of shape (N, 2) containing (longitude, latitude) coordinates.
		road_cost: Array of shape (N,) containing road accessibility costs.
		seclusion: Array of shape (N,) containing seclusion scores.
		k: Number of nearest neighbors for transition matrix (default: 16).
		beta_cost: Cost penalty coefficient (default: 1.0, legacy parameter).
		beta_secl: Seclusion reward coefficient (default: 0.5, legacy parameter).
		
	Attributes:
		grid_xy: Grid coordinates.
		road_cost: Road cost array.
		seclusion: Seclusion scores array.
		k: Nearest neighbors parameter (default: 16).
		beta_cost: Cost penalty coefficient (legacy, not used in day/night matrices).
		beta_secl: Seclusion reward coefficient (legacy, not used in day/night matrices).
		_P: Cached transition matrix (legacy, kept for backward compatibility).
		_P_day: Cached day transition matrix (lazy-loaded).
		_P_night: Cached night transition matrix (lazy-loaded).
	"""
	
	def __init__(self, grid_xy, road_cost, seclusion, k=16, beta_cost=1.0, beta_secl=0.5,
	             beta_cost_day=None, beta_secl_day=None,
	             beta_cost_night=None, beta_secl_night=None,
	             corridor_score=None, beta_corr_day=0.0, beta_corr_night=0.0):
		self.grid_xy = grid_xy
		self.road_cost = road_cost
		self.seclusion = seclusion
		self.k = k
		self.beta_cost = beta_cost  # Legacy parameter
		self.beta_secl = beta_secl  # Legacy parameter
		
		# Day/night specific parameters (default to legacy values if not provided)
		self.beta_cost_day = beta_cost_day if beta_cost_day is not None else beta_cost
		self.beta_secl_day = beta_secl_day if beta_secl_day is not None else beta_secl
		self.beta_cost_night = beta_cost_night if beta_cost_night is not None else 0.8
		self.beta_secl_night = beta_secl_night if beta_secl_night is not None else 0.8
		
		# Corridor score and beta_corr parameters (code defaults: 0.0 for graceful fallback)
		self.corridor_score = corridor_score
		self.beta_corr_day = beta_corr_day
		self.beta_corr_night = beta_corr_night
		
		self._P = None  # Cache for transition matrix (legacy, kept for backward compatibility)
		self._P_day = None  # Cache for day transition matrix
		self._P_night = None  # Cache for night transition matrix
	
	def _ensure_transition(self):
		"""Lazy-load and cache transition matrix (legacy method, uses day matrix).
		
		Returns:
			Transition matrix of shape (N, N).
		"""
		if self._P is None:
			self._P = self._ensure_transition_day()
		return self._P
	
	def _ensure_transition_day(self):
		"""Lazy-load and cache day transition matrix.
		
		Day matrix uses beta_cost_day, beta_secl_day, and beta_corr_day values.
		
		Returns:
			Transition matrix of shape (N, N) for day movement.
		"""
		if self._P_day is None:
			from reinforcement_learning.movement_model import build_transition
			self._P_day = build_transition(
				self.grid_xy, self.road_cost, self.seclusion,
				k=self.k, beta_cost=self.beta_cost_day, beta_secl=self.beta_secl_day,
				corridor_score=self.corridor_score, beta_corr=self.beta_corr_day
			)
		return self._P_day
	
	def _ensure_transition_night(self):
		"""Lazy-load and cache night transition matrix.
		
		Night matrix uses beta_cost_night, beta_secl_night, and beta_corr_night values
		to favor more secluded locations during night hours.
		
		Returns:
			Transition matrix of shape (N, N) for night movement.
		"""
		if self._P_night is None:
			from reinforcement_learning.movement_model import build_transition
			self._P_night = build_transition(
				self.grid_xy, self.road_cost, self.seclusion,
				k=self.k, beta_cost=self.beta_cost_night, beta_secl=self.beta_secl_night,
				corridor_score=self.corridor_score, beta_corr=self.beta_corr_night
			)
		return self._P_night
	
	def seed_from_point(self, lon, lat, sigma_mi=5.0):
		"""Create Gaussian seed from a point location.
		
		Args:
			lon: Longitude in degrees.
			lat: Latitude in degrees.
			sigma_mi: Gaussian standard deviation in miles (default: 5.0).
			
		Returns:
			Array of shape (N,) containing normalized probabilities.
		"""
		return _seed_from_disappearance(self.grid_xy, lon, lat, sigma_mi=sigma_mi)
	
	def step(self, pi, bucket=None, steps=1):
		"""Apply Markov chain propagation step with day/night transition matrices.
		
		Propagates probability distribution through transition matrix.
		Uses same convention as movement_model: P.T @ p where p is treated
		as a column vector. Selects day or night transition matrix based on
		bucket parameter.
		
		Args:
			pi: Probability distribution array of shape (N,).
			bucket: Optional time bucket (e.g., "day", "night", "overnight").
				If "night" or "overnight", uses night matrix (more seclusion).
				Otherwise, uses day matrix.
			steps: Number of propagation steps (default: 1).
			
		Returns:
			Normalized probability distribution after propagation.
		"""
		# Use night matrix for night/overnight, day matrix otherwise
		if bucket in ("night", "overnight"):
			P = self._ensure_transition_night()
		else:
			P = self._ensure_transition_day()
		
		# Use same convention as movement_model: P.T @ p (treating p as column vector)
		p = pi.copy()
		for _ in range(steps):
			p = P.T @ p
			p = _normalize(p)
		
		return p
	
	def risk_adjust(self, pi, bucket=None):
		"""Apply risk adjustment using road/seclusion factors.
		
		This is a lightweight multiplicative adjustment. For a more
		sophisticated approach, could integrate with risk_map logic.
		Currently returns the input unchanged (identity function).
		
		Args:
			pi: Probability distribution array of shape (N,).
			bucket: Optional time bucket for future use.
			
		Returns:
			Risk-adjusted probability distribution.
		"""
		# Simple multiplicative adjustment based on seclusion
		# Higher seclusion -> higher risk of being found there
		# Normalize seclusion to [0, 1] range
		sc = self.seclusion.copy()
		sc_min = float(np.min(sc))
		sc_max = float(np.max(sc))
		sc_range = sc_max - sc_min
		
		if sc_range > 1e-9:
			sc_norm = (sc - sc_min) / sc_range
			# Boost probability in secluded areas (multiply by 1 + seclusion)
			pi_adjusted = pi * (1.0 + 0.3 * sc_norm)
			return _normalize(pi_adjusted)
		
		return pi


class GeoCorridorIndex:
	"""Nearest-corridor lookup using GeoPandas + Shapely STRtree.

	This class loads corridors from a GeoJSON file and builds a spatial index
	for efficient nearest-corridor queries. Falls back to road segments JSON
	if GeoJSON is unavailable. All distances are in miles.

	Args:
		corridors_path: Path to GeoJSON file containing LineString/MultiLineString
			features. Must have an ID column (default: 'id'); will fallback to
			'name' or index.
		id_field: Name of the field containing corridor identifiers (default: 'id').
		road_segments_path: Path to road segments JSON file as fallback
			(default: 'data/transportation/va_road_segments.json').

	Attributes:
		gdf_ll: GeoDataFrame in lon/lat (EPSG:4326).
		gdf_m: GeoDataFrame projected to meters (EPSG:3857).
		ids: Array of corridor identifier strings.
		geoms: Array of Shapely geometries in meters.
		tree: STRtree spatial index for fast queries.
		road_segments_data: Dictionary containing road segments data (fallback mode).
		use_road_segments: Boolean indicating if using road segments fallback.
	"""

	def __init__(self, corridors_path="data/geo/va_corridors.geojson", id_field="id",
	             road_segments_path="data/transportation/va_road_segments.json"):
		self.gdf_ll = None
		self.gdf_m = None
		self.ids = None
		self.geoms = None
		self.tree = None
		self.id_field = id_field
		self.road_segments_data = None
		self.use_road_segments = False

		# Try to load from GeoJSON first
		try:
			gdf = gpd.read_file(corridors_path)
			if not gdf.empty:
				# Choose an id column
				id_col = id_field if id_field in gdf.columns else ("name" if "name" in gdf.columns else None)
				if id_col is None:
					# Synthesize an id
					gdf["__cid"] = [f"corr_{i}" for i in range(len(gdf))]
					id_col = "__cid"

				# Keep only line-like features
				gdf = gdf[gdf.geometry.notnull()].explode(index_parts=False, ignore_index=True)
				gdf = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])]

				if not gdf.empty:
					# Store lon/lat copy and a projected (meters) copy for distance math
					self.gdf_ll = gdf[[id_col, "geometry"]].rename(columns={id_col: "cid"}).copy()
					self.gdf_m = self.gdf_ll.to_crs(3857)

					self.ids = self.gdf_ll["cid"].to_numpy()
					self.geoms = self.gdf_m.geometry.to_numpy()
					self.tree = STRtree(self.geoms)
					return
		except Exception as e:
			pass

		# Fallback to road segments JSON
		try:
			with open(road_segments_path, "r", encoding="utf-8") as f:
				data = json.load(f)
			segments = data.get("road_segments", [])
			
			if segments:
				self.road_segments_data = data
				self.use_road_segments = True
				print(f"[INFO] Using road segments from {road_segments_path} for corridor matching.")
				return
		except Exception as e:
			pass

		print(f"[WARN] Could not load corridors from {corridors_path} or {road_segments_path}. Corridor bonus disabled.")

	def nearest_corridor(self, lon, lat, radius_mi):
		"""Return nearest corridor identifier within the specified radius.

		Args:
			lon: Longitude coordinate of zone center.
			lat: Latitude coordinate of zone center.
			radius_mi: Search radius in miles.

		Returns:
			Corridor identifier string if a corridor is within radius_mi, else None.
		"""
		if self.tree is not None:
			pt_ll = Point(float(lon), float(lat))
			pt_m = gpd.GeoSeries([pt_ll], crs=4326).to_crs(3857).iloc[0]

			# Query candidates by envelope; then compute exact distance
			buf_m = float(radius_mi) * MI_TO_M
			hits_ix = self.tree.query(pt_m.buffer(buf_m).envelope, predicate="intersects")

			if len(hits_ix) == 0:
				return None

			# Compute true nearest
			candidates = [self.geoms[i] for i in hits_ix]
			dists = [pt_m.distance(geom) for geom in candidates]
			j = min(range(len(dists)), key=dists.__getitem__)

			if dists[j] <= buf_m:
				# Map STRtree index -> original row
				try:
					i_global = hits_ix[j]
					return str(self.ids[i_global])
				except Exception:
					# Brute fallback (rare)
					geom = candidates[j]
					for i_global2, g in enumerate(self.geoms):
						if g.equals_exact(geom, 0.0):
							return str(self.ids[i_global2])

			return None

		# Fallback: Use road segments data with transit stations for approximate matching
		if self.use_road_segments and self.road_segments_data:
			from src.geography.distance import haversine_distance
			
			segments = self.road_segments_data.get("road_segments", [])
			if not segments:
				return None

			transit_stations = []
			try:
				transit_path = Path("data/transportation/va_transit.json")
				if transit_path.exists():
					with open(transit_path, "r", encoding="utf-8") as f:
						transit_data = json.load(f)
					transit_stations = transit_data.get("stations", [])
			except Exception:
				pass

			# Find transit stations within radius of query point
			nearby_stations = []
			for station in transit_stations:
				geom = station.get("geometry", {})
				if geom and geom.get("type") == "Point":
					coords = geom.get("coordinates", [])
					if len(coords) >= 2:
						st_lon, st_lat = float(coords[0]), float(coords[1])
						dist = haversine_distance(lat, lon, st_lat, st_lon)
						if dist <= radius_mi:
							nearby_stations.append({
								"lat": st_lat,
								"lon": st_lon,
								"name": station.get("name", ""),
								"tags": station.get("tags", {}),
								"dist": dist
							})

			# Match nearby stations to corridors using road segments
			corridor_matches = {}  # {corridor_id: min_distance}
			
			for station in nearby_stations:
				station_name = station["name"].upper()
				station_tags = str(station["tags"]).upper()
				
				# Check each road segment for matches
				for seg in segments:
					rd = seg.get("routeDesignation", {})
					route_system = rd.get("routeSystem", "")
					route_number = rd.get("routeNumber", "")
					
					# Only consider Interstates and US Routes
					if route_system in ["Interstate", "US Route"] and route_number and route_number != "Unknown":
						# Create corridor ID (e.g., "I-95", "US-29")
						if route_system == "Interstate":
							corridor_id = f"I-{route_number}"
						else:
							corridor_id = f"US-{route_number}"
						
						# Get local names for matching
						local_names = seg.get("localNames", [])
						
						# Check if station matches this corridor
						for corr_name in local_names:
							corr_name_upper = corr_name.upper()
							if corr_name_upper in station_name or corr_name_upper in station_tags:
								# Found a match
								if corridor_id not in corridor_matches:
									corridor_matches[corridor_id] = station["dist"]
								else:
									corridor_matches[corridor_id] = min(corridor_matches[corridor_id], station["dist"])

			# Return the closest matching corridor
			if corridor_matches:
				nearest_id = min(corridor_matches.items(), key=lambda x: x[1])[0]
				return nearest_id

			return None

		return None


class GeoStateMask:
	"""Virginia boundary containment (precise).

	This class loads Virginia boundary polygons and checks whether circular
	search zones are fully contained within the state. All distances are in miles.

	Args:
		boundary_path: Path to GeoJSON file containing Polygon/MultiPolygon of
			Virginia in EPSG:4326 (WGS84).
		name_field: Optional field name to filter features (for national files).
		name_value: Optional value to match in name_field (for national files).

	Attributes:
		poly_m: Union of all boundary polygons projected to EPSG:3857 (meters).
	"""

	def __init__(self, boundary_path="data/geo/va_boundary.geojson", name_field=None, name_value=None):
		# name_field/name_value are only needed if loading a *national* file
		# and need to filter to 'Virginia'. For dedicated VA file, leave None.
		try:
			gdf = gpd.read_file(boundary_path)
		except Exception as e:
			print(f"[WARN] Could not read boundary file: {boundary_path} ({e}). In-state check disabled.")
			self.poly_m = None
			return

		gdf = gdf[gdf.geometry.notnull()].explode(index_parts=False, ignore_index=True)

		if name_field and name_field in gdf.columns and name_value:
			gdf = gdf[gdf[name_field] == name_value]

		if gdf.empty:
			print(f"[WARN] Boundary file has no geometries after filtering: {boundary_path}")
			self.poly_m = None
			return

		# Dissolve to single polygon (union), then project to meters
		poly_ll = unary_union(gdf.geometry)
		self.poly_m = gpd.GeoSeries([poly_ll], crs=4326).to_crs(3857).iloc[0]

	def contains_circle(self, lat, lon, radius_mi):
		"""Check if a circular zone is fully contained within state boundaries.

		Args:
			lat: Latitude coordinate of zone center.
			lon: Longitude coordinate of zone center.
			radius_mi: Zone radius in miles.

		Returns:
			True if the entire circle is within Virginia boundaries, else False.
			Returns True if boundary file was not loaded (permissive fallback).
		"""
		if self.poly_m is None:
			return True  # Permissive fallback

		pt_ll = Point(float(lon), float(lat))
		pt_m = gpd.GeoSeries([pt_ll], crs=4326).to_crs(3857).iloc[0]
		circle = pt_m.buffer(float(radius_mi) * MI_TO_M)  # meters

		return circle.within(self.poly_m)


def compute_va_mask(grid_xy, boundary_path="data/geo/va_boundary.geojson"):
	"""Compute boolean mask for Virginia boundary containment.
	
	Creates a boolean array indicating which grid points are inside
	the Virginia state boundary. Used to remove probability outside VA.
	
	Args:
		grid_xy: Array of shape (N, 2) with (longitude, latitude) coordinates.
		boundary_path: Path to Virginia boundary GeoJSON file.
		
	Returns:
		Boolean array of shape (N,) where True means point is inside VA boundary.
		Returns all True if boundary file cannot be loaded (permissive fallback).
	"""
	try:
		gdf = gpd.read_file(boundary_path)
	except Exception as e:
		print(f"[WARN] Could not load boundary file: {boundary_path} ({e}). Using permissive mask (all True).")
		return np.ones(len(grid_xy), dtype=bool)
	
	if gdf.empty:
		print(f"[WARN] Boundary file is empty: {boundary_path}. Using permissive mask (all True).")
		return np.ones(len(grid_xy), dtype=bool)
	
	# Explode MultiPolygons and filter null geometries
	gdf = gdf[gdf.geometry.notnull()].explode(index_parts=False, ignore_index=True)
	
	if gdf.empty:
		print(f"[WARN] No valid geometries in boundary file. Using permissive mask (all True).")
		return np.ones(len(grid_xy), dtype=bool)
	
	# Create union of all boundary polygons in geographic CRS
	boundary_union = unary_union(gdf.geometry)
	
	# Create mask by checking if each point is inside boundary
	mask = np.array([
		boundary_union.contains(Point(lon, lat))
		for lon, lat in grid_xy
	], dtype=bool)
	
	return mask


def compute_corridor_score(grid_xy, corridor_index, max_radius_mi=5.0):
	"""Compute corridor proximity scores for grid cells.
	
	Creates a proximity score array [0, 1] for each grid cell based on
	distance to the nearest major interstate or highway corridor in Virginia.
	Higher scores indicate closer proximity to corridors (1.0 = on/near corridor,
	0.0 = far from corridors). Applies to all major interstates and highways.
	
	Args:
		grid_xy: Array of shape (N, 2) with (longitude, latitude) coordinates.
		corridor_index: GeoCorridorIndex instance for corridor lookup.
		max_radius_mi: Maximum radius in miles for proximity scoring (default: 5.0).
			Points beyond this distance get score 0.0.
			
	Returns:
		Array of shape (N,) with values in [0, 1] representing corridor proximity.
		Returns all zeros if corridor_index is None or invalid.
	"""
	if corridor_index is None:
		print("[WARN] Corridor index is None. Returning zero corridor scores.")
		return np.zeros(len(grid_xy), dtype=float)
	
	# Check if we have GeoJSON geometries (preferred) or road segments (fallback)
	if not hasattr(corridor_index, 'gdf_ll') or corridor_index.gdf_ll is None:
		if not hasattr(corridor_index, 'use_road_segments') or not corridor_index.use_road_segments:
			print("[WARN] Corridor index has no valid geometries. Returning zero corridor scores.")
			return np.zeros(len(grid_xy), dtype=float)
	
	scores = np.zeros(len(grid_xy), dtype=float)
	
	# Use geographic CRS for distance computation (simpler)
	if hasattr(corridor_index, 'gdf_ll') and corridor_index.gdf_ll is not None:
		corridor_gdf = corridor_index.gdf_ll
		
		for i, (lon, lat) in enumerate(grid_xy):
			point = Point(lon, lat)
			min_dist_mi = float('inf')
			
			# Compute distance to all corridor geometries
			for geom in corridor_gdf.geometry:
				if geom is not None:
					try:
						# Distance in degrees
						dist_deg = point.distance(geom)
						# Convert to miles: 1 degree lat ≈ 69.13 miles
						# For longitude, scale by cos(latitude)
						coslat = np.cos(np.radians(lat))
						# Approximate: use average of lat and lon scaling
						dist_mi = dist_deg * 69.13 * ((1.0 + coslat) / 2.0)
						min_dist_mi = min(min_dist_mi, dist_mi)
					except Exception:
						continue
			
			if min_dist_mi < float('inf') and min_dist_mi <= max_radius_mi:
				# Convert distance to score: score = max(0, 1 - dist / max_radius_mi)
				score = max(0.0, 1.0 - (min_dist_mi / max_radius_mi))
				scores[i] = score
	
	return scores


def _assert_seed_used(case):
	"""Assert that case has last_seen coordinates available.
	
	Raises AssertionError if last_seen coordinates are missing.
	This helps catch cases that might fall back to defaults.
	
	Args:
		case: Case dictionary containing location data.
		
	Raises:
		AssertionError: If last_seen coordinates are missing.
	"""
	sp = case.get("spatial", {})
	lat, lon = sp.get("last_seen_lat"), sp.get("last_seen_lon")
	assert lat is not None and lon is not None, \
		f"[{case.get('case_id')}] missing last_seen coords"


def choose_sigma_mi(lat, lon):
	"""Choose sigma based on geographic region.
	
	Returns sigma in miles based on known metro areas in Virginia.
	
	Args:
		lat: Latitude of last-seen location.
		lon: Longitude of last-seen location.
		
	Returns:
		Sigma value in miles.
	"""
	# NoVA / DC ring: 38.6 ≤ lat ≤ 39.1, -77.6 ≤ lon ≤ -76.8
	if 38.6 <= lat <= 39.1 and -77.6 <= lon <= -76.8:
		return 1.86  # 3.0 km = 1.86 miles
	# Tidewater / Hampton Roads: 36.6 ≤ lat ≤ 37.4, -76.9 ≤ lon ≤ -75.9
	if 36.6 <= lat <= 37.4 and -76.9 <= lon <= -75.9:
		return 2.17  # 3.5 km = 2.17 miles
	# Richmond metro: 37.3 ≤ lat ≤ 37.7, -77.7 ≤ lon ≤ -77.2
	if 37.3 <= lat <= 37.7 and -77.7 <= lon <= -77.2:
		return 2.17  # 3.5 km = 2.17 miles
	# Else suburban/rural
	return 3.73  # 6.0 km = 3.73 miles


def nms_by_center(zones, min_center_gap_mi=0.25):
	"""Non-max suppression: remove zones with centers too close together.
	
	Args:
		zones: List of zone dictionaries (should be sorted by priority, highest first)
		min_center_gap_mi: Minimum distance between zone centers in miles
		
	Returns:
		Filtered list of zones with sufficient spacing (always preserves at least 1 zone)
	"""
	from math import radians, sin, cos, asin, sqrt
	R = 3958.7613  # Earth radius in miles
	
	def mi(a, b):
		dlat = radians(b[0] - a[0])
		dlon = radians(b[1] - a[1])
		h = sin(dlat/2)**2 + cos(radians(a[0])) * cos(radians(b[0])) * sin(dlon/2)**2
		return 2 * R * asin(sqrt(h))
	
	if not zones:
		return zones
	
	# Always keep the first (highest priority) zone
	keep = [zones[0]]
	
	# Add subsequent zones only if they're far enough from all kept zones
	for z in zones[1:]:
		c = (z["center_lat"], z["center_lon"])
		if all(mi(c, (w["center_lat"], w["center_lon"])) >= min_center_gap_mi 
		       for w in keep):
			keep.append(z)
	
	return keep


def _warn_duplicates(window_label, zones, eps_mi=0.05):
	"""Warn on duplicate or near-duplicate zones in a window.
	
	Checks for:
	- Exact duplicates (same center and radius)
	- Near-duplicates (centers too close together)
	
	Args:
		window_label: Window identifier string (e.g., "0-24").
		zones: List of zone dictionaries.
		eps_mi: Minimum distance threshold in miles for near-duplicate detection (default: 0.05).
	"""
	from math import radians, sin, cos, asin, sqrt
	R = 3958.7613  # Earth radius in miles
	
	# Check exact duplicates
	seen = set()
	dup = 0
	for z in zones:
		key = (round(z["center_lat"], 6), round(z["center_lon"], 6), 
		       round(z["radius_miles"], 2))
		if key in seen:
			dup += 1
		seen.add(key)
	
	# Check near-duplicates (too close centers)
	def _mi(a, b):
		dlat = radians(b[0] - a[0])
		dlon = radians(b[1] - a[1])
		h = sin(dlat/2)**2 + cos(radians(a[0])) * cos(radians(b[0])) * sin(dlon/2)**2
		return 2 * R * asin(sqrt(h))
	
	coords = [(z["center_lat"], z["center_lon"]) for z in zones]
	near = 0
	for i in range(len(coords)):
		for j in range(i+1, len(coords)):
			if _mi(coords[i], coords[j]) < eps_mi:
				near += 1
	
	if dup or near:
		print(f"[WARN] {window_label}: dup={dup}, near={near}, n={len(zones)}")


def run_one_with_propagation(
	case, cfg_path, outdir,
	grid_xy, hotspots, road_cost, seclusion, truth_adapter,
	corridor_index=None, in_state_mask=None,
	windows=("0-24", "24-48", "48-72"),
	time_buckets=("day", "night", "day"),
	topk=3,
	radius_mi=10.0,
	survival_weights=None,
	forecaster=None
):
	"""Execute RL episode with sequential belief propagation.
	
	Generates search zones by seeding from disappearance location and
	propagating belief state sequentially through time windows. This is
	an alternative to run_one() that uses propagation instead of independent
	window computation.
	
	Args:
		case: Case dictionary containing case_id and location data.
		cfg_path: Path to reward configuration JSON file.
		outdir: Output directory for results. Will be created if missing.
		grid_xy: Array of shape (N, 2) with (longitude, latitude) coordinates.
		hotspots: List of hotspot tuples (unused in propagation mode but kept for signature compatibility).
		road_cost: Array of shape (N,) with road accessibility costs.
		seclusion: Array of shape (N,) with seclusion scores.
		truth_adapter: Function that extracts truth points and computes metrics.
		corridor_index: Optional object for corridor matching.
		in_state_mask: Optional object for state boundary validation.
		windows: Tuple of window ID strings (default: ("0-24", "24-48", "48-72")).
		time_buckets: Tuple of time bucket strings for future use (default: ("day", "night", "day")).
		topk: Number of top zones per window (default: 3).
		radius_mi: Default zone radius in miles (default: 10.0).
		survival_weights: Optional dict mapping window IDs to survival weights.
		forecaster: Optional MovementForecaster instance (created if None).
		
	Returns:
		Tuple containing:
		- actions: Dictionary mapping window IDs to zone lists.
		- rew: Dictionary with window_scores and episode_score keys.
		
	The results are also written to outdir/zones_rl.jsonl as a JSONL record.
	"""
	import pathlib
	from reinforcement_learning.rl_env import ZoneRLEnv
	from reinforcement_learning.zone_rl import score_zones_per_window
	from reinforcement_learning.rewards import load_cfg
	
	# 1) Extract last-seen coordinates with fallback
	try:
		lat, lon = assert_seed_present(case)
		# Audit log: record seed coordinates for debugging
		case.setdefault("provenance", {})["debug_seed"] = {
			"last_seen_lat": lat,
			"last_seen_lon": lon
		}
	except (KeyError, Exception):
		# Fallback: use case_id if coordinates unavailable
		case_id = case.get("case_id", "unknown")
		case.setdefault("provenance", {})["debug_seed"] = {
			"case_id": case_id,
			"error": "missing_coords"
		}
		# For sigma calculation, use default coordinates (center of VA)
		lat, lon = 37.5, -78.5  # Default: center of Virginia
	case["provenance"]["generator"] = "propagate_v1"
	
	# 2) Initialize or reuse forecaster
	f = forecaster or MovementForecaster(grid_xy, road_cost, seclusion)
	
	# 3) Seed belief from disappearance point with geographic region-based sigma
	sigma_mi = choose_sigma_mi(lat, lon)
	pi = f.seed_from_point(lon, lat, sigma_mi=sigma_mi)
	
	# Optional profile-aware tweak if forecaster supports it
	if hasattr(f, "risk_adjust"):
		pi = _normalize(f.risk_adjust(pi, bucket=time_buckets[0] if time_buckets else None))
	
	# 4) Initialize evaluation infrastructure
	env = ZoneRLEnv(cfg_path)
	cfg = load_cfg(cfg_path)
	
	# Get survival weights from config if not provided
	if survival_weights is None:
		survival_weights = {w["id"]: w.get("weight", 1.0) for w in cfg.get("time_windows", [])}
		# Ensure all windows have weights
		for wid in windows:
			if wid not in survival_weights:
				survival_weights[wid] = 1.0
	
	# 5) Generate zones for each window with sequential propagation
	actions = {}
	
	for wl, bucket in zip(windows, time_buckets):
		if hasattr(f, "step"):
			pi = f.step(pi, bucket=bucket, steps=1)
		
		# Optional reweighting by dynamic signals (roads, seclusion, etc.)
		if hasattr(f, "risk_adjust"):
			pi = _normalize(f.risk_adjust(pi, bucket=bucket))
		
		# Apply survival weight for this window (earlier windows often heavier)
		if survival_weights and wl in survival_weights:
			pi = pi * float(survival_weights[wl])
			pi = _normalize(pi)
		
		# Choose top-K indices and form zones
		idx = np.argsort(-pi)[:topk]
		zones = []
		
		# Calculate priority sum for normalization
		p_sum = float(pi[idx].sum()) or 1.0
		
		for i in idx:
			lon_val = float(grid_xy[i, 0])
			lat_val = float(grid_xy[i, 1])
			
			z = {
				"zone_id": f"z{i:04d}",
				"center_lon": lon_val,
				"center_lat": lat_val,
				"radius_miles": float(radius_mi),
				"priority": float(pi[i] / p_sum),  # Normalized priority
			}
			
			# Enrich for rewards
			if corridor_index:
				z["corridor"] = corridor_index.nearest_corridor(lon_val, lat_val, z["radius_miles"])
			else:
				z["corridor"] = None
			
			if in_state_mask:
				z["in_state"] = bool(in_state_mask.contains_circle(lat_val, lon_val, z["radius_miles"]))
			else:
				z["in_state"] = True
			
			# Add seed reference for debugging 
			z["seed_ref"] = case["provenance"]["debug_seed"].copy()
			
			zones.append(z)
		
		# Warn on duplicate zones
		_warn_duplicates(wl, zones)
		
		# Apply non-max suppression to remove zones with centers too close
		zones = nms_by_center(zones, min_center_gap_mi=0.25)
		
		actions[wl] = zones
	
	# 6) Evaluate with existing infrastructure
	truth_by_window = truth_adapter(case, actions)
	rew = env.step(actions, truth_by_window)
	
	# 7) Score zones per window
	zone_scores_by_window = {}
	for w in env.windows:
		wid = w["id"]
		zlist = actions.get(wid, [])
		
		if not zlist:
			zone_scores_by_window[wid] = []
			continue
		
		d = truth_by_window[wid]["d_center_true_miles"]
		h = truth_by_window[wid]["t_hit_hr"]
		c = truth_by_window[wid].get("corridor_match", [False] * len(zlist))
		
		zone_scores_by_window[wid] = score_zones_per_window(zlist, w, d, h, c, cfg)
	
	# 8) Log results
	out = pathlib.Path(outdir)
	out.mkdir(parents=True, exist_ok=True)
	
	record = {
		"case_id": case["case_id"],
		"zones": actions,
		"zone_scores": zone_scores_by_window,
		"reward": rew
	}
	
	with open(out / "zones_rl.jsonl", "a", encoding="utf-8") as fw:
		fw.write(json.dumps(record) + "\n")
	
	return actions, rew


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description="Generate RL search zones from movement model predictions")
	parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "propagate"],
	                    help="Zone generation mode: 'baseline' uses existing run_one(), 'propagate' uses sequential propagation")
	parser.add_argument("--config", type=str, default="reinforcement_learning/search_reward_config.json",
	                    help="Path to reward configuration JSON file")
	parser.add_argument("--outdir", type=str, default="eda_out",
	                    help="Output directory for results")
	parser.add_argument("--log-debug", action="store_true",
	                    help="Enable debug logging and diagnostics")
	parser.add_argument("--sample", type=int, default=0,
	                    help="Sample N cases for quick testing (0 = all cases)")
	args = parser.parse_args()
	
	cfg = args.config
	outdir = args.outdir
	grid_xy, hotspots, road_cost, seclusion, va_mask, corridor_score = load_grid_and_layers()
	
	corridor_index = GeoCorridorIndex("data/geo/va_corridors.geojson", id_field="id")
	in_state_mask = GeoStateMask("data/geo/va_boundary.geojson")
	
	# Create shared forecaster for propagation mode (reused across cases)
	forecaster = None
	if args.mode == "propagate":
		forecaster = MovementForecaster(grid_xy, road_cost, seclusion)
	
	case_files = list(glob.glob("data/synthetic_cases/GRD-*.json"))
	if args.sample > 0:
		case_files = case_files[:args.sample]
		print(f"[INFO] Processing {len(case_files)} case files (sampled from {len(list(glob.glob('data/synthetic_cases/GRD-*.json')))}...")
	else:
		print(f"[INFO] Processing {len(case_files)} case files...")
	
	for fp in case_files:
		case = json.load(open(fp, "r", encoding="utf-8"))
		
		if args.mode == "baseline":
			actions, rew = run_one(case, cfg, outdir, grid_xy, hotspots, road_cost, seclusion, truth_adapter,
			                      corridor_index=corridor_index, in_state_mask=in_state_mask)
		elif args.mode == "propagate":
			# Define windows and time buckets
			windows = ("0-24", "24-48", "48-72")
			time_buckets = ("day", "night", "day")
			
			# Get survival weights from config
			from reinforcement_learning.rewards import load_cfg
			cfg_obj = load_cfg(cfg)
			survival_weights = {w["id"]: w.get("weight", 1.0) for w in cfg_obj.get("time_windows", [])}
			
			actions, rew = run_one_with_propagation(
				case, cfg, outdir, grid_xy, hotspots, road_cost, seclusion, truth_adapter,
				corridor_index=corridor_index, in_state_mask=in_state_mask,
				windows=windows, time_buckets=time_buckets,
				topk=3, radius_mi=10.0,
				survival_weights=survival_weights, forecaster=forecaster
			)
		else:
			raise ValueError(f"Unknown mode: {args.mode}")
		
		print(f"{case['case_id']} episode_return={rew['episode_score']:.3f}")


