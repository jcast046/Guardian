"""Reward computation for reinforcement learning search zone evaluation.

Implements hierarchical reward functions for evaluating search zones:
- Zone-level scoring based on distance, hit time, and bonuses
- Window-level aggregation with overlap penalties
- Episode-level weighted combination across time windows

Reward structures are configurable via JSON configuration files.
"""

import json
import math


def load_cfg(p):
	"""Load reward configuration from JSON file.

	Args:
		p: Path to configuration JSON file.

	Returns:
		dict: Dictionary containing rl_search_config section from JSON file.

	Raises:
		FileNotFoundError: If configuration file does not exist.
		KeyError: If rl_search_config key is missing from JSON file.
	"""
	return json.load(open(p, "r", encoding="utf-8"))["rl_search_config"]


def _dist_score(d_mi, r_mi):
	"""Compute distance-based score for zone evaluation.

	Computes a score based on how close the truth point is to the zone center.
	Points inside the zone radius get full score, with score decreasing as
	distance increases beyond the radius.

	Args:
		d_mi: Distance from zone center to truth point in miles.
		r_mi: Zone radius in miles.

	Returns:
		Score between 0 and 1, where 1.0 indicates point is at or within radius.
	"""
	return 1.0 / (1.0 + max(0.0, d_mi - r_mi))


def _time_score(t_hit, ws, we):
	"""Compute time-based score for zone evaluation.

	Rewards earlier hits within the time window. If no hit occurs, returns
	zero score. The score is normalized by window duration.

	Args:
		t_hit: Hit time in hours, or None if no hit occurred.
		ws: Window start time in hours.
		we: Window end time in hours.

	Returns:
		Score between 0 and 1, where 1.0 indicates hit at window start and
		0.0 indicates no hit or hit at window end.
	"""
	denom = max(1e-6, we - ws)
	return max(0.0, (we - (t_hit if t_hit is not None else we)) / denom)


def hybrid_score(d_mi, r_mi, t_hit, ws, we, prm):
	"""Compute hybrid reward score combining distance and time components.

	Combines distance-based and time-based scores with configurable weights.
	Applies bonuses for points well inside the zone and for corridor matches.

	Args:
		d_mi: Distance from zone center to truth point in miles.
		r_mi: Zone radius in miles.
		t_hit: Hit time in hours, or None if no hit occurred.
		ws: Window start time in hours.
		we: Window end time in hours.
		prm: Parameter dictionary containing:
			- alpha: Weight for distance component (default: 0.7)
			- beta: Weight for time component (default: 0.3)
			- inside_bonus: Bonus for points well inside zone (default: 0.1)
			- inside_threshold: Distance score threshold for bonus (default: 0.85)
			- corridor_bonus: Bonus for corridor matches (default: 0.0)

	Returns:
		Hybrid reward score combining all components.
	"""
	inside = _dist_score(d_mi, r_mi)

	return (prm.get("alpha", 0.7) * inside
	        + prm.get("beta", 0.3) * _time_score(t_hit, ws, we)
	        + (prm.get("inside_bonus", 0.1) if inside >= prm.get("inside_threshold", 0.85) else 0.0)
	        + prm.get("corridor_bonus", 0.0))


def radius_penalty(r_mi, prm):
	"""Compute penalty for zone radius size.

	Penalizes larger zones to encourage more focused search areas. The penalty
	scales with radius raised to a power, normalized by a maximum radius.

	Args:
		r_mi: Zone radius in miles.
		prm: Parameter dictionary containing:
			- lambda_radius: Penalty scaling factor (default: 0.2)
			- max_radius_miles: Maximum radius for normalization (default: 50.0)
			- p: Power exponent for penalty curve (default: 2)

	Returns:
		Negative penalty value (more negative for larger radii).
	"""
	lam = prm.get("lambda_radius", 0.2)
	maxr = prm.get("max_radius_miles", 50.0)
	p = prm.get("p", 2)

	return -lam * ((r_mi / maxr) ** p)


def window_score(zones, cfg_window, dists, hits, corridor_flags, cfg):
	"""Compute aggregate window-level score from individual zone scores.

	Computes zone-level scores, then aggregates them using softmax weighting
	based on zone priorities. Applies penalties for zone overlap and wasted
	zones (zones that miss with large distances).

	Args:
		zones: List of zone dictionaries with radius_miles, in_state, and
			priority keys.
		cfg_window: Window configuration dictionary with start_hr, end_hr,
			and id keys.
		dists: List of minimum distances (miles) from zone centers to truth points.
		hits: List of earliest hit times (hours) or None if no hit.
		corridor_flags: List of booleans indicating corridor matches.
		cfg: Full reward configuration dictionary.

	Returns:
		Aggregate window score combining weighted zone scores and penalties.
	"""
	import itertools
	from src.geography.distance import haversine_distance

	hy = cfg["reward_structures"]["zone_level"]["hybrid"]["parameters"]
	reg = cfg["reward_structures"]["zone_level"]["regularizers"]
	ws, we = cfg_window["start_hr"], cfg_window["end_hr"]

	zs = []

	for i, z in enumerate(zones):
		add = {"corridor_bonus": 0.05} if (corridor_flags and corridor_flags[i]) else {}

		s = hybrid_score(dists[i], z["radius_miles"], hits[i], ws, we, {**hy, **add})
		s += radius_penalty(z["radius_miles"], reg["radius_penalty"]["parameters"])

		if z.get("in_state") is False:
			s += reg["out_of_state"]["penalty_value"]

		zs.append(s)

	pri = [z.get("priority", 0.0) for z in zones]
	m = max(pri) if pri else 0.0

	if m == 0.0 and len(pri) > 0:
		pri = [p + 1e-6 for p in pri]
		m = max(pri)

	w = [math.exp(p - m) for p in pri]
	Z = sum(w) + 1e-9

	base = sum((w_i / Z) * s_i for w_i, s_i in zip(w, zs))

	def _pair_overlap_mi(z1, z2):
		"""Compute overlap fraction between two circular zones."""
		d = haversine_distance(z1["center_lat"], z1["center_lon"], z2["center_lat"], z2["center_lon"])
		r1, r2 = z1["radius_miles"], z2["radius_miles"]
		if d >= r1 + r2:
			return 0.0
		if d <= min(r1, r2):
			return 1.0
		return max(0.0, 1.0 - (d - min(r1, r2)) / max(1e-6, (r1 + r2 - min(r1, r2))))

	pairs = list(itertools.combinations(zones, 2))
	mean_overlap = sum(_pair_overlap_mi(a, b) for a, b in pairs) / max(1, len(pairs))
	ov_cfg = cfg["reward_structures"]["window_aggregation"]["penalties"]["overlap_penalty"]["parameters"]
	lam_ov = ov_cfg.get("lambda_overlap", 0.1)
	base -= lam_ov * mean_overlap

	wz = cfg["reward_structures"]["window_aggregation"]["penalties"]["wasted_zone"]
	thr = wz["parameters"]["threshold_to_true_by_window"][cfg_window["id"]]
	wasted = sum(wz["penalty_value"] for d, h in zip(dists, hits) if (h is None and d > thr))

	return base + wasted


def episode_score(window_scores, cfg):
	"""Compute aggregate episode score from window scores.

	Combines window-level scores using configurable weights. Each time window
	contributes to the final episode score proportional to its weight.

	Args:
		window_scores: Dictionary mapping window IDs to window scores.
		cfg: Configuration dictionary containing time_windows with id and
			weight keys.

	Returns:
		Weighted sum of window scores.
	"""
	weights = {w["id"]: w["weight"] for w in cfg["time_windows"]}

	return sum(weights[k] * v for k, v in window_scores.items())


