"""Zone selection and scoring for reinforcement learning search operations.

Implements the core RL workflow for generating search zones from risk maps,
scoring zone performance, and executing complete episodes with reward
computation and result persistence.
"""

import json
import pathlib
import numpy as np

from .movement_model import risk_map
from .rl_env import ZoneRLEnv
from .rewards import load_cfg, hybrid_score, radius_penalty


def topk_zones(grid_xy, R, K: int = 3, radius_mi: float = 10.0, corridor_index=None, in_state_mask=None):
	"""Select top K zones from risk map based on priority scores.

	Extracts the K grid cells with highest risk scores and converts them
	into zone dictionaries with metadata. Optionally enriches zones with
	corridor matching and state boundary validation if the corresponding
	index/mask objects are provided.

	Args:
		grid_xy: Array of shape (N, 2) containing (longitude, latitude) coordinates.
		R: Array of shape (N,) containing risk scores for each grid cell.
		K: Number of top zones to select (default: 3).
		radius_mi: Search radius in miles for each zone (default: 10.0).
		corridor_index: Optional object with nearest_corridor() method for
			corridor matching. If None, corridor field is set to None.
		in_state_mask: Optional object with contains_circle() method for
			state boundary validation. If None, in_state field defaults to True.

	Returns:
		List of K zone dictionaries, each containing:
		- zone_id: Unique identifier string
		- center_lon: Zone center longitude
		- center_lat: Zone center latitude
		- radius_miles: Search radius
		- priority: Risk score from input array
		- corridor: Corridor identifier string or None
		- in_state: Boolean indicating state boundary containment
	"""
	idx = np.argsort(-R)[:K]

	zones = []

	for i in idx:
		lon, lat = float(grid_xy[i, 0]), float(grid_xy[i, 1])

		z = dict(zone_id=f"z{i:04d}", center_lon=lon, center_lat=lat,
		         radius_miles=radius_mi, priority=float(R[i]))

		z["corridor"] = None
		z["in_state"] = True

		if corridor_index is not None:
			z["corridor"] = corridor_index.nearest_corridor(lon, lat, radius_mi)

		if in_state_mask is not None:
			z["in_state"] = bool(in_state_mask.contains_circle(lat, lon, radius_mi))

		zones.append(z)

	return zones


def score_zones_per_window(zones, cfg_window, dists, hits, corridor_flags, cfg):
	"""Compute individual zone scores within a time window.

	Computes reward scores for each zone based on distance to truth, hit time,
	corridor matching, radius penalty, and state boundary validation. Applies
	bonuses and penalties according to the reward configuration.

	Args:
		zones: List of zone dictionaries with radius_miles and in_state keys.
		cfg_window: Window configuration dictionary with start_hr and end_hr keys.
		dists: List of minimum distances (miles) from zone centers to truth points.
		hits: List of earliest hit times (hours) or None if no hit.
		corridor_flags: List of booleans indicating corridor matches.
		cfg: Full reward configuration dictionary.

	Returns:
		List of zone scores, one per zone in input order.
	"""
	hy = cfg["reward_structures"]["zone_level"]["hybrid"]["parameters"]
	reg = cfg["reward_structures"]["zone_level"]["regularizers"]
	ws, we = cfg_window["start_hr"], cfg_window["end_hr"]

	out = []

	for i, z in enumerate(zones):
		add = {"corridor_bonus": 0.05} if (corridor_flags and corridor_flags[i]) else {}

		s = hybrid_score(dists[i], z["radius_miles"], hits[i], ws, we, {**hy, **add})
		s += radius_penalty(z["radius_miles"], reg["radius_penalty"]["parameters"])

		if z.get("in_state") is False:
			s += reg["out_of_state"]["penalty_value"]

		out.append(s)

	return out


def run_one(case_obj, cfg_path, outdir, grid_xy, hotspots, road_cost, seclusion, truth_adapter, corridor_index=None, in_state_mask=None):
	"""Execute a complete RL episode for a single case.

	Generates search zones for each time window using risk maps, evaluates
	them against ground truth, computes rewards, and persists results to disk.
	This function orchestrates the full RL workflow from risk map computation
	through reward calculation.

	Args:
		case_obj: Case dictionary containing case_id and other case data.
		cfg_path: Path to reward configuration JSON file.
		outdir: Output directory for results. Will be created if missing.
		grid_xy: Array of shape (N, 2) with (longitude, latitude) coordinates.
		hotspots: List of hotspot tuples for movement model.
		road_cost: Array of shape (N,) with road accessibility costs.
		seclusion: Array of shape (N,) with seclusion scores.
		truth_adapter: Function that extracts truth points and computes metrics.
		corridor_index: Optional object for corridor matching.
		in_state_mask: Optional object for state boundary validation.

	Returns:
		Tuple containing:
		- actions: Dictionary mapping window IDs to zone lists.
		- rew: Dictionary with window_scores and episode_score keys.

	The results are also written to outdir/zones_rl.jsonl as a JSONL record.
	"""
	env = ZoneRLEnv(cfg_path)
	cfg = load_cfg(cfg_path)
	t_mid = {w["id"]: (w["start_hr"] + w["end_hr"]) / 2.0 for w in cfg["time_windows"]}

	case_prior = None
	seed_ref = None
	try:
		from reinforcement_learning.build_rl_zones import get_last_seen, choose_sigma_mi, _seed_from_disappearance
		lat, lon = get_last_seen(case_obj)
		sigma_mi = choose_sigma_mi(lat, lon)
		case_prior = _seed_from_disappearance(grid_xy, lon, lat, sigma_mi=sigma_mi)
		seed_ref = {"last_seen_lat": lat, "last_seen_lon": lon}
	except (KeyError, Exception):
		case_prior = None
		case_id = case_obj.get("case_id", "unknown")
		seed_ref = {"case_id": case_id, "error": "missing_coords"}

	actions = {wid: topk_zones(grid_xy, risk_map(grid_xy, hotspots, road_cost, seclusion, th, init=case_prior), K=3,
	                          corridor_index=corridor_index, in_state_mask=in_state_mask)
	          for wid, th in t_mid.items()}
	
	from reinforcement_learning.build_rl_zones import nms_by_center
	for wid, zones in actions.items():
		actions[wid] = nms_by_center(zones, min_center_gap_mi=0.25)

	truth_by_window = truth_adapter(case_obj, actions)
	rew = env.step(actions, truth_by_window)

	for wid, zones in actions.items():
		for z in zones:
			z["seed_ref"] = seed_ref.copy()

	cfg = load_cfg(cfg_path)
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

	out = pathlib.Path(outdir)
	out.mkdir(parents=True, exist_ok=True)

	record = {
		"case_id": case_obj["case_id"],
		"zones": actions,
		"zone_scores": zone_scores_by_window,
		"reward": rew
	}

	with open(out / "zones_rl.jsonl", "a", encoding="utf-8") as fw:
		fw.write(json.dumps(record) + "\n")

	return actions, rew


