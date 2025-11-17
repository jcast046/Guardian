"""Reinforcement learning environment for search zone evaluation.

Provides the ZoneRLEnv class, which implements the RL environment interface
for evaluating search zones across multiple time windows.
"""

from .rewards import load_cfg, window_score, episode_score


class ZoneRLEnv:
	"""Reinforcement learning environment for search zone evaluation.

	Evaluates search zones across multiple time windows by computing reward
	scores based on zone placement, truth point distances, and hit times.
	Aggregates window-level scores into an episode-level score.
	"""

	def __init__(self, cfg_path):
		"""Initialize the RL environment with configuration.

		Args:
			cfg_path: Path to reward configuration JSON file.
		"""
		self.cfg = load_cfg(cfg_path)
		self.windows = self.cfg["time_windows"]

	def step(self, actions_by_window, truth_by_window):
		"""Evaluate search zones and compute rewards.

		Computes window-level scores for each time window based on zone
		performance, then aggregates them into an episode score.

		Args:
			actions_by_window: Dictionary mapping window IDs to lists of
				zone dictionaries.
			truth_by_window: Dictionary mapping window IDs to dictionaries
				containing:
				- d_center_true_miles: List of minimum distances (miles)
				- t_hit_hr: List of hit times (hours) or None
				- corridor_match: List of corridor match booleans

		Returns:
			Dictionary containing:
			- window_scores: Dictionary mapping window IDs to window scores
			- episode_score: Aggregate episode score across all windows
		"""
		w_scores = {}

		for w in self.windows:
			wid = w["id"]
			zones = actions_by_window.get(wid, [])

			if not zones:
				w_scores[wid] = 0.0
				continue

			d = truth_by_window[wid]["d_center_true_miles"]
			h = truth_by_window[wid]["t_hit_hr"]
			c = truth_by_window[wid].get("corridor_match", [False] * len(zones))

			w_scores[wid] = window_score(zones, w, d, h, c, self.cfg)

		return {"window_scores": w_scores, "episode_score": episode_score(w_scores, self.cfg)}


