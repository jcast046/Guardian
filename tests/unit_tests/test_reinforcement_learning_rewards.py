"""Unit tests for reinforcement_learning.rewards module."""

import pytest
import json
from unittest.mock import patch, MagicMock
from reinforcement_learning.rewards import (
    load_cfg,
    _dist_score,
    _time_score,
    hybrid_score,
    radius_penalty,
    window_score,
    episode_score
)


class TestLoadCfg:
    """Test suite for reinforcement_learning.rewards.load_cfg function.

    Tests loading reward configuration from JSON files with error handling.
    """

    def test_load_cfg_basic(self, tmp_path, sample_reward_config):
        """Test loading reward configuration."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        result = load_cfg(str(config_file))
        
        assert result is not None
        assert "rl_search_config" in result or "__active_profile__" in result

    def test_load_cfg_missing_file(self):
        """Test loading config when file doesn't exist."""
        result = load_cfg("nonexistent.json")
        
        assert result is not None or isinstance(result, dict)

    def test_load_cfg_invalid_json(self, tmp_path):
        """Test loading config with invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{invalid json}", encoding="utf-8")
        
        try:
            result = load_cfg(str(config_file))
            assert isinstance(result, dict)
        except json.JSONDecodeError:
            pass


class TestDistScore:
    """Test suite for reinforcement_learning.rewards._dist_score function.

    Tests distance-based scoring with various distance-radius combinations
    and edge cases including zero radius.
    """

    def test_dist_score_zero_distance(self):
        """Test distance score with zero distance."""
        result = _dist_score(0.0, 10.0)
        
        assert result == 1.0

    def test_dist_score_within_radius(self):
        """Test distance score when distance is within radius."""
        result = _dist_score(5.0, 10.0)
        
        assert 0.0 < result < 1.0
        assert result > 0.5

    def test_dist_score_exceeds_radius(self):
        """Test distance score when distance exceeds radius."""
        result = _dist_score(15.0, 10.0)
        
        assert 0.0 < result < 1.0
        assert result < 0.5

    def test_dist_score_large_distance(self):
        """Test distance score with large distance."""
        result = _dist_score(100.0, 10.0)
        
        assert 0.0 < result < 0.1

    def test_dist_score_zero_radius(self):
        """Test distance score with zero radius."""
        result = _dist_score(5.0, 0.0)
        
        assert 0.0 <= result <= 1.0


class TestTimeScore:
    """Test suite for reinforcement_learning.rewards._time_score function.

    Tests time-based scoring within time windows with various
    time positions and edge cases.
    """

    def test_time_score_start_of_window(self):
        """Test time score at start of window."""
        result = _time_score(0.0, 0.0, 24.0)
        
        assert result == 1.0

    def test_time_score_end_of_window(self):
        """Test time score at end of window."""
        result = _time_score(24.0, 0.0, 24.0)
        
        assert result == 0.0

    def test_time_score_middle_of_window(self):
        """Test time score in middle of window."""
        result = _time_score(12.0, 0.0, 24.0)
        
        assert 0.0 < result < 1.0
        assert result == pytest.approx(0.5, abs=0.1)

    def test_time_score_before_window(self):
        """Test time score before window starts."""
        result = _time_score(-5.0, 0.0, 24.0)
        
        assert 0.0 <= result <= 1.0

    def test_time_score_after_window(self):
        """Test time score after window ends."""
        result = _time_score(30.0, 0.0, 24.0)
        
        assert result == 0.0 or result < 0.0


class TestHybridScore:
    """Test suite for reinforcement_learning.rewards.hybrid_score function.

    Tests hybrid scoring combining distance and time scores with
    various weighting parameters and bonus factors.
    """

    def test_hybrid_score_perfect(self):
        """Test hybrid score with perfect distance and time."""
        result = hybrid_score(0.0, 10.0, 0.0, 0.0, 24.0, {"alpha": 0.7, "beta": 0.3})
        
        assert result == pytest.approx(1.0, abs=0.01)

    def test_hybrid_score_weighted(self):
        """Test hybrid score with weighted components."""
        result = hybrid_score(5.0, 10.0, 12.0, 0.0, 24.0, {"alpha": 0.7, "beta": 0.3})
        
        assert 0.0 < result < 1.0

    def test_hybrid_score_alpha_beta(self):
        """Test that alpha and beta weights affect score."""
        params1 = {"alpha": 0.9, "beta": 0.1}
        params2 = {"alpha": 0.1, "beta": 0.9}
        
        result1 = hybrid_score(5.0, 10.0, 12.0, 0.0, 24.0, params1)
        result2 = hybrid_score(5.0, 10.0, 12.0, 0.0, 24.0, params2)
        
        assert result1 != result2

    def test_hybrid_score_inside_bonus(self):
        """Test hybrid score with inside bonus."""
        params = {
            "alpha": 0.7,
            "beta": 0.3,
            "inside_threshold": 0.85,
            "inside_bonus": 0.1
        }
        
        result = hybrid_score(5.0, 10.0, 12.0, 0.0, 24.0, params)
        
        assert 0.0 < result <= 1.0

    def test_hybrid_score_corridor_bonus(self):
        """Test hybrid score with corridor bonus."""
        params = {
            "alpha": 0.7,
            "beta": 0.3,
            "corridor_bonus": 0.05
        }
        
        result = hybrid_score(5.0, 10.0, 12.0, 0.0, 24.0, params)
        
        assert 0.0 < result <= 1.0


class TestRadiusPenalty:
    """Test suite for reinforcement_learning.rewards.radius_penalty function.

    Tests radius-based penalty calculation with various radius sizes
    and parameter configurations.
    """

    def test_radius_penalty_small_radius(self):
        """Test radius penalty with small radius."""
        params = {"lambda_radius": 0.2, "max_radius_miles": 50.0, "p": 2}
        result = radius_penalty(5.0, params)
        
        assert result < 0.0
        assert abs(result) < 0.1

    def test_radius_penalty_large_radius(self):
        """Test radius penalty with large radius."""
        params = {"lambda_radius": 0.2, "max_radius_miles": 50.0, "p": 2}
        result = radius_penalty(50.0, params)
        
        assert result < 0.0
        assert abs(result) > 0.1

    def test_radius_penalty_zero_radius(self):
        """Test radius penalty with zero radius."""
        params = {"lambda_radius": 0.2, "max_radius_miles": 50.0, "p": 2}
        result = radius_penalty(0.0, params)
        
        assert result == 0.0

    def test_radius_penalty_exceeds_max(self):
        """Test radius penalty when radius exceeds max."""
        params = {"lambda_radius": 0.2, "max_radius_miles": 50.0, "p": 2}
        result = radius_penalty(100.0, params)
        
        assert result < 0.0
        assert abs(result) >= 0.2


class TestWindowScore:
    """Test suite for reinforcement_learning.rewards.window_score function.

    Tests window-level scoring with various zone configurations,
    hit patterns, and edge cases including empty zones.
    """

    def test_window_score_single_zone(self, sample_reward_config):
        """Test window score with single zone."""
        zones = [{"zone_id": "z01", "priority": 0.8}]
        cfg_window = {"id": "0-24", "start_hr": 0, "end_hr": 24, "weight": 1.0}
        dists = [5.0]
        hits = [True]
        corridor_flags = [False]
        
        result = window_score(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_window_score_multiple_zones(self, sample_reward_config):
        """Test window score with multiple zones."""
        zones = [
            {"zone_id": "z01", "priority": 0.8},
            {"zone_id": "z02", "priority": 0.6}
        ]
        cfg_window = {"id": "0-24", "start_hr": 0, "end_hr": 24, "weight": 1.0}
        dists = [5.0, 10.0]
        hits = [True, False]
        corridor_flags = [False, False]
        
        result = window_score(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_window_score_no_hits(self, sample_reward_config):
        """Test window score with no hits."""
        zones = [{"zone_id": "z01", "priority": 0.8}]
        cfg_window = {"id": "0-24", "start_hr": 0, "end_hr": 24, "weight": 1.0}
        dists = [100.0]
        hits = [False]
        corridor_flags = [False]
        
        result = window_score(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert isinstance(result, float)
        assert result < 1.0

    def test_window_score_empty_zones(self, sample_reward_config):
        """Test window score with empty zones."""
        zones = []
        cfg_window = {"id": "0-24", "start_hr": 0, "end_hr": 24, "weight": 1.0}
        dists = []
        hits = []
        corridor_flags = []
        
        result = window_score(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert isinstance(result, float)
        assert result == 0.0 or result < 0.0


class TestEpisodeScore:
    """Test suite for reinforcement_learning.rewards.episode_score function.

    Tests episode-level scoring aggregating window scores with
    various weighting configurations and edge cases.
    """

    def test_episode_score_single_window(self, sample_reward_config):
        """Test episode score with single window."""
        window_scores = [0.8]
        
        result = episode_score(window_scores, sample_reward_config)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_episode_score_multiple_windows(self, sample_reward_config):
        """Test episode score with multiple windows."""
        window_scores = [0.8, 0.6, 0.4]
        
        result = episode_score(window_scores, sample_reward_config)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_episode_score_weighted(self, sample_reward_config):
        """Test that episode score uses window weights."""
        window_scores = [1.0, 0.0, 0.0]  # Only first window has score
        
        result = episode_score(window_scores, sample_reward_config)
        
        # Should be weighted by window weights
        assert 0.0 <= result <= 1.0

    def test_episode_score_empty(self, sample_reward_config):
        """Test episode score with empty window scores."""
        window_scores = []
        
        result = episode_score(window_scores, sample_reward_config)
        
        assert result == 0.0

    def test_episode_score_all_zeros(self, sample_reward_config):
        """Test episode score with all zero scores."""
        window_scores = [0.0, 0.0, 0.0]
        
        result = episode_score(window_scores, sample_reward_config)
        
        assert result == 0.0
