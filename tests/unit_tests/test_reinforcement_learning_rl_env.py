"""Unit tests for reinforcement_learning.rl_env module."""

import pytest
import json
from unittest.mock import patch, MagicMock
from reinforcement_learning.rl_env import ZoneRLEnv


class TestZoneRLEnv:
    """Test suite for reinforcement_learning.rl_env.ZoneRLEnv class.

    Tests reinforcement learning environment for zone-based search
    including initialization, step function, and multiple window handling.
    """

    def test_init(self, tmp_path, sample_reward_config):
        """Test ZoneRLEnv initialization."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        env = ZoneRLEnv(str(config_file))
        
        assert env.cfg is not None
        assert "time_windows" in env.cfg or "rl_search_config" in env.cfg
        assert env.windows is not None

    def test_step_single_window(self, tmp_path, sample_reward_config):
        """Test step function with single window."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        env = ZoneRLEnv(str(config_file))
        
        actions_by_window = {
            "0-24": [
                {"zone_id": "z01", "priority": 0.8, "radius_miles": 10.0, "in_state": True}
            ]
        }
        truth_by_window = {
            "0-24": {
                "d_center_true_miles": [5.0],
                "t_hit_hr": [12.0],
                "corridor_match": [False]
            }
        }
        
        result = env.step(actions_by_window, truth_by_window)
        
        assert "window_scores" in result
        assert "episode_score" in result
        assert "0-24" in result["window_scores"]
        assert isinstance(result["episode_score"], float)

    def test_step_multiple_windows(self, tmp_path, sample_reward_config):
        """Test step function with multiple windows."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        env = ZoneRLEnv(str(config_file))
        
        actions_by_window = {
            "0-24": [
                {"zone_id": "z01", "priority": 0.8, "radius_miles": 10.0, "in_state": True}
            ],
            "24-48": [
                {"zone_id": "z02", "priority": 0.6, "radius_miles": 15.0, "in_state": True}
            ]
        }
        truth_by_window = {
            "0-24": {
                "d_center_true_miles": [5.0],
                "t_hit_hr": [12.0],
                "corridor_match": [False]
            },
            "24-48": {
                "d_center_true_miles": [10.0],
                "t_hit_hr": [36.0],
                "corridor_match": [False]
            }
        }
        
        result = env.step(actions_by_window, truth_by_window)
        
        assert "window_scores" in result
        assert "episode_score" in result
        assert "0-24" in result["window_scores"]
        assert "24-48" in result["window_scores"]
        assert isinstance(result["episode_score"], float)

    def test_step_empty_zones(self, tmp_path, sample_reward_config):
        """Test step function with empty zones."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        env = ZoneRLEnv(str(config_file))
        
        actions_by_window = {
            "0-24": []
        }
        truth_by_window = {
            "0-24": {
                "d_center_true_miles": [],
                "t_hit_hr": [],
                "corridor_match": []
            }
        }
        
        result = env.step(actions_by_window, truth_by_window)
        
        assert "window_scores" in result
        assert result["window_scores"]["0-24"] == 0.0

    def test_step_missing_window(self, tmp_path, sample_reward_config):
        """Test step function with missing window in actions."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        env = ZoneRLEnv(str(config_file))
        
        actions_by_window = {}  # No windows
        truth_by_window = {
            "0-24": {
                "d_center_true_miles": [5.0],
                "t_hit_hr": [12.0],
                "corridor_match": [False]
            }
        }
        
        result = env.step(actions_by_window, truth_by_window)
        
        assert "window_scores" in result
        # Missing window should have score 0.0
        assert result["window_scores"].get("0-24", 0.0) == 0.0

    def test_step_multiple_zones_per_window(self, tmp_path, sample_reward_config):
        """Test step function with multiple zones per window."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        env = ZoneRLEnv(str(config_file))
        
        actions_by_window = {
            "0-24": [
                {"zone_id": "z01", "priority": 0.8, "radius_miles": 10.0, "in_state": True},
                {"zone_id": "z02", "priority": 0.6, "radius_miles": 15.0, "in_state": True}
            ]
        }
        truth_by_window = {
            "0-24": {
                "d_center_true_miles": [5.0, 10.0],
                "t_hit_hr": [12.0, 18.0],
                "corridor_match": [False, False]
            }
        }
        
        result = env.step(actions_by_window, truth_by_window)
        
        assert "window_scores" in result
        assert "episode_score" in result
        assert isinstance(result["window_scores"]["0-24"], float)

    def test_step_no_hit_times(self, tmp_path, sample_reward_config):
        """Test step function with no hit times (None values)."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        env = ZoneRLEnv(str(config_file))
        
        actions_by_window = {
            "0-24": [
                {"zone_id": "z01", "priority": 0.8, "radius_miles": 10.0, "in_state": True}
            ]
        }
        truth_by_window = {
            "0-24": {
                "d_center_true_miles": [100.0],  # Far away
                "t_hit_hr": [None],  # No hit
                "corridor_match": [False]
            }
        }
        
        result = env.step(actions_by_window, truth_by_window)
        
        assert "window_scores" in result
        assert "episode_score" in result
        # Score should be lower with no hits
        assert result["window_scores"]["0-24"] < 1.0

    def test_step_corridor_match(self, tmp_path, sample_reward_config):
        """Test step function with corridor matches."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        env = ZoneRLEnv(str(config_file))
        
        actions_by_window = {
            "0-24": [
                {"zone_id": "z01", "priority": 0.8, "radius_miles": 10.0, "in_state": True}
            ]
        }
        truth_by_window = {
            "0-24": {
                "d_center_true_miles": [5.0],
                "t_hit_hr": [12.0],
                "corridor_match": [True]  # Corridor match
            }
        }
        
        result = env.step(actions_by_window, truth_by_window)
        
        assert "window_scores" in result
        # Corridor match should increase score
        assert result["window_scores"]["0-24"] > 0.0

    def test_step_out_of_state(self, tmp_path, sample_reward_config):
        """Test step function with out-of-state zones."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        env = ZoneRLEnv(str(config_file))
        
        actions_by_window = {
            "0-24": [
                {"zone_id": "z01", "priority": 0.8, "radius_miles": 10.0, "in_state": False}
            ]
        }
        truth_by_window = {
            "0-24": {
                "d_center_true_miles": [5.0],
                "t_hit_hr": [12.0],
                "corridor_match": [False]
            }
        }
        
        result = env.step(actions_by_window, truth_by_window)
        
        assert "window_scores" in result
        # Out-of-state should have penalty
        assert result["window_scores"]["0-24"] < 1.0
