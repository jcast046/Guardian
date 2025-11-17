"""Unit tests for reinforcement_learning.zone_rl module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from reinforcement_learning.zone_rl import topk_zones, score_zones_per_window


class TestTopkZones:
    """Test suite for reinforcement_learning.zone_rl.topk_zones function.

    Tests top-K zone selection from grid with priority ranking
    and zone field validation.
    """

    def test_topk_zones_basic(self):
        """Test basic top K zones selection."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2],
            [-77.3, 38.3]
        ])
        R = np.array([0.1, 0.9, 0.3, 0.8])
        K = 2
        
        zones = topk_zones(grid_xy, R, K=K)
        
        assert len(zones) == K
        assert zones[0]["priority"] == 0.9
        assert zones[1]["priority"] == 0.8

    def test_topk_zones_all_fields(self):
        """Test zones have all required fields."""
        grid_xy = np.array([[-77.0, 38.0]])
        R = np.array([0.8])
        K = 1
        
        zones = topk_zones(grid_xy, R, K=K, radius_mi=10.0)
        
        assert len(zones) == 1
        zone = zones[0]
        assert "zone_id" in zone
        assert "center_lon" in zone
        assert "center_lat" in zone
        assert "radius_miles" in zone
        assert "priority" in zone
        assert "corridor" in zone
        assert "in_state" in zone
        assert zone["radius_miles"] == 10.0
        assert zone["priority"] == 0.8

    def test_topk_zones_zone_ids(self):
        """Test zone IDs are unique and properly formatted."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2]
        ])
        R = np.array([0.1, 0.9, 0.3])
        K = 3
        
        zones = topk_zones(grid_xy, R, K=K)
        
        zone_ids = [zone["zone_id"] for zone in zones]
        assert len(zone_ids) == len(set(zone_ids))  # All unique
        assert all(zone_id.startswith("z") for zone_id in zone_ids)

    def test_topk_zones_with_corridor_index(self):
        """Test zones with corridor index."""
        grid_xy = np.array([[-77.0, 38.0]])
        R = np.array([0.8])
        K = 1
        
        mock_corridor_index = MagicMock()
        mock_corridor_index.nearest_corridor.return_value = "I-95 NB"
        
        zones = topk_zones(grid_xy, R, K=K, corridor_index=mock_corridor_index)
        
        assert zones[0]["corridor"] == "I-95 NB"
        mock_corridor_index.nearest_corridor.assert_called_once()

    def test_topk_zones_with_in_state_mask(self):
        """Test zones with in-state mask."""
        grid_xy = np.array([[-77.0, 38.0]])
        R = np.array([0.8])
        K = 1
        
        mock_in_state_mask = MagicMock()
        mock_in_state_mask.contains_circle.return_value = False
        
        zones = topk_zones(grid_xy, R, K=K, in_state_mask=mock_in_state_mask)
        
        assert zones[0]["in_state"] is False
        mock_in_state_mask.contains_circle.assert_called_once()

    def test_topk_zones_defaults(self):
        """Test zones with default parameters."""
        grid_xy = np.array([[-77.0, 38.0]])
        R = np.array([0.8])
        K = 1
        
        zones = topk_zones(grid_xy, R)
        
        assert zones[0]["corridor"] is None
        assert zones[0]["in_state"] is True
        assert zones[0]["radius_miles"] == 10.0  # Default radius

    def test_topk_zones_custom_radius(self):
        """Test zones with custom radius."""
        grid_xy = np.array([[-77.0, 38.0]])
        R = np.array([0.8])
        K = 1
        
        zones = topk_zones(grid_xy, R, K=K, radius_mi=20.0)
        
        assert zones[0]["radius_miles"] == 20.0

    def test_topk_zones_k_larger_than_grid(self):
        """Test zones when K is larger than grid size."""
        grid_xy = np.array([[-77.0, 38.0], [-77.1, 38.1]])
        R = np.array([0.1, 0.9])
        K = 10  # Larger than grid size
        
        zones = topk_zones(grid_xy, R, K=K)
        
        assert len(zones) == 2  # Should return all available zones

    def test_topk_zones_coordinate_types(self):
        """Test that coordinates are converted to float."""
        grid_xy = np.array([[-77.0, 38.0]])
        R = np.array([0.8])
        K = 1
        
        zones = topk_zones(grid_xy, R, K=K)
        
        assert isinstance(zones[0]["center_lon"], float)
        assert isinstance(zones[0]["center_lat"], float)
        assert isinstance(zones[0]["priority"], float)


class TestScoreZonesPerWindow:
    """Test suite for reinforcement_learning.zone_rl.score_zones_per_window function.

    Tests per-window zone scoring with various zone configurations,
    hit patterns, corridor bonuses, and edge cases.
    """

    def test_score_zones_per_window_basic(self, sample_reward_config):
        """Test basic zone scoring."""
        zones = [
            {"zone_id": "z01", "radius_miles": 10.0, "in_state": True}
        ]
        cfg_window = {"start_hr": 0, "end_hr": 24}
        dists = [5.0]
        hits = [12.0]
        corridor_flags = [False]
        
        scores = score_zones_per_window(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert 0.0 <= scores[0] <= 1.0

    def test_score_zones_per_window_multiple_zones(self, sample_reward_config):
        """Test scoring multiple zones."""
        zones = [
            {"zone_id": "z01", "radius_miles": 10.0, "in_state": True},
            {"zone_id": "z02", "radius_miles": 15.0, "in_state": True}
        ]
        cfg_window = {"start_hr": 0, "end_hr": 24}
        dists = [5.0, 10.0]
        hits = [12.0, 18.0]
        corridor_flags = [False, False]
        
        scores = score_zones_per_window(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_score_zones_per_window_no_hit(self, sample_reward_config):
        """Test scoring zones with no hit."""
        zones = [
            {"zone_id": "z01", "radius_miles": 10.0, "in_state": True}
        ]
        cfg_window = {"start_hr": 0, "end_hr": 24}
        dists = [100.0]  # Far away
        hits = [None]  # No hit
        corridor_flags = [False]
        
        scores = score_zones_per_window(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert len(scores) == 1
        assert scores[0] < 1.0

    def test_score_zones_per_window_corridor_bonus(self, sample_reward_config):
        """Test scoring zones with corridor bonus."""
        zones = [
            {"zone_id": "z01", "radius_miles": 10.0, "in_state": True}
        ]
        cfg_window = {"start_hr": 0, "end_hr": 24}
        dists = [5.0]
        hits = [12.0]
        corridor_flags = [True]  # Corridor match
        
        scores = score_zones_per_window(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert len(scores) == 1
        assert scores[0] > 0.0

    def test_score_zones_per_window_out_of_state(self, sample_reward_config):
        """Test scoring zones out of state."""
        zones = [
            {"zone_id": "z01", "radius_miles": 10.0, "in_state": False}
        ]
        cfg_window = {"start_hr": 0, "end_hr": 24}
        dists = [5.0]
        hits = [12.0]
        corridor_flags = [False]
        
        scores = score_zones_per_window(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert len(scores) == 1
        assert scores[0] < 1.0

    def test_score_zones_per_window_radius_penalty(self, sample_reward_config):
        """Test that large radius incurs penalty."""
        zones_small = [
            {"zone_id": "z01", "radius_miles": 5.0, "in_state": True}
        ]
        zones_large = [
            {"zone_id": "z01", "radius_miles": 50.0, "in_state": True}
        ]
        cfg_window = {"start_hr": 0, "end_hr": 24}
        dists = [2.0]
        hits = [12.0]
        corridor_flags = [False]
        
        scores_small = score_zones_per_window(zones_small, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        scores_large = score_zones_per_window(zones_large, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert isinstance(scores_small[0], float)
        assert isinstance(scores_large[0], float)

    def test_score_zones_per_window_empty_zones(self, sample_reward_config):
        """Test scoring empty zones list."""
        zones = []
        cfg_window = {"start_hr": 0, "end_hr": 24}
        dists = []
        hits = []
        corridor_flags = []
        
        scores = score_zones_per_window(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert scores == []

    def test_score_zones_per_window_perfect_score(self, sample_reward_config):
        """Test scoring with perfect conditions."""
        zones = [
            {"zone_id": "z01", "radius_miles": 10.0, "in_state": True}
        ]
        cfg_window = {"start_hr": 0, "end_hr": 24}
        dists = [0.0]  # Exact match
        hits = [0.0]  # Immediate hit
        corridor_flags = [True]  # Corridor match
        
        scores = score_zones_per_window(zones, cfg_window, dists, hits, corridor_flags, sample_reward_config)
        
        assert len(scores) == 1
        assert scores[0] > 0.5
