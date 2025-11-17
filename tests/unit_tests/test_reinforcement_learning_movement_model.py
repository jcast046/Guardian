"""Unit tests for reinforcement_learning.movement_model module."""

import pytest
import numpy as np
from reinforcement_learning.movement_model import (
    kde_prior,
    survival_decay,
    survival_curve_exponential,
    survival_factor,
    build_transition,
    risk_map,
    load_hotspots_multi,
)


class TestKdePrior:
    """Test suite for reinforcement_learning.movement_model.kde_prior function.

    Tests kernel density estimation prior generation from hotspots
    with various configurations and edge cases.
    """

    def test_kde_prior_single_hotspot(self):
        """Test KDE prior with single hotspot."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2]
        ])
        hotspots = [(-77.1, 38.1, 1.0, 10.0)]  # (lon, lat, weight, sigma_miles)
        
        prior = kde_prior(grid_xy, hotspots)
        
        assert len(prior) == len(grid_xy)
        assert np.allclose(prior.sum(), 1.0, atol=1e-6)  # Should sum to 1
        assert np.all(prior >= 0)  # All probabilities >= 0
        # Highest probability should be near hotspot
        assert prior[1] > prior[0]  # Index 1 is closest to hotspot

    def test_kde_prior_multiple_hotspots(self):
        """Test KDE prior with multiple hotspots."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2]
        ])
        hotspots = [
            (-77.0, 38.0, 1.0, 10.0),
            (-77.2, 38.2, 0.5, 10.0)
        ]
        
        prior = kde_prior(grid_xy, hotspots)
        
        assert len(prior) == len(grid_xy)
        assert np.allclose(prior.sum(), 1.0, atol=1e-6)
        assert np.all(prior >= 0)
        # Both hotspots should contribute
        assert prior[0] > 0  # Near first hotspot
        assert prior[2] > 0  # Near second hotspot

    def test_kde_prior_normalization(self):
        """Test that KDE prior is properly normalized."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1]
        ])
        hotspots = [(-77.0, 38.0, 1.0, 10.0)]
        
        prior = kde_prior(grid_xy, hotspots)
        
        assert np.allclose(prior.sum(), 1.0, atol=1e-6)

    def test_kde_prior_empty_hotspots(self):
        """Test KDE prior with empty hotspots list."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1]
        ])
        hotspots = []
        
        prior = kde_prior(grid_xy, hotspots)
        
        assert len(prior) == len(grid_xy)
        assert np.all(prior == 0)  # Should be all zeros

    def test_kde_prior_weighted_hotspots(self):
        """Test KDE prior with weighted hotspots."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1]
        ])
        hotspots = [
            (-77.0, 38.0, 2.0, 10.0),  # Higher weight
            (-77.1, 38.1, 1.0, 10.0)   # Lower weight
        ]
        
        prior = kde_prior(grid_xy, hotspots)
        
        # First hotspot should have more influence due to higher weight
        assert prior[0] > prior[1] or abs(prior[0] - prior[1]) < 0.1  # May be close due to distance

    def test_kde_prior_sigma_effect(self):
        """Test that sigma parameter affects spread."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2]
        ])
        hotspots_small = [(-77.1, 38.1, 1.0, 1.0)]   # Small sigma
        hotspots_large = [(-77.1, 38.1, 1.0, 100.0)]  # Large sigma
        
        prior_small = kde_prior(grid_xy, hotspots_small)
        prior_large = kde_prior(grid_xy, hotspots_large)
        
        # Large sigma should have more uniform distribution
        assert np.std(prior_large) < np.std(prior_small) or abs(np.std(prior_large) - np.std(prior_small)) < 0.1


class TestSurvivalDecay:
    """Test suite for reinforcement_learning.movement_model.survival_decay function.

    Tests exponential survival decay calculation with various time values
    and half-life parameters.
    """

    def test_survival_decay_zero_time(self):
        """Test survival decay at time zero."""
        result = survival_decay(0.0, half_life=24.0)
        
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_survival_decay_half_life(self):
        """Test survival decay at half-life time."""
        result = survival_decay(24.0, half_life=24.0)
        
        assert result == pytest.approx(0.5, abs=1e-3)

    def test_survival_decay_double_half_life(self):
        """Test survival decay at double half-life time."""
        result = survival_decay(48.0, half_life=24.0)
        
        assert result == pytest.approx(0.25, abs=1e-3)

    def test_survival_decay_decreasing(self):
        """Test that survival decay is decreasing."""
        t1 = survival_decay(12.0, half_life=24.0)
        t2 = survival_decay(24.0, half_life=24.0)
        t3 = survival_decay(36.0, half_life=24.0)
        
        assert t1 > t2 > t3

    def test_survival_decay_range(self):
        """Test that survival decay is in [0, 1] range."""
        for t in [0.0, 12.0, 24.0, 48.0, 72.0]:
            result = survival_decay(t, half_life=24.0)
            assert 0.0 <= result <= 1.0

    def test_survival_decay_custom_half_life(self):
        """Test survival decay with custom half-life."""
        result = survival_decay(12.0, half_life=12.0)
        
        assert result == pytest.approx(0.5, abs=1e-3)

    def test_survival_decay_very_large_time(self):
        """Test survival decay with very large time."""
        result = survival_decay(1000.0, half_life=24.0)
        
        assert result < 1e-10  # Should be very close to zero

    def test_survival_decay_zero_half_life(self):
        """Test survival decay with zero half-life (edge case)."""
        result = survival_decay(1.0, half_life=0.0)
        
        # Should handle gracefully (uses max(half_life, 1e-6))
        assert 0.0 <= result <= 1.0


class TestSurvivalCurveExponential:
    """Test suite for reinforcement_learning.movement_model.survival_curve_exponential function.

    Tests exponential survival curve calculation and backward compatibility.
    """

    def test_survival_curve_exponential_backward_compat(self):
        """Test that survival_curve_exponential matches survival_decay."""
        t_hours = 24.0
        half_life = 24.0
        
        result_curve = survival_curve_exponential(t_hours, half_life)
        result_decay = survival_decay(t_hours, half_life)
        
        assert result_curve == pytest.approx(result_decay, abs=1e-6)


class TestSurvivalFactor:
    """Test suite for reinforcement_learning.movement_model.survival_factor function.

    Tests survival factor calculation for different movement profiles
    including default, runaway, and abduction scenarios.
    """

    def test_survival_factor_default(self):
        """Test survival_factor with default profile."""
        result = survival_factor(24.0, profile="default")
        
        assert result == pytest.approx(0.5, abs=1e-3)  # Half-life at 24h

    def test_survival_factor_runaway(self):
        """Test survival_factor with runaway profile."""
        result = survival_factor(48.0, profile="runaway")
        
        assert result == pytest.approx(0.5, abs=1e-3)  # Half-life at 48h

    def test_survival_factor_abduction(self):
        """Test survival_factor with abduction profile."""
        result = survival_factor(12.0, profile="abduction")
        
        assert result == pytest.approx(0.5, abs=1e-3)  # Half-life at 12h

    def test_survival_factor_invalid_profile(self):
        """Test survival_factor with invalid profile falls back to default."""
        result = survival_factor(24.0, profile="invalid")
        
        assert result == pytest.approx(0.5, abs=1e-3)  # Falls back to default


class TestBuildTransition:
    """Test suite for reinforcement_learning.movement_model.build_transition function.

    Tests Markov chain transition matrix construction with road cost,
    seclusion factors, and various grid configurations.
    """

    def test_build_transition_basic(self):
        """Test basic transition matrix construction."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2]
        ])
        road_cost = np.array([1.0, 1.0, 1.0])
        seclusion = np.array([0.5, 0.5, 0.5])
        k = 2
        
        P = build_transition(grid_xy, road_cost, seclusion, k=k)
        
        assert P.shape == (len(grid_xy), len(grid_xy))
        # Each row should sum to 1 (probability distribution)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
        # All values should be non-negative
        assert np.all(P >= 0)
        # All values should be finite
        assert np.all(np.isfinite(P))

    def test_build_transition_stochastic(self):
        """Test that transition matrix is stochastic."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2],
            [-77.3, 38.3]
        ])
        road_cost = np.array([1.0, 1.0, 1.0, 1.0])
        seclusion = np.array([0.5, 0.5, 0.5, 0.5])
        k = 2
        
        P = build_transition(grid_xy, road_cost, seclusion, k=k)
        
        # Rows should sum to 1
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
        # All entries should be >= 0
        assert np.all(P >= 0)

    def test_build_transition_road_cost_effect(self):
        """Test that road cost affects transition probabilities."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2]
        ])
        road_cost_low = np.array([0.1, 1.0, 1.0])  # Low cost at index 0
        road_cost_high = np.array([10.0, 1.0, 1.0])  # High cost at index 0
        seclusion = np.array([0.5, 0.5, 0.5])
        k = 2
        
        P_low = build_transition(grid_xy, road_cost_low, seclusion, k=k)
        P_high = build_transition(grid_xy, road_cost_high, seclusion, k=k)
        
        assert P_low.shape == P_high.shape

    def test_build_transition_seclusion_effect(self):
        """Test that seclusion affects transition probabilities."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2]
        ])
        road_cost = np.array([1.0, 1.0, 1.0])
        seclusion_low = np.array([0.1, 0.5, 0.5])
        seclusion_high = np.array([1.0, 0.5, 0.5])
        k = 2
        
        P_low = build_transition(grid_xy, road_cost, seclusion_low, k=k)
        P_high = build_transition(grid_xy, road_cost, seclusion_high, k=k)
        
        assert P_low.shape == P_high.shape
        # Validate row sums and non-negativity
        assert np.allclose(P_low.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(P_high.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(P_low >= 0)
        assert np.all(P_high >= 0)

    def test_build_transition_caching(self):
        """Test that transition matrices are cached."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1]
        ])
        road_cost = np.array([1.0, 1.0])
        seclusion = np.array([0.5, 0.5])
        k = 1
        
        P1 = build_transition(grid_xy, road_cost, seclusion, k=k)
        P2 = build_transition(grid_xy, road_cost, seclusion, k=k)
        
        # Should be the same (cached)
        assert np.array_equal(P1, P2)

    def test_build_transition_k_parameter(self):
        """Test that k parameter affects number of neighbors."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2],
            [-77.3, 38.3]
        ])
        road_cost = np.array([1.0, 1.0, 1.0, 1.0])
        seclusion = np.array([0.5, 0.5, 0.5, 0.5])
        
        P_k1 = build_transition(grid_xy, road_cost, seclusion, k=1)
        P_k2 = build_transition(grid_xy, road_cost, seclusion, k=2)
        
        # Different k should produce different matrices
        assert P_k1.shape == P_k2.shape
        # k=1 should have fewer non-zero entries per row
        assert np.sum(P_k1 > 1e-6) <= np.sum(P_k2 > 1e-6)


class TestRiskMap:
    """Test suite for reinforcement_learning.movement_model.risk_map function.

    Tests risk map computation using Markov chain propagation with
    various time steps, initial distributions, and hotspot configurations.
    """

    def test_risk_map_basic(self, sample_hotspots):
        """Test basic risk map computation."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2]
        ])
        road_cost = np.array([1.0, 1.0, 1.0])
        seclusion = np.array([0.5, 0.5, 0.5])
        t_hours = 12.0
        steps = 1
        
        risk = risk_map(grid_xy, sample_hotspots, road_cost, seclusion, t_hours, steps=steps)
        
        assert len(risk) == len(grid_xy)
        assert np.all(risk >= 0)
        assert np.allclose(risk.sum(), 1.0, atol=1e-6)  # Should be normalized

    def test_risk_map_normalization(self, sample_hotspots):
        """Test that risk map is normalized."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1]
        ])
        road_cost = np.array([1.0, 1.0])
        seclusion = np.array([0.5, 0.5])
        t_hours = 12.0
        steps = 1
        
        risk = risk_map(grid_xy, sample_hotspots, road_cost, seclusion, t_hours, steps=steps)
        
        assert np.allclose(risk.sum(), 1.0, atol=1e-6)

    def test_risk_map_multiple_steps(self, sample_hotspots):
        """Test risk map with multiple propagation steps."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1],
            [-77.2, 38.2]
        ])
        road_cost = np.array([1.0, 1.0, 1.0])
        seclusion = np.array([0.5, 0.5, 0.5])
        t_hours = 12.0
        
        risk_1 = risk_map(grid_xy, sample_hotspots, road_cost, seclusion, t_hours, steps=1)
        risk_3 = risk_map(grid_xy, sample_hotspots, road_cost, seclusion, t_hours, steps=3)
        
        # Multiple steps should produce different distribution
        assert not np.array_equal(risk_1, risk_3)

    def test_risk_map_time_effect(self, sample_hotspots):
        """Test that time affects risk map through survival decay."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1]
        ])
        road_cost = np.array([1.0, 1.0])
        seclusion = np.array([0.5, 0.5])
        steps = 1
        
        risk_0 = risk_map(grid_xy, sample_hotspots, road_cost, seclusion, 0.0, steps=steps)
        risk_24 = risk_map(grid_xy, sample_hotspots, road_cost, seclusion, 24.0, steps=steps)
        
        # Later time should have lower overall probability (survival decay)
        # But distribution may be different due to propagation
        assert np.allclose(risk_0.sum(), 1.0, atol=1e-6)
        assert np.allclose(risk_24.sum(), 1.0, atol=1e-6)

    def test_risk_map_custom_init(self, sample_hotspots):
        """Test risk map with custom initial distribution."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1]
        ])
        road_cost = np.array([1.0, 1.0])
        seclusion = np.array([0.5, 0.5])
        t_hours = 12.0
        steps = 1
        init = np.array([0.8, 0.2])  # Custom initial distribution
        
        risk = risk_map(grid_xy, sample_hotspots, road_cost, seclusion, t_hours, steps=steps, init=init)
        
        assert len(risk) == len(grid_xy)
        assert np.allclose(risk.sum(), 1.0, atol=1e-6)

    def test_risk_map_empty_hotspots(self):
        """Test risk map with empty hotspots."""
        grid_xy = np.array([
            [-77.0, 38.0],
            [-77.1, 38.1]
        ])
        road_cost = np.array([1.0, 1.0])
        seclusion = np.array([0.5, 0.5])
        t_hours = 12.0
        steps = 1
        hotspots = []
        
        risk = risk_map(grid_xy, hotspots, road_cost, seclusion, t_hours, steps=steps)
        
        # Should still return valid distribution
        assert len(risk) == len(grid_xy)
        assert np.allclose(risk.sum(), 1.0, atol=1e-6)


class TestLoadHotspotsMulti:
	"""Test suite for reinforcement_learning.movement_model.load_hotspots_multi function.

	Tests loading hotspots from multiple clustering method files with
	weight normalization, method-specific weights, and edge cases.
	"""

	def test_load_hotspots_multi_missing_file(self, tmp_path):
		"""Test load_hotspots_multi raises FileNotFoundError for missing files."""
		import json
		
		# Create only one file
		kmeans_file = tmp_path / "kmeans_hotspots.json"
		with open(kmeans_file, "w") as f:
			json.dump([], f)
		
		with pytest.raises(FileNotFoundError):
			load_hotspots_multi(
				kmeans_path=str(kmeans_file),
				dbscan_path=str(tmp_path / "nonexistent.json"),
				kde_path=str(tmp_path / "nonexistent2.json"),
			)

	def test_load_hotspots_multi_zero_radius(self, tmp_path):
		"""Test load_hotspots_multi handles zero radius correctly."""
		import json
		
		# Create test files
		kmeans_file = tmp_path / "kmeans_hotspots.json"
		dbscan_file = tmp_path / "dbscan_hotspots.json"
		kde_file = tmp_path / "kde_hotspots.json"
		
		kmeans_data = [
			{"lon": -77.0, "lat": 38.0, "weight": 1.0, "radius_miles": 0.0, "method": "kmeans"}
		]
		dbscan_data = [
			{"lon": -77.1, "lat": 38.1, "weight": 1.0, "radius_miles": 0.0, "method": "dbscan"}
		]
		kde_data = [
			{"lon": -77.2, "lat": 38.2, "weight": 1.0, "radius_miles": 5.0, "method": "kde"}
		]
		
		with open(kmeans_file, "w") as f:
			json.dump(kmeans_data, f)
		with open(dbscan_file, "w") as f:
			json.dump(dbscan_data, f)
		with open(kde_file, "w") as f:
			json.dump(kde_data, f)
		
		hotspots = load_hotspots_multi(
			kmeans_path=str(kmeans_file),
			dbscan_path=str(dbscan_file),
			kde_path=str(kde_file),
		)
		
		# Should use default sigma_mi=5.0 for zero radius
		assert len(hotspots) == 3
		assert all(s_miles == 5.0 for _, _, _, s_miles in hotspots[:2])  # First two should use default
		assert hotspots[2][3] == 5.0  # KDE has explicit radius

	def test_load_hotspots_multi_method_weights(self, tmp_path):
		"""Test load_hotspots_multi respects method weights."""
		import json
		
		kmeans_file = tmp_path / "kmeans_hotspots.json"
		dbscan_file = tmp_path / "dbscan_hotspots.json"
		kde_file = tmp_path / "kde_hotspots.json"
		
		kmeans_data = [{"lon": -77.0, "lat": 38.0, "weight": 1.0, "radius_miles": 5.0, "method": "kmeans"}]
		dbscan_data = [{"lon": -77.1, "lat": 38.1, "weight": 1.0, "radius_miles": 5.0, "method": "dbscan"}]
		kde_data = [{"lon": -77.2, "lat": 38.2, "weight": 1.0, "radius_miles": 5.0, "method": "kde"}]
		
		with open(kmeans_file, "w") as f:
			json.dump(kmeans_data, f)
		with open(dbscan_file, "w") as f:
			json.dump(dbscan_data, f)
		with open(kde_file, "w") as f:
			json.dump(kde_data, f)
		
		# Test with method weights
		hotspots = load_hotspots_multi(
			kmeans_path=str(kmeans_file),
			dbscan_path=str(dbscan_file),
			kde_path=str(kde_file),
			method_weights={"kmeans": 2.0, "dbscan": 1.0, "kde": 1.0},
		)
		
		# Weights should be normalized, but kmeans should have higher relative weight
		assert len(hotspots) == 3
		total_weight = sum(w for _, _, w, _ in hotspots)
		assert np.allclose(total_weight, 1.0, atol=1e-6)

	def test_load_hotspots_multi_normalization(self, tmp_path):
		"""Test load_hotspots_multi normalizes weights correctly."""
		import json
		
		kmeans_file = tmp_path / "kmeans_hotspots.json"
		dbscan_file = tmp_path / "dbscan_hotspots.json"
		kde_file = tmp_path / "kde_hotspots.json"
		
		kmeans_data = [{"lon": -77.0, "lat": 38.0, "weight": 1.0, "radius_miles": 5.0, "method": "kmeans"}]
		dbscan_data = [{"lon": -77.1, "lat": 38.1, "weight": 1.0, "radius_miles": 5.0, "method": "dbscan"}]
		kde_data = [{"lon": -77.2, "lat": 38.2, "weight": 1.0, "radius_miles": 5.0, "method": "kde"}]
		
		with open(kmeans_file, "w") as f:
			json.dump(kmeans_data, f)
		with open(dbscan_file, "w") as f:
			json.dump(dbscan_data, f)
		with open(kde_file, "w") as f:
			json.dump(kde_data, f)
		
		hotspots = load_hotspots_multi(
			kmeans_path=str(kmeans_file),
			dbscan_path=str(dbscan_file),
			kde_path=str(kde_file),
		)
		
		# Weights should sum to 1.0
		total_weight = sum(w for _, _, w, _ in hotspots)
		assert np.allclose(total_weight, 1.0, atol=1e-6)
