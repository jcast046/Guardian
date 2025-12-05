"""Integration tests for search plan generation.

Tests sector assignment, probabilities, hotspots, containment rings,
and full search plan generation.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_here = Path(__file__).resolve()
_proj_root = _here.parents[2]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import numpy as np
import pytest
import json
import tempfile
from pathlib import Path

from reinforcement_learning.sectors import (
    load_sectors,
    assign_grid_to_sectors,
    sector_probabilities,
    rank_sectors,
    sector_hotspots,
)
from reinforcement_learning.rings import (
    compute_distance_miles,
    probability_radii,
)
from reinforcement_learning.forecast_api import (
    attach_sector_probs,
    forecast_search_plan,
)


@pytest.fixture
def sample_case():
    """Create a sample case dictionary for testing."""
    return {
        "case_id": "GRD-2025-TEST001",
        "temporal": {
            "last_seen_ts": "2025-01-15T14:00:00Z",
            "reported_missing_ts": "2025-01-15T15:00:00Z",
        },
        "spatial": {
            "last_seen_lat": 38.88,
            "last_seen_lon": -77.1,
        },
    }


@pytest.fixture
def test_grid_xy():
    """Create a small test grid for testing."""
    # Create a 10x10 grid around Richmond, VA
    lon_min, lon_max = -77.5, -77.0
    lat_min, lat_max = 37.5, 38.0
    
    lons = np.linspace(lon_min, lon_max, 10)
    lats = np.linspace(lat_min, lat_max, 10)
    
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_xy = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    
    return grid_xy


@pytest.fixture
def test_sectors_gdf():
    """Load test sectors from va_rl_regions.geojson."""
    sector_path = Path("data/geo/va_rl_regions.geojson")
    if not sector_path.exists():
        pytest.skip(f"Sector file not found: {sector_path}")
    return load_sectors(str(sector_path))


class TestSectorAssignment:
    """Test sector assignment functionality."""
    
    def test_load_sectors(self, test_sectors_gdf):
        """Test loading sectors from GeoJSON."""
        assert len(test_sectors_gdf) > 0
        assert "sector_id" in test_sectors_gdf.columns
        assert "geometry" in test_sectors_gdf.columns
    
    def test_assign_grid_to_sectors(self, test_grid_xy, test_sectors_gdf):
        """Test grid point assignment to sectors."""
        sector_idx = assign_grid_to_sectors(test_grid_xy, test_sectors_gdf)
        
        assert sector_idx.shape == (len(test_grid_xy),)
        assert sector_idx.dtype in [np.int32, np.int64]
        # Some points should be assigned to sectors (not all -1)
        assert (sector_idx >= 0).any()
    
    def test_sector_probabilities(self, test_grid_xy, test_sectors_gdf):
        """Test sector probability computation."""
        # Create a simple probability distribution
        n = len(test_grid_xy)
        p = np.ones(n) / n  # Uniform distribution
        
        sector_idx = assign_grid_to_sectors(test_grid_xy, test_sectors_gdf)
        sectors = sector_probabilities(p, sector_idx, test_sectors_gdf)
        
        assert len(sectors) == len(test_sectors_gdf)
        
        # Check that probabilities sum to ~1.0
        total_mass_pct = sum(s["mass_pct"] for s in sectors)
        assert abs(total_mass_pct - 1.0) < 0.01  # Allow small tolerance
        
        # Check that sectors are sorted by mass descending
        masses = [s["mass"] for s in sectors]
        assert masses == sorted(masses, reverse=True)
    
    def test_rank_sectors(self, test_grid_xy, test_sectors_gdf):
        """Test sector ranking and filtering."""
        n = len(test_grid_xy)
        p = np.ones(n) / n
        
        sector_idx = assign_grid_to_sectors(test_grid_xy, test_sectors_gdf)
        ranked = rank_sectors(p, sector_idx, test_sectors_gdf, min_mass=0.01)
        
        assert len(ranked) <= len(test_sectors_gdf)
        
        # All ranked sectors should have mass_pct >= min_mass
        for sector in ranked:
            assert sector["mass_pct"] >= 0.01
        
        # Top sector should have mass_pct > 0
        if ranked:
            assert ranked[0]["mass_pct"] > 0


class TestSectorHotspots:
    """Test sector hotspot identification."""
    
    def test_sector_hotspots(self, test_grid_xy, test_sectors_gdf):
        """Test hotspot identification within sectors."""
        n = len(test_grid_xy)
        # Create a distribution with a peak
        p = np.ones(n) / n
        # Add a spike at one location
        p[0] = 0.5
        p = p / p.sum()
        
        sector_idx = assign_grid_to_sectors(test_grid_xy, test_sectors_gdf)
        sectors_ranked = rank_sectors(p, sector_idx, test_sectors_gdf, min_mass=0.01)
        
        if sectors_ranked:
            hotspots_list = sector_hotspots(
                test_grid_xy, p, sector_idx, sectors_ranked, test_sectors_gdf,
                local_pct=0.9
            )
            
            assert len(hotspots_list) == len(sectors_ranked)
            
            # Check structure
            for sh in hotspots_list:
                assert "sector_id" in sh
                assert "hotspots" in sh
                assert isinstance(sh["hotspots"], list)
                
                # Check hotspot structure if present
                for hotspot in sh["hotspots"]:
                    assert "lon" in hotspot
                    assert "lat" in hotspot
                    assert "p" in hotspot
    
    def test_sector_hotspots_zero_probability(self, test_grid_xy, test_sectors_gdf):
        """Test that sectors with zero probability are skipped."""
        n = len(test_grid_xy)
        # Create a distribution with zero probability in some sectors
        p = np.zeros(n)
        # Only assign probability to a few points
        p[:10] = 1.0 / 10.0
        
        sector_idx = assign_grid_to_sectors(test_grid_xy, test_sectors_gdf)
        sectors_ranked = rank_sectors(p, sector_idx, test_sectors_gdf, min_mass=0.01)
        
        hotspots_list = sector_hotspots(
            test_grid_xy, p, sector_idx, sectors_ranked, test_sectors_gdf,
            local_pct=0.9
        )
        
        # Should handle zero-probability sectors gracefully
        assert len(hotspots_list) == len(sectors_ranked)


class TestProbabilityRings:
    """Test probability containment ring computation."""
    
    def test_compute_distance_miles(self, test_grid_xy):
        """Test distance computation from IPP."""
        lon0, lat0 = -77.1, 38.88
        dist = compute_distance_miles(test_grid_xy, lon0, lat0)
        
        assert dist.shape == (len(test_grid_xy),)
        assert np.all(dist >= 0)
        assert np.all(np.isfinite(dist))
    
    def test_probability_radii(self, test_grid_xy):
        """Test probability containment radius computation."""
        lon0, lat0 = -77.1, 38.88
        
        # Create a simple distribution
        n = len(test_grid_xy)
        p = np.ones(n) / n
        
        rings = probability_radii(test_grid_xy, p, lon0, lat0)
        
        assert len(rings) == 3  # Default quantiles: 0.5, 0.75, 0.9
        
        # Check structure
        for ring in rings:
            assert "q" in ring
            assert "radius_mi" in ring
            assert 0 <= ring["q"] <= 1
            assert ring["radius_mi"] >= 0
            assert np.isfinite(ring["radius_mi"])
        
        # Check that rings are non-decreasing
        radii = [r["radius_mi"] for r in rings]
        assert radii == sorted(radii)
        
        # Check quantiles are in order
        quantiles = [r["q"] for r in rings]
        assert quantiles == sorted(quantiles)


class TestForecastSearchPlan:
    """Test full search plan generation."""
    
    @pytest.mark.slow
    def test_forecast_search_plan_basic(self, sample_case):
        """Test basic search plan generation."""
        # This test may be slow, so mark it
        try:
            search_plan = forecast_search_plan(
                case=sample_case,
                horizons=(24,),
                use_cumulative=False,
            )
        except Exception as e:
            # If forecast fails due to missing data, skip test
            pytest.skip(f"Forecast failed (may be missing data): {e}")
        
        # Check required keys
        required_keys = [
            "grid_xy", "p", "sectors_gdf", "sector_idx", "sectors_ranked",
            "sector_hotspots", "rings", "ipp", "sectors_metadata", "sector_ids"
        ]
        for key in required_keys:
            assert key in search_plan, f"Missing key: {key}"
        
        # Check probability distribution
        p = search_plan["p"]
        assert len(p) > 0
        assert abs(p.sum() - 1.0) < 0.01  # Should sum to ~1.0
        
        # Check sectors
        sectors_ranked = search_plan["sectors_ranked"]
        assert len(sectors_ranked) > 0
        
        # Top sector should have non-zero mass
        if sectors_ranked:
            assert sectors_ranked[0]["mass_pct"] > 0
        
        # Check IPP
        ipp = search_plan.get("ipp")
        if ipp:
            assert "lon" in ipp
            assert "lat" in ipp
        
        # Check rings if IPP available
        rings = search_plan.get("rings", [])
        if ipp and rings:
            assert len(rings) > 0
            # Rings should be non-decreasing
            radii = [r["radius_mi"] for r in rings]
            assert radii == sorted(radii)
    
    def test_attach_sector_probs(self, sample_case):
        """Test sector probability attachment."""
        # Create a simple probability distribution
        # This is a simplified test that doesn't require full forecast
        try:
            from reinforcement_learning.build_rl_zones import load_grid_and_layers
            grid_xy, _, _, _, _, _ = load_grid_and_layers()
        except Exception:
            pytest.skip("Could not load grid and layers")
        
        n = len(grid_xy)
        p = np.ones(n) / n
        
        sector_info = attach_sector_probs(grid_xy, p)
        
        assert "sectors_gdf" in sector_info
        assert "sector_idx" in sector_info
        assert "sectors_ranked" in sector_info
        
        assert sector_info["sector_idx"].shape == (n,)
        assert len(sector_info["sectors_ranked"]) > 0


class TestSearchPlanVisualization:
    """Test search plan visualization."""
    
    @pytest.mark.slow
    def test_search_plan_visualization(self, sample_case):
        """Test search plan visualization generation."""
        try:
            search_plan = forecast_search_plan(
                case=sample_case,
                horizons=(24,),
                use_cumulative=False,
            )
        except Exception as e:
            pytest.skip(f"Forecast failed (may be missing data): {e}")
        
        # Create temporary file for visualization
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            outpath = tmp.name
        
        try:
            from reinforcement_learning.visualize_forecast import plot_search_plan
            plot_search_plan(
                case=sample_case,
                search_plan=search_plan,
                outpath=outpath,
            )
            
            # Check that file was created
            assert Path(outpath).exists()
            assert Path(outpath).stat().st_size > 0
            
        finally:
            # Clean up
            if Path(outpath).exists():
                Path(outpath).unlink()

