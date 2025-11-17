"""Unit tests for src.geography.distance module."""

import pytest
from src.geography.distance import haversine_distance, manhattan_distance


class TestHaversineDistance:
    """Test suite for haversine_distance function.

    Tests great-circle distance calculation between geographic coordinates
    with various scenarios including same point, known distances, and edge cases.
    """

    def test_same_point(self):
        """Test distance between same point."""
        distance = haversine_distance(38.88, -77.1, 38.88, -77.1)
        
        assert distance == pytest.approx(0.0, abs=0.01)

    def test_richmond_to_norfolk(self):
        """Test distance between Richmond and Norfolk."""
        # Richmond, VA: 37.5407, -77.4360
        # Norfolk, VA: 36.8468, -76.2852
        distance = haversine_distance(37.5407, -77.4360, 36.8468, -76.2852)
        
        # Approximately 90 miles
        assert 85 < distance < 95

    def test_north_south_distance(self):
        """Test distance along meridian (north-south)."""
        # 1 degree latitude ≈ 69 miles
        distance = haversine_distance(38.0, -77.0, 39.0, -77.0)
        
        assert 65 < distance < 73

    def test_east_west_distance(self):
        """Test distance along parallel (east-west)."""
        # At 38°N, 1 degree longitude ≈ 54.6 miles
        distance = haversine_distance(38.0, -77.0, 38.0, -76.0)
        
        assert 50 < distance < 60

    def test_small_distance(self):
        """Test distance for nearby points."""
        distance = haversine_distance(38.88, -77.1, 38.89, -77.11)
        
        assert distance > 0
        assert distance < 10

    def test_known_coordinates(self):
        """Test distance with known coordinate pairs."""
        distance = haversine_distance(0.0, 0.0, 1.0, 1.0)
        
        assert distance > 0


class TestManhattanDistance:
    """Test suite for manhattan_distance function.

    Tests Manhattan (L1) distance calculation between geographic coordinates
    and comparison with Haversine distance.
    """

    def test_same_point(self):
        """Test Manhattan distance between same point."""
        distance = manhattan_distance(38.88, -77.1, 38.88, -77.1)
        
        assert distance == pytest.approx(0.0, abs=0.01)

    def test_north_south(self):
        """Test Manhattan distance along latitude."""
        distance = manhattan_distance(38.0, -77.0, 39.0, -77.0)
        
        assert 65 < distance < 73

    def test_east_west(self):
        """Test Manhattan distance along longitude."""
        distance = manhattan_distance(38.0, -77.0, 38.0, -76.0)
        
        assert 50 < distance < 60

    def test_manhattan_vs_haversine(self):
        """Test that Manhattan distance is typically larger than Haversine."""
        lat1, lon1 = 38.88, -77.1
        lat2, lon2 = 38.89, -77.11
        
        manhattan = manhattan_distance(lat1, lon1, lat2, lon2)
        haversine = haversine_distance(lat1, lon1, lat2, lon2)
        
        assert manhattan >= haversine

    def test_diagonal_distance(self):
        """Test Manhattan distance for diagonal movement."""
        distance = manhattan_distance(38.0, -77.0, 39.0, -76.0)
        
        assert distance > 0
        assert distance > 50
