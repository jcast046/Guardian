"""Unit tests for metrics.rl module."""

import pytest
from metrics.rl import _zone_score, _flatten_zones, _zone_center


class TestZoneScore:
    """Test suite for metrics.rl._zone_score function.

    Tests zone scoring with priority, priority_llm, and score fields,
    including precedence rules and edge cases.
    """

    def test_zone_priority(self):
        """Test zone scoring with priority field."""
        zone = {"priority": 0.8}
        result = _zone_score(zone)
        
        assert result == 0.8

    def test_zone_priority_llm(self):
        """Test zone scoring with priority_llm field."""
        zone = {"priority_llm": 0.75}
        result = _zone_score(zone)
        
        assert result == 0.75

    def test_zone_score_field(self):
        """Test zone scoring with score field."""
        zone = {"score": 0.9}
        result = _zone_score(zone)
        
        assert result == 0.9

    def test_zone_precedence(self):
        """Test that priority takes precedence over other fields."""
        zone = {"priority": 0.8, "priority_llm": 0.7, "score": 0.9}
        result = _zone_score(zone)
        
        assert result == 0.8

    def test_zone_no_score(self):
        """Test zone scoring with no score fields."""
        zone = {}
        result = _zone_score(zone)
        
        assert result == 0.0

    def test_zone_none_values(self):
        """Test zone scoring with None values."""
        zone = {"priority": None, "priority_llm": 0.7}
        result = _zone_score(zone)
        
        assert result == 0.7


class TestFlattenZones:
    """Test suite for metrics.rl._flatten_zones function.

    Tests flattening zone structures from dictionary format (time windows)
    to flat lists with various edge cases.
    """

    def test_flatten_dict_zones(self):
        """Test flattening zones from dictionary format."""
        zones = {
            "0-24": [
                {"zone_id": "z01", "priority": 0.8},
                {"zone_id": "z02", "priority": 0.6}
            ],
            "24-48": [
                {"zone_id": "z03", "priority": 0.7}
            ]
        }
        result = _flatten_zones(zones)
        
        assert len(result) == 3
        assert result[0]["zone_id"] == "z01"
        assert result[1]["zone_id"] == "z02"
        assert result[2]["zone_id"] == "z03"

    def test_flatten_list_zones(self):
        """Test flattening zones from list format."""
        zones = [
            {"zone_id": "z01", "priority": 0.8},
            {"zone_id": "z02", "priority": 0.6}
        ]
        result = _flatten_zones(zones)
        
        assert len(result) == 2
        assert result == zones

    def test_flatten_empty_dict(self):
        """Test flattening empty dictionary."""
        zones = {}
        result = _flatten_zones(zones)
        
        assert result == []

    def test_flatten_empty_list(self):
        """Test flattening empty list."""
        zones = []
        result = _flatten_zones(zones)
        
        assert result == []

    def test_flatten_none(self):
        """Test flattening None."""
        result = _flatten_zones(None)
        
        assert result == []

    def test_flatten_nested_dict(self):
        """Test flattening nested dictionary structure."""
        zones = {
            "window1": [
                {"zone_id": "z01"}
            ],
            "window2": [
                {"zone_id": "z02"},
                {"zone_id": "z03"}
            ]
        }
        result = _flatten_zones(zones)
        
        assert len(result) == 3


class TestZoneCenter:
    """Test suite for metrics.rl._zone_center function.

    Tests extraction of zone center coordinates from various field formats
    including center dict and center_lat/center_lon fields.
    """

    def test_center_dict(self):
        """Test extracting center from center dictionary."""
        zone = {
            "center": {
                "lat": 38.88,
                "lon": -77.1
            }
        }
        lat, lon = _zone_center(zone)
        
        assert lat == 38.88
        assert lon == -77.1

    def test_center_lat_lon_fields(self):
        """Test extracting center from center_lat/center_lon fields."""
        zone = {
            "center_lat": 38.88,
            "center_lon": -77.1
        }
        lat, lon = _zone_center(zone)
        
        assert lat == 38.88
        assert lon == -77.1

    def test_center_precedence(self):
        """Test that center dict takes precedence over center_lat/lon."""
        zone = {
            "center": {"lat": 38.88, "lon": -77.1},
            "center_lat": 39.0,
            "center_lon": -77.2
        }
        lat, lon = _zone_center(zone)
        
        assert lat == 38.88
        assert lon == -77.1

    def test_center_missing(self):
        """Test extracting center when missing."""
        zone = {}
        lat, lon = _zone_center(zone)
        
        assert lat is None
        assert lon is None

    def test_center_partial(self):
        """Test extracting center with partial data."""
        zone = {"center_lat": 38.88}
        lat, lon = _zone_center(zone)
        
        assert lat is None
        assert lon is None

    def test_center_empty_dict(self):
        """Test extracting center from empty center dict."""
        zone = {"center": {}}
        lat, lon = _zone_center(zone)
        
        assert lat is None
        assert lon is None
