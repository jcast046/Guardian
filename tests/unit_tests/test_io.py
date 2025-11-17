"""Unit tests for metrics.io module."""

import pytest
import json
import tempfile
import pathlib
from metrics.io import read_json_blocks, coord_of, haversine_miles


class TestReadJsonBlocks:
    """Test suite for metrics.io.read_json_blocks function.

    Tests reading JSON/JSONL files with various formats including
    compact JSONL, pretty-printed JSON, and edge cases.
    """

    def test_read_jsonl(self, tmp_path):
        """Test reading JSONL format."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"case_id": "GRD-001"}\n{"case_id": "GRD-002"}\n')
        
        result = read_json_blocks(str(jsonl_file))
        
        assert len(result) == 2
        assert result[0]["case_id"] == "GRD-001"
        assert result[1]["case_id"] == "GRD-002"

    def test_read_pretty_json(self, tmp_path):
        """Test reading pretty-printed JSON blocks."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{\n  "case_id": "GRD-001"\n}\n\n{\n  "case_id": "GRD-002"\n}\n')
        
        result = read_json_blocks(str(json_file))
        
        assert len(result) == 2
        assert result[0]["case_id"] == "GRD-001"
        assert result[1]["case_id"] == "GRD-002"

    def test_read_empty_file(self, tmp_path):
        """Test reading empty file."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")
        
        result = read_json_blocks(str(jsonl_file))
        
        assert result == []

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file."""
        result = read_json_blocks("nonexistent.jsonl")
        
        assert result == []

    def test_read_invalid_json(self, tmp_path):
        """Test reading file with invalid JSON."""
        jsonl_file = tmp_path / "invalid.jsonl"
        jsonl_file.write_text('{"case_id": "GRD-001"}\n{invalid json}\n')
        
        result = read_json_blocks(str(jsonl_file))
        
        assert len(result) >= 1
        assert result[0]["case_id"] == "GRD-001"


class TestCoordOf:
    """Test suite for metrics.io.coord_of function.

    Tests coordinate extraction from various field formats including
    lat/lon, latitude/longitude, center_lat/lon, last_seen_lat/lon,
    and nested location objects.
    """

    def test_coord_lat_lon(self):
        """Test extracting coordinates from lat/lon fields."""
        row = {"lat": 38.88, "lon": -77.1}
        lat, lon = coord_of(row)
        
        assert lat == 38.88
        assert lon == -77.1

    def test_coord_latitude_longitude(self):
        """Test extracting coordinates from latitude/longitude fields."""
        row = {"latitude": 38.88, "longitude": -77.1}
        lat, lon = coord_of(row)
        
        assert lat == 38.88
        assert lon == -77.1

    def test_coord_center_lat_lon(self):
        """Test extracting coordinates from center_lat/center_lon fields."""
        row = {"center_lat": 38.88, "center_lon": -77.1}
        lat, lon = coord_of(row)
        
        assert lat == 38.88
        assert lon == -77.1

    def test_coord_last_seen_lat_lon(self):
        """Test extracting coordinates from last_seen_lat/last_seen_lon fields."""
        row = {"last_seen_lat": 38.88, "last_seen_lon": -77.1}
        lat, lon = coord_of(row)
        
        assert lat == 38.88
        assert lon == -77.1

    def test_coord_nested_location(self):
        """Test extracting coordinates from nested location object."""
        row = {
            "location": {
                "lat": 38.88,
                "lon": -77.1
            }
        }
        lat, lon = coord_of(row)
        
        assert lat == 38.88
        assert lon == -77.1

    def test_coord_missing(self):
        """Test extracting coordinates when missing."""
        row = {"case_id": "GRD-001"}
        lat, lon = coord_of(row)
        
        assert lat is None
        assert lon is None

    def test_coord_partial(self):
        """Test extracting coordinates when only one coordinate is present."""
        row = {"lat": 38.88}
        lat, lon = coord_of(row)
        
        assert lat is None
        assert lon is None


class TestHaversineMiles:
    """Test suite for metrics.io.haversine_miles function.

    Tests great-circle distance calculation in miles with various
    scenarios including known distances and edge cases.
    """

    def test_same_point(self):
        """Test distance between same point."""
        distance = haversine_miles(38.88, -77.1, 38.88, -77.1)
        
        assert distance == pytest.approx(0.0, abs=0.01)

    def test_richmond_to_norfolk(self):
        """Test distance between Richmond and Norfolk."""
        # Richmond, VA: 37.5407, -77.4360
        # Norfolk, VA: 36.8468, -76.2852
        distance = haversine_miles(37.5407, -77.4360, 36.8468, -76.2852)
        
        assert 85 < distance < 95

    def test_known_distance(self):
        """Test distance with known coordinates."""
        distance = haversine_miles(38.88, -77.1, 38.89, -77.11)
        
        assert distance > 0
        assert distance < 10

    def test_antipodal_points(self):
        """Test distance between antipodal points."""
        distance = haversine_miles(0.0, 0.0, 0.0, 180.0)
        
        assert 12000 < distance < 13000

    def test_north_south(self):
        """Test distance along meridian (north-south)."""
        distance = haversine_miles(38.0, -77.0, 39.0, -77.0)
        
        assert 65 < distance < 73

    def test_east_west(self):
        """Test distance along parallel (east-west)."""
        distance = haversine_miles(38.0, -77.0, 38.0, -76.0)
        
        assert 50 < distance < 60
