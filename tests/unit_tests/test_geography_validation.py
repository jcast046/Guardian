"""Unit tests for src.geography.validation module."""

import pytest
from src.geography.validation import (
    is_geographically_accurate_road,
    is_major_road_in_region,
    is_obviously_incorrect_road
)


class TestIsGeographicallyAccurateRoad:
    """Test suite for is_geographically_accurate_road function.

    Tests validation of road names against geographic context including
    place name matching, major interstates, and local roads.
    """

    def test_accurate_road_matching_place(self):
        """Test road that matches nearby place name."""
        road_name = "Richmond Highway"
        region = "Central Virginia"
        nearby_places = [
            {"name": "Richmond", "distance": 2.0}
        ]
        lat, lon = 37.5407, -77.4360
        
        result = is_geographically_accurate_road(road_name, region, nearby_places, lat, lon)
        
        assert result is True

    def test_accurate_major_interstate(self):
        """Test major interstate that should be accurate."""
        road_name = "I-95"
        region = "Central Virginia"
        nearby_places = []
        lat, lon = 37.5407, -77.4360
        
        result = is_geographically_accurate_road(road_name, region, nearby_places, lat, lon)
        
        assert result is True

    def test_inaccurate_road_obviously_wrong(self):
        """Test road that is obviously incorrect."""
        road_name = "Natural Bridge Road"
        region = "Northern Virginia"
        nearby_places = []
        lat, lon = 38.88, -77.1
        
        result = is_geographically_accurate_road(road_name, region, nearby_places, lat, lon)
        
        assert result is False

    def test_accurate_local_road(self):
        """Test local road that should be accurate."""
        road_name = "Main Street"
        region = "Central Virginia"
        nearby_places = []
        lat, lon = 37.5407, -77.4360
        
        result = is_geographically_accurate_road(road_name, region, nearby_places, lat, lon)
        
        assert result is True


class TestIsMajorRoadInRegion:
    """Test suite for is_major_road_in_region function.

    Tests validation of major roads (interstates, US routes, state routes)
    against regional contexts in Virginia.
    """

    def test_i95_most_regions(self):
        """Test I-95 which runs through most of Virginia."""
        assert is_major_road_in_region("I-95", "Northern Virginia", 38.88, -77.1) is True
        assert is_major_road_in_region("I-95", "Central Virginia", 37.54, -77.43) is True
        assert is_major_road_in_region("I-95", "Tidewater", 36.84, -76.28) is True

    def test_i81_western_va(self):
        """Test I-81 which runs through western Virginia."""
        assert is_major_road_in_region("I-81", "Valley", 38.0, -79.0) is True
        assert is_major_road_in_region("I-81", "Southwest", 36.0, -81.0) is True

    def test_i66_northern_va(self):
        """Test I-66 which is Northern Virginia specific."""
        assert is_major_road_in_region("I-66", "Northern Virginia", 38.88, -77.1) is True
        assert is_major_road_in_region("I-66", "Central Virginia", 37.54, -77.43) is False

    def test_i395_i495_northern_va(self):
        """Test I-395 and I-495 which are Northern Virginia specific."""
        assert is_major_road_in_region("I-395", "Northern Virginia", 38.88, -77.1) is True
        assert is_major_road_in_region("I-495", "Northern Virginia", 38.88, -77.1) is True
        assert is_major_road_in_region("I-395", "Central Virginia", 37.54, -77.43) is False

    def test_i264_i564_tidewater(self):
        """Test I-264 and I-564 which are Tidewater specific."""
        assert is_major_road_in_region("I-264", "Tidewater", 36.84, -76.28) is True
        assert is_major_road_in_region("I-564", "Tidewater", 36.84, -76.28) is True
        assert is_major_road_in_region("I-264", "Northern Virginia", 38.88, -77.1) is False

    def test_us_routes(self):
        """Test US routes which are generally widespread."""
        assert is_major_road_in_region("US-29", "Northern Virginia", 38.88, -77.1) is True
        assert is_major_road_in_region("US-29", "Central Virginia", 37.54, -77.43) is True

    def test_va_routes(self):
        """Test VA state routes which are region-specific."""
        assert is_major_road_in_region("VA-123", "Northern Virginia", 38.88, -77.1) is True
        assert is_major_road_in_region("VA-123", "Central Virginia", 37.54, -77.43) is True

    def test_i64_regions(self):
        """Test I-64 which runs through central and eastern Virginia."""
        assert is_major_road_in_region("I-64", "Tidewater", 36.84, -76.28) is True
        assert is_major_road_in_region("I-64", "Central Virginia", 37.54, -77.43) is True
        assert is_major_road_in_region("I-64", "Northern Virginia", 38.88, -77.1) is True


class TestIsObviouslyIncorrectRoad:
    """Test suite for is_obviously_incorrect_road function.

    Tests detection of obviously incorrect road names based on
    regional mismatches and known geographic boundaries.
    """

    def test_western_va_in_northern_va(self):
        """Test western VA roads incorrectly in Northern VA."""
        assert is_obviously_incorrect_road("Natural Bridge", "Northern Virginia") is True
        assert is_obviously_incorrect_road("Shenandoah", "Northern Virginia") is True
        assert is_obviously_incorrect_road("Mudlick", "Northern Virginia") is True

    def test_tidewater_in_northern_va(self):
        """Test Tidewater roads incorrectly in Northern VA."""
        assert is_obviously_incorrect_road("Norfolk", "Northern Virginia") is True
        assert is_obviously_incorrect_road("Virginia Beach", "Northern Virginia") is True

    def test_northern_va_in_central_va(self):
        """Test Northern VA roads incorrectly in Central VA."""
        assert is_obviously_incorrect_road("Franconia", "Central Virginia") is True
        assert is_obviously_incorrect_road("Tysons", "Central Virginia") is True

    def test_tidewater_in_central_va(self):
        """Test Tidewater roads incorrectly in Central VA."""
        assert is_obviously_incorrect_road("Norfolk", "Central Virginia") is True
        assert is_obviously_incorrect_road("Virginia Beach", "Central Virginia") is True

    def test_western_va_in_central_va(self):
        """Test western VA roads incorrectly in Central VA."""
        assert is_obviously_incorrect_road("Natural Bridge", "Central Virginia") is True
        assert is_obviously_incorrect_road("Shenandoah", "Central Virginia") is True

    def test_northern_va_in_tidewater(self):
        """Test Northern VA roads incorrectly in Tidewater."""
        assert is_obviously_incorrect_road("Franconia", "Tidewater") is True
        assert is_obviously_incorrect_road("Tysons", "Tidewater") is True

    def test_western_va_in_tidewater(self):
        """Test western VA roads incorrectly in Tidewater."""
        assert is_obviously_incorrect_road("Natural Bridge", "Tidewater") is True
        assert is_obviously_incorrect_road("Shenandoah", "Tidewater") is True

    def test_i81_in_northern_va(self):
        """Test I-81 incorrectly in Northern Virginia."""
        assert is_obviously_incorrect_road("I-81", "Northern Virginia") is True

    def test_i85_in_central_va(self):
        """Test I-85 incorrectly in Central Virginia."""
        assert is_obviously_incorrect_road("I-85", "Central Virginia") is True

    def test_correct_road(self):
        """Test road that is correct for the region."""
        assert is_obviously_incorrect_road("I-95", "Northern Virginia") is False
        assert is_obviously_incorrect_road("Main Street", "Central Virginia") is False
        assert is_obviously_incorrect_road("I-64", "Tidewater") is False

    def test_case_insensitive(self):
        """Test that road name matching is case insensitive."""
        assert is_obviously_incorrect_road("NATURAL BRIDGE", "Northern Virginia") is True
        assert is_obviously_incorrect_road("natural bridge", "Northern Virginia") is True
        assert is_obviously_incorrect_road("Natural Bridge", "Northern Virginia") is True
