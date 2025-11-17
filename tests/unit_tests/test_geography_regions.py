"""Unit tests for src.geography.regions module."""

import pytest
from src.geography.regions import get_region_from_coordinates, find_nearby_places
from src.geography.distance import haversine_distance


class TestGetRegionFromCoordinates:
    """Test suite for get_region_from_coordinates function.

    Tests region classification from geographic coordinates using
    GeoJSON region boundaries with various edge cases.
    """

    def test_region_nova(self, sample_geojson_regions):
        """Test region classification for Northern Virginia."""
        region = get_region_from_coordinates(38.5, -77.5, sample_geojson_regions)
        
        assert region == "NoVA"

    def test_region_tidewater(self, sample_geojson_regions):
        """Test region classification for Tidewater."""
        region = get_region_from_coordinates(37.0, -77.0, sample_geojson_regions)
        
        assert region == "Tidewater"

    def test_region_unknown(self, sample_geojson_regions):
        """Test region classification for coordinates outside all regions."""
        region = get_region_from_coordinates(40.0, -80.0, sample_geojson_regions)
        
        assert region == "Unknown"

    def test_region_boundary(self, sample_geojson_regions):
        """Test region classification at boundary."""
        region = get_region_from_coordinates(38.0, -77.0, sample_geojson_regions)
        
        assert region in ["NoVA", "Tidewater", "Unknown"]

    def test_region_empty_features(self):
        """Test region classification with empty features."""
        empty_geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        region = get_region_from_coordinates(38.5, -77.5, empty_geojson)
        
        assert region == "Unknown"

    def test_region_no_polygon(self):
        """Test region classification with non-polygon geometry."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-77.0, 38.0]
                    },
                    "properties": {"region_tag": "NoVA"}
                }
            ]
        }
        region = get_region_from_coordinates(38.5, -77.5, geojson)
        
        assert region == "Unknown"


class TestFindNearbyPlaces:
    """Test suite for find_nearby_places function.

    Tests finding nearby places from a gazetteer within specified
    distance with distance calculation and sorting verification.
    """

    def test_find_nearby_within_distance(self, sample_gazetteer):
        """Test finding places within max distance."""
        lat, lon = 37.5407, -77.4360
        places = find_nearby_places(lat, lon, sample_gazetteer, max_distance=100.0)
        
        assert len(places) > 0
        assert any(place["name"] == "Richmond" for place in places)

    def test_find_nearby_outside_distance(self, sample_gazetteer):
        """Test finding places outside max distance."""
        lat, lon = 40.0, -80.0
        places = find_nearby_places(lat, lon, sample_gazetteer, max_distance=10.0)
        
        assert len(places) == 0

    def test_find_nearby_distance_calculation(self, sample_gazetteer):
        """Test that distances are calculated correctly."""
        lat, lon = 37.5407, -77.4360
        places = find_nearby_places(lat, lon, sample_gazetteer, max_distance=100.0)
        
        for place in places:
            assert place["distance"] <= 100.0
            assert "distance" in place
            assert "name" in place
            assert "type" in place

    def test_find_nearby_sorted_by_distance(self, sample_gazetteer):
        """Test that places are sorted by distance."""
        lat, lon = 37.5407, -77.4360
        places = find_nearby_places(lat, lon, sample_gazetteer, max_distance=200.0)
        
        if len(places) > 1:
            distances = [place["distance"] for place in places]
            assert distances == sorted(distances)

    def test_find_nearby_empty_gazetteer(self):
        """Test finding places with empty gazetteer."""
        empty_gazetteer = {"entries": []}
        places = find_nearby_places(37.5407, -77.4360, empty_gazetteer, max_distance=10.0)
        
        assert places == []

    def test_find_nearby_missing_coordinates(self, sample_gazetteer):
        """Test finding places when entries lack coordinates."""
        incomplete_gazetteer = {
            "entries": [
                {"name": "Place1", "type": "city"},  # No lat/lon
                {"name": "Place2", "type": "city", "lat": 37.5407, "lon": -77.4360}
            ]
        }
        places = find_nearby_places(37.5407, -77.4360, incomplete_gazetteer, max_distance=10.0)
        
        assert len(places) == 1
        assert places[0]["name"] == "Place2"

    def test_find_nearby_default_max_distance(self, sample_gazetteer):
        """Test finding places with default max distance."""
        lat, lon = 37.5407, -77.4360
        places = find_nearby_places(lat, lon, sample_gazetteer)
        
        for place in places:
            assert place["distance"] <= 10.0
