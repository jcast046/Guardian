"""Unit tests for metrics.ops module."""

import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock
from metrics.ops import _load_va_boundary


class TestLoadVaBoundary:
    """Test suite for metrics.ops._load_va_boundary function.

    Tests loading Virginia boundary GeoJSON files with various scenarios
    including missing files, invalid JSON, and shapely availability.
    """

    def test_load_va_boundary_valid_geojson(self, tmp_path):
        """Test loading valid GeoJSON boundary file."""
        geojson_file = tmp_path / "va_boundary.geojson"
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-78.0, 36.0], [-77.0, 36.0], [-77.0, 37.0], [-78.0, 37.0], [-78.0, 36.0]]]
                    }
                }
            ]
        }
        geojson_file.write_text(json.dumps(geojson_data), encoding="utf-8")
        
        # Mock shapely availability
        with patch('metrics.ops.SHAPELY', True):
            with patch('metrics.ops.shape') as mock_shape:
                mock_geom = MagicMock()
                mock_shape.return_value = mock_geom
                result = _load_va_boundary(str(geojson_file))
                
                assert isinstance(result, list)

    def test_load_va_boundary_missing_file(self):
        """Test loading boundary when file doesn't exist."""
        with patch('metrics.ops.SHAPELY', True):
            result = _load_va_boundary("nonexistent.geojson")
            
            assert result == []

    def test_load_va_boundary_no_shapely(self, tmp_path):
        """Test loading boundary when shapely is not available."""
        geojson_file = tmp_path / "va_boundary.geojson"
        geojson_file.write_text('{"type": "FeatureCollection"}', encoding="utf-8")
        
        with patch('metrics.ops.SHAPELY', False):
            result = _load_va_boundary(str(geojson_file))
            
            assert result == []

    def test_load_va_boundary_invalid_geojson(self, tmp_path):
        """Test loading invalid GeoJSON file."""
        geojson_file = tmp_path / "va_boundary.geojson"
        geojson_file.write_text('{invalid json}', encoding="utf-8")
        
        with patch('metrics.ops.SHAPELY', True):
            # Should handle JSON decode error gracefully or raise
            #  may return empty list or raise
            try:
                result = _load_va_boundary(str(geojson_file))
                assert isinstance(result, list)
            except json.JSONDecodeError:
                # This is also acceptable behavior
                pass

    def test_load_va_boundary_empty_features(self, tmp_path):
        """Test loading GeoJSON with no features."""
        geojson_file = tmp_path / "va_boundary.geojson"
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
        geojson_file.write_text(json.dumps(geojson_data), encoding="utf-8")
        
        with patch('metrics.ops.SHAPELY', True):
            with patch('metrics.ops.shape') as mock_shape:
                result = _load_va_boundary(str(geojson_file))
                
                assert result == []
