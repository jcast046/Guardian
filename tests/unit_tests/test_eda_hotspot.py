"""Unit tests for eda_hotspot.py module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import eda_hotspot


class TestLoadCases:
    """Test suite for eda_hotspot.load_cases function.

    Tests loading case data from JSONL/CSV files with state filtering,
    coordinate validation, and age band creation.
    """

    def test_load_cases_jsonl(self, tmp_path):
        """Test loading cases from JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"age": 15, "gender": "M", "county": "Richmond", "lat": 38.88, "lon": -77.1}\n'
            '{"age": 12, "gender": "F", "county": "Fairfax", "lat": 38.85, "lon": -77.3}\n',
            encoding="utf-8"
        )
        
        df = eda_hotspot.load_cases(str(jsonl_file), None)
        
        assert len(df) == 2
        assert "age" in df.columns
        assert "gender" in df.columns
        assert "county" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns

    def test_load_cases_csv(self, tmp_path):
        """Test loading cases from CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_content = "age,gender,county,lat,lon\n15,M,Richmond,38.88,-77.1\n12,F,Fairfax,38.85,-77.3\n"
        csv_file.write_text(csv_content, encoding="utf-8")
        
        df = eda_hotspot.load_cases(str(csv_file), None)
        
        assert len(df) == 2
        assert "age" in df.columns

    def test_load_cases_state_filter(self, tmp_path):
        """Test loading cases with state filter."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"age": 15, "gender": "M", "county": "Richmond", "lat": 38.88, "lon": -77.1, "state": "VA"}\n'
            '{"age": 12, "gender": "F", "county": "Fairfax", "lat": 38.85, "lon": -77.3, "state": "CA"}\n',
            encoding="utf-8"
        )
        
        df = eda_hotspot.load_cases(str(jsonl_file), "VA")
        
        assert len(df) == 1
        assert df.iloc[0]["state"] == "VA"

    def test_load_cases_coordinate_validation(self, tmp_path):
        """Test that invalid coordinates are filtered out."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"age": 15, "gender": "M", "county": "Richmond", "lat": 38.88, "lon": -77.1}\n'
            '{"age": 12, "gender": "F", "county": "Fairfax", "lat": 200.0, "lon": -77.3}\n'  # Invalid lat
            '{"age": 10, "gender": "M", "county": "Arlington", "lat": 38.90, "lon": -200.0}\n',  # Invalid lon
            encoding="utf-8"
        )
        
        df = eda_hotspot.load_cases(str(jsonl_file), None)
        
        assert len(df) == 1
        assert df.iloc[0]["county"] == "Richmond"

    def test_load_cases_age_band_creation(self, tmp_path):
        """Test that age bands are created correctly."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"age": 10, "gender": "M", "county": "Richmond", "lat": 38.88, "lon": -77.1}\n'
            '{"age": 15, "gender": "F", "county": "Fairfax", "lat": 38.85, "lon": -77.3}\n',
            encoding="utf-8"
        )
        
        df = eda_hotspot.load_cases(str(jsonl_file), None)
        
        assert "age_band" in df.columns
        assert df[df["age"] == 10]["age_band"].iloc[0] == "≤12"
        assert df[df["age"] == 15]["age_band"].iloc[0] == "13–17"

    def test_load_cases_missing_file(self):
        """Test loading cases when file doesn't exist."""
        with pytest.raises(SystemExit):
            eda_hotspot.load_cases("nonexistent.jsonl", None)

    def test_load_cases_missing_columns(self, tmp_path):
        """Test loading cases with missing required columns."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        
        with pytest.raises(SystemExit):
            eda_hotspot.load_cases(str(jsonl_file), None)


class TestEnsureOutdir:
    """Test suite for eda_hotspot.ensure_outdir function.

    Tests directory creation and handling of existing directories.
    """

    def test_ensure_outdir_creates_directory(self, tmp_path):
        """Test that ensure_outdir creates directory if it doesn't exist."""
        outdir = tmp_path / "test_output"
        
        eda_hotspot.ensure_outdir(str(outdir))
        
        assert outdir.exists()
        assert outdir.is_dir()

    def test_ensure_outdir_existing_directory(self, tmp_path):
        """Test that ensure_outdir handles existing directory."""
        outdir = tmp_path / "test_output"
        outdir.mkdir()
        
        eda_hotspot.ensure_outdir(str(outdir))
        
        assert outdir.exists()


class TestMakeGridFromDf:
    """Test suite for eda_hotspot._make_grid_from_df function.

    Tests grid generation from DataFrame with various configurations
    including padding and resolution parameters.
    """

    def test_make_grid_from_df_basic(self):
        """Test creating grid from DataFrame."""
        df = pd.DataFrame({
            "lat": [38.0, 39.0, 38.5],
            "lon": [-77.0, -76.0, -76.5]
        })
        
        grid = eda_hotspot._make_grid_from_df(df, nx=10, ny=10)
        
        assert grid.shape[0] == 100
        assert grid.shape[1] == 2

    def test_make_grid_from_df_with_padding(self):
        """Test creating grid with padding."""
        df = pd.DataFrame({
            "lat": [38.0, 39.0],
            "lon": [-77.0, -76.0]
        })
        
        grid = eda_hotspot._make_grid_from_df(df, nx=5, ny=5, pad=0.5)
        
        assert grid.shape[0] == 25
        assert grid.min(axis=0)[0] < -77.0  # lon min should be less
        assert grid.max(axis=0)[0] > -76.0  # lon max should be more


class TestExportRlInputs:
    """Test export_rl_inputs function."""

    def test_export_rl_inputs_basic(self, tmp_path):
        """Test exporting RL inputs."""
        df = pd.DataFrame({
            "lat": [38.0, 39.0, 38.5],
            "lon": [-77.0, -76.0, -76.5]
        })
        
        eda_hotspot.export_rl_inputs(df, str(tmp_path), nx=10, ny=10)
        
        # Check that files were created
        assert (tmp_path / "grid_xy.npy").exists()
        assert (tmp_path / "road_cost.npy").exists()
        assert (tmp_path / "seclusion.npy").exists()

    def test_export_rl_inputs_grid_xy(self, tmp_path):
        """Test that grid_xy.npy is created with correct shape."""
        df = pd.DataFrame({
            "lat": [38.0, 39.0],
            "lon": [-77.0, -76.0]
        })
        
        eda_hotspot.export_rl_inputs(df, str(tmp_path), nx=5, ny=5)
        
        grid_xy = np.load(tmp_path / "grid_xy.npy")
        
        assert grid_xy.shape[0] == 25  # 5x5 grid
        assert grid_xy.shape[1] == 2  # lon, lat

    @patch('eda_hotspot.KernelDensity', None)
    def test_export_rl_inputs_no_sklearn(self, tmp_path):
        """Test exporting RL inputs when scikit-learn is not available."""
        df = pd.DataFrame({
            "lat": [38.0, 39.0],
            "lon": [-77.0, -76.0]
        })
        
        # Should not raise error, just skip KDE hotspots
        eda_hotspot.export_rl_inputs(df, str(tmp_path))
        
        # Grid files should still be created
        assert (tmp_path / "grid_xy.npy").exists()


class TestKdeHeat:
    """Test _kde_heat function."""

    @patch('eda_hotspot.KernelDensity')
    def test_kde_heat_basic(self, mock_kde):
        """Test KDE heat computation."""
        # Mock KernelDensity
        mock_instance = MagicMock()
        mock_instance.fit.return_value = None
        mock_instance.score_samples.return_value = np.array([-1.0, -2.0, -3.0])
        mock_kde.return_value = mock_instance
        
        xs = np.array([-77.0, -76.0, -75.0])
        ys = np.array([38.0, 39.0, 40.0])
        
        X, Y, Z = eda_hotspot._kde_heat(xs, ys, bw=1000.0, gridsize=10)
        
        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert Z.shape == (10, 10)

    def test_kde_heat_no_sklearn(self):
        """Test KDE heat when scikit-learn is not available."""
        # Temporarily set KernelDensity to None
        original_kde = eda_hotspot.KernelDensity
        eda_hotspot.KernelDensity = None
        
        try:
            xs = np.array([-77.0, -76.0])
            ys = np.array([38.0, 39.0])
            
            with pytest.raises(RuntimeError):
                eda_hotspot._kde_heat(xs, ys, bw=1000.0)
        finally:
            eda_hotspot.KernelDensity = original_kde

    @patch('eda_hotspot.KernelDensity')
    def test_kde_heat_with_bounds(self, mock_kde):
        """Test KDE heat with fixed bounds."""
        mock_instance = MagicMock()
        mock_instance.fit.return_value = None
        mock_instance.score_samples.return_value = np.array([-1.0, -2.0])
        mock_kde.return_value = mock_instance
        
        xs = np.array([-77.0, -76.0])
        ys = np.array([38.0, 39.0])
        bounds = (-78.0, -75.0, 37.0, 40.0)
        
        X, Y, Z = eda_hotspot._kde_heat(xs, ys, bw=1000.0, bounds=bounds, gridsize=5)
        
        assert X.shape == (5, 5)
        assert Y.shape == (5, 5)
        assert Z.shape == (5, 5)


class TestMaskRasterToPolygon:
    """Test _mask_raster_to_polygon function."""

    @patch('eda_hotspot.gpd')
    def test_mask_raster_to_polygon_basic(self, mock_gpd):
        """Test masking raster to polygon."""
        # Create mock boundary
        mock_boundary = MagicMock()
        mock_boundary.to_crs.return_value = mock_boundary
        mock_boundary.dissolve.return_value.geometry.unary_union.buffer.return_value = MagicMock()
        
        Z = np.ones((10, 10))
        X = np.linspace(-78.0, -76.0, 10)
        Y = np.linspace(37.0, 39.0, 10)
        X, Y = np.meshgrid(X, Y)
        
        # Mock vectorized.covers
        with patch('eda_hotspot.vectorized') as mock_vec:
            mock_vec.covers.return_value = np.ones((10, 10), dtype=bool)
            result = eda_hotspot._mask_raster_to_polygon(Z, X, Y, mock_boundary)
            
            assert result.shape == Z.shape

    def test_mask_raster_to_polygon_no_boundary(self):
        """Test masking raster when boundary is None."""
        Z = np.ones((10, 10))
        X = np.linspace(-78.0, -76.0, 10)
        Y = np.linspace(37.0, 39.0, 10)
        X, Y = np.meshgrid(X, Y)
        
        result = eda_hotspot._mask_raster_to_polygon(Z, X, Y, None)
        
        # Should return original Z
        assert np.array_equal(result, Z)
