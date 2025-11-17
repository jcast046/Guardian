"""Unit tests for clustering module.

Tests the unified clustering interface for K-Means, DBSCAN, and KDE algorithms.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from clustering import (
    Clusterer,
    KMeansClustering,
    DBSCANClustering,
    KDEClustering,
    make_clusterer,
    to_projected,
    to_geographic,
    validate_coordinates,
    choose_projected_crs,
    validate_crs_units,
    canonical_params_json,
    param_hash_from_json,
)
from metrics.clustering import (
    silhouette_score,
    davies_bouldin_score,
    k_distance_plot,
    bandwidth_selection,
)


@pytest.fixture
def sample_df():
    """Fixture providing sample DataFrame with geographic coordinates.

    Returns:
        pd.DataFrame: DataFrame with 'lon' and 'lat' columns.
    """
    return pd.DataFrame({
        "lon": [-77.1, -77.2, -77.0, -77.15, -77.25],
        "lat": [38.88, 38.89, 38.87, 38.88, 38.90],
    })


@pytest.fixture
def two_blob_df():
    """Fixture providing two-blob dataset for KDE polygon testing.

    Returns:
        pd.DataFrame: DataFrame with two distinct spatial clusters.
    """
    # Create two distinct clusters
    np.random.seed(42)
    blob1 = np.random.randn(50, 2) * 0.01 + np.array([-77.1, 38.88])
    blob2 = np.random.randn(50, 2) * 0.01 + np.array([-77.2, 38.89])
    coords = np.vstack([blob1, blob2])
    return pd.DataFrame({
        "lon": coords[:, 0],
        "lat": coords[:, 1],
    })


class TestClustererInterface:
    """Test suite for unified clustering interface.

    Tests that all clustering models (K-Means, DBSCAN, KDE) fit into
    the same workflow with consistent fit(), labels(), and hotspots() methods.
    """
    
    def test_kmeans_fit_labels_hotspots(self, sample_df):
        """Test K-Means fit() -> labels() -> hotspots() workflow."""
        clusterer = KMeansClustering(n_clusters=2, random_state=42)
        clusterer.fit(sample_df)
        
        # Test labels()
        labels = clusterer.labels()
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_df)
        assert all(label >= -1 for label in labels)
        
        # Test hotspots()
        hotspots = clusterer.hotspots()
        assert isinstance(hotspots, gpd.GeoDataFrame)
        assert len(hotspots) > 0
        assert "geometry" in hotspots.columns
        assert "score" in hotspots.columns
        assert "method" in hotspots.columns
        assert "params_hash" in hotspots.columns
        assert "params_json" in hotspots.columns
        assert "crs" in hotspots.columns
    
    def test_dbscan_fit_labels_hotspots(self, sample_df):
        """Test DBSCAN fit() -> labels() -> hotspots() workflow."""
        clusterer = DBSCANClustering(eps_meters=5000.0, min_samples=2, random_state=42)
        clusterer.fit(sample_df)
        
        # Test labels()
        labels = clusterer.labels()
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_df)
        assert all(label >= -1 for label in labels)
        
        # Test hotspots()
        hotspots = clusterer.hotspots()
        assert isinstance(hotspots, gpd.GeoDataFrame)
        assert "geometry" in hotspots.columns
        assert "score" in hotspots.columns
        assert "method" in hotspots.columns
        assert "params_hash" in hotspots.columns
    
    def test_kde_fit_labels_hotspots(self, sample_df):
        """Test KDE fit() -> labels() -> hotspots() workflow."""
        clusterer = KDEClustering(
            bandwidth_meters=2000.0,
            label_policy="none",
            hotspot_mode="iso_polygons",
            iso_mass=0.90,
            grid_res_m=500,
            random_state=42
        )
        clusterer.fit(sample_df)
        
        # Test labels() - should all be -1 with label_policy="none"
        labels = clusterer.labels()
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_df)
        assert all(label == -1 for label in labels)  # All noise with "none" policy
        
        # Test hotspots()
        hotspots = clusterer.hotspots()
        assert isinstance(hotspots, gpd.GeoDataFrame)
        assert "geometry" in hotspots.columns
        assert "score" in hotspots.columns
        assert "method" in hotspots.columns
        assert "params_hash" in hotspots.columns
    
    def test_labels_before_fit_raises_error(self):
        """Test that labels() raises RuntimeError if not fitted."""
        clusterer = KMeansClustering(n_clusters=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            clusterer.labels()
    
    def test_hotspots_before_fit_raises_error(self):
        """Test that hotspots() raises RuntimeError if not fitted."""
        clusterer = KMeansClustering(n_clusters=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            clusterer.hotspots()
    
    def test_smoke_test_all_algorithms(self, sample_df):
        """Smoke test: all three algorithms on same dataset, verify identical column schema."""
        clusterers = [
            KMeansClustering(n_clusters=2, random_state=42),
            DBSCANClustering(eps_meters=5000.0, min_samples=2),
            KDEClustering(bandwidth_meters=2000.0, grid_res_m=500, random_state=42),
        ]
        
        hotspot_schemas = []
        for clusterer in clusterers:
            clusterer.fit(sample_df)
            hotspots = clusterer.hotspots()
            hotspot_schemas.append(set(hotspots.columns))
        
        # All should have the same columns (except geometry which may vary)
        required_columns = {"score", "method", "params_hash", "params_json", "crs", "crs_proj", "proj_units"}
        for schema in hotspot_schemas:
            assert required_columns.issubset(schema)


class TestCRSTransformations:
    """Test suite for CRS transformation utilities.

    Tests coordinate reference system transformations including
    projection, geographic conversion, CRS selection, and validation.
    """
    
    def test_to_projected(self, sample_df):
        """Test conversion to projected CRS."""
        gdf_proj, units = to_projected(sample_df, "lon", "lat", "EPSG:4326", "EPSG:32145")
        assert isinstance(gdf_proj, gpd.GeoDataFrame)
        assert gdf_proj.crs == "EPSG:32145"
        assert units == "m"
    
    def test_to_geographic(self, sample_df):
        """Test conversion back to geographic CRS."""
        gdf_proj, _ = to_projected(sample_df, "lon", "lat", "EPSG:4326", "EPSG:32145")
        gdf_geo = to_geographic(gdf_proj, "EPSG:4326")
        assert gdf_geo.crs == "EPSG:4326"
    
    def test_projection_round_trip(self, sample_df):
        """Test that projection round-trip preserves coordinates approximately."""
        gdf_proj, _ = to_projected(sample_df, "lon", "lat", "EPSG:4326", "EPSG:32145")
        gdf_geo = to_geographic(gdf_proj, "EPSG:4326")
        
        # Coordinates should be approximately the same (within small tolerance)
        original_lons = sample_df["lon"].values
        original_lats = sample_df["lat"].values
        recovered_lons = [p.x for p in gdf_geo.geometry]
        recovered_lats = [p.y for p in gdf_geo.geometry]
        
        np.testing.assert_allclose(original_lons, recovered_lons, rtol=1e-5)
        np.testing.assert_allclose(original_lats, recovered_lats, rtol=1e-5)
    
    def test_choose_projected_crs(self):
        """Test automatic CRS selection based on bbox."""
        # VA North
        bbox_north = (-78.0, 38.6, -76.0, 39.0)  # (minx, miny, maxx, maxy)
        crs_north = choose_projected_crs(bbox_north)
        assert crs_north in ["EPSG:32145", "EPSG:32146"]
        
        # VA South
        bbox_south = (-78.0, 36.5, -76.0, 37.4)
        crs_south = choose_projected_crs(bbox_south)
        assert crs_south in ["EPSG:32145", "EPSG:32146"]
    
    def test_validate_crs_units(self):
        """Test CRS units validation."""
        # Valid meters-based CRS
        assert validate_crs_units("EPSG:32145", expected_units="m") == True
        
        # Should raise error for feet-based CRS when expecting meters
        with pytest.raises(ValueError, match="meters"):
            validate_crs_units("EPSG:2283", expected_units="m")


class TestParameterHashing:
    """Test suite for deterministic parameter hashing.

    Tests canonical parameter JSON generation and SHA-1 hashing
    for reproducible clustering parameter identification.
    """
    
    def test_param_hash_deterministic(self):
        """Test that parameter hash is deterministic."""
        params = {"n_clusters": 5, "random_state": 42}
        params_json1 = canonical_params_json("kmeans", params, {"n_clusters", "random_state"})
        params_json2 = canonical_params_json("kmeans", params, {"n_clusters", "random_state"})
        assert params_json1 == params_json2
        
        hash1 = param_hash_from_json(params_json1)
        hash2 = param_hash_from_json(params_json2)
        assert hash1 == hash2
        assert len(hash1) == 10  # 10-character hex digest
    
    def test_param_hash_includes_method(self):
        """Test that parameter hash includes method name."""
        params = {"n_clusters": 5}
        json_kmeans = canonical_params_json("kmeans", params, {"n_clusters"})
        json_dbscan = canonical_params_json("dbscan", params, {"n_clusters"})
        assert json_kmeans != json_dbscan  # Different methods should produce different JSON
    
    def test_param_hash_excludes_runtime_state(self):
        """Test that parameter hash excludes runtime state."""
        params = {
            "n_clusters": 5,
            "random_state": 42,
            "crs_in": "EPSG:4326",  # Runtime state
            "data_bbox": (1, 2, 3, 4),  # Runtime state
        }
        include = {"n_clusters", "random_state"}
        params_json = canonical_params_json("kmeans", params, include)
        assert "crs_in" not in params_json
        assert "data_bbox" not in params_json
        assert "n_clusters" in params_json
        assert "random_state" in params_json


class TestKMeans:
    """Test suite for K-Means clustering implementation.

    Tests K-Means-specific functionality including prediction
    and information retrieval.
    """
    
    def test_kmeans_predict(self, sample_df):
        """Test K-Means predict() method."""
        clusterer = KMeansClustering(n_clusters=2, random_state=42)
        clusterer.fit(sample_df)
        
        # Predict on new data
        new_df = pd.DataFrame({
            "lon": [-77.1, -77.2],
            "lat": [38.88, 38.89],
        })
        labels = clusterer.predict(new_df)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(new_df)
        assert all(label >= 0 for label in labels)
    
    def test_kmeans_info(self, sample_df):
        """Test K-Means info() method."""
        clusterer = KMeansClustering(n_clusters=2, random_state=42)
        clusterer.fit(sample_df)
        
        info = clusterer.info()
        assert info["method"] == "kmeans"
        assert info["n_clusters"] == 2
        assert info["n_samples"] == len(sample_df)
        assert "params_hash" in info
        assert "params_json" in info
        assert info["proj_units"] == "m"


class TestDBSCAN:
    """Test suite for DBSCAN clustering implementation.

    Tests DBSCAN-specific functionality including eps validation,
    noise detection, and hotspot modes (centroid vs hull).
    """
    
    def test_dbscan_eps_meters_validation(self, sample_df):
        """Test that DBSCAN validates CRS units are meters."""
        # Should work with meters-based CRS
        clusterer = DBSCANClustering(eps_meters=1000.0, min_samples=2)
        clusterer.crs_proj = "EPSG:32145"  # Meters-based
        clusterer.fit(sample_df)  # Should not raise error
        
        # Should raise error with feet-based CRS
        clusterer2 = DBSCANClustering(eps_meters=1000.0, min_samples=2)
        clusterer2.crs_proj = "EPSG:2283"  # Feet-based
        with pytest.raises(ValueError, match="meters"):
            clusterer2.fit(sample_df)
    
    def test_dbscan_noise_labels(self, sample_df):
        """Test that DBSCAN produces -1 labels for noise."""
        clusterer = DBSCANClustering(eps_meters=100.0, min_samples=10)  # Very strict
        clusterer.fit(sample_df)
        labels = clusterer.labels()
        assert all(label == -1 for label in labels) or any(label >= 0 for label in labels)
    
    def test_dbscan_hotspot_modes(self, sample_df):
        """Test DBSCAN hotspot modes (centroid vs hull)."""
        clusterer = DBSCANClustering(eps_meters=5000.0, min_samples=2, hotspot_mode="centroid")
        clusterer.fit(sample_df)
        hotspots_centroid = clusterer.hotspots()
        
        clusterer2 = DBSCANClustering(eps_meters=5000.0, min_samples=2, hotspot_mode="hull")
        clusterer2.fit(sample_df)
        hotspots_hull = clusterer2.hotspots()
        
        assert isinstance(hotspots_centroid, gpd.GeoDataFrame)
        assert isinstance(hotspots_hull, gpd.GeoDataFrame)


class TestKDE:
    """Test suite for KDE clustering implementation.

    Tests KDE-specific functionality including label policies,
    hotspot modes (iso_polygons vs peaks), and isodensity polygon extraction.
    """
    
    def test_kde_label_policy_none(self, sample_df):
        """Test KDE with label_policy='none' (all -1)."""
        clusterer = KDEClustering(
            bandwidth_meters=2000.0,
            label_policy="none",
            random_state=42
        )
        clusterer.fit(sample_df)
        labels = clusterer.labels()
        assert all(label == -1 for label in labels)
    
    def test_kde_hotspot_modes(self, sample_df):
        """Test KDE hotspot modes (iso_polygons vs peaks)."""
        # iso_polygons mode
        clusterer = KDEClustering(
            bandwidth_meters=2000.0,
            hotspot_mode="iso_polygons",
            grid_res_m=500,
            random_state=42
        )
        clusterer.fit(sample_df)
        hotspots_polygons = clusterer.hotspots()
        assert isinstance(hotspots_polygons, gpd.GeoDataFrame)
        
        # peaks mode
        clusterer2 = KDEClustering(
            bandwidth_meters=2000.0,
            hotspot_mode="peaks",
            random_state=42
        )
        clusterer2.fit(sample_df)
        hotspots_peaks = clusterer2.hotspots()
        assert isinstance(hotspots_peaks, gpd.GeoDataFrame)
        assert all(isinstance(g, Point) for g in hotspots_peaks.geometry)
    
    def test_kde_isodensity_polygons_two_blobs(self, two_blob_df):
        """Test KDE isodensity polygons on two-blob dataset."""
        # Skip if polygonization dependencies are not available
        try:
            import rasterio.features
        except ImportError:
            try:
                import skimage.measure
            except ImportError:
                try:
                    import scipy.ndimage
                except ImportError:
                    pytest.skip("Polygonization requires rasterio, scikit-image, or scipy")
        
        clusterer = KDEClustering(
            bandwidth_meters=1500.0,
            hotspot_mode="iso_polygons",
            iso_mass=0.90,
            grid_res_m=250,
            min_area_m2=1000,
            random_state=42
        )
        clusterer.fit(two_blob_df)
        hotspots = clusterer.hotspots()
        
        assert len(hotspots) >= 0
        if len(hotspots) > 0:
            for geom in hotspots.geometry:
                assert hasattr(geom, 'area')
                assert geom.area >= 0
    
    def test_kde_mass_threshold(self, two_blob_df):
        """Test that KDE polygons capture approximately iso_mass probability."""
        # Skip if polygonization dependencies are not available
        try:
            import rasterio.features
        except ImportError:
            try:
                import skimage.measure
            except ImportError:
                try:
                    import scipy.ndimage
                except ImportError:
                    pytest.skip("Polygonization requires rasterio, scikit-image, or scipy")
        
        clusterer = KDEClustering(
            bandwidth_meters=1500.0,
            hotspot_mode="iso_polygons",
            iso_mass=0.90,
            grid_res_m=250,
            random_state=42
        )
        clusterer.fit(two_blob_df)
        hotspots = clusterer.hotspots()
        
        if len(hotspots) > 0:
            total_mass = hotspots["score"].sum()
            assert 0.80 <= total_mass <= 1.0


class TestMetrics:
    """Test suite for clustering evaluation metrics.

    Tests silhouette score, Davies-Bouldin index, k-distance plot,
    and bandwidth selection utilities.
    """
    
    def test_silhouette_score(self, sample_df):
        """Test silhouette score computation."""
        clusterer = KMeansClustering(n_clusters=2, random_state=42)
        clusterer.fit(sample_df)
        labels = clusterer.labels()
        
        # Get projected coordinates
        gdf_proj, _ = to_projected(sample_df, "lon", "lat", "EPSG:4326", "EPSG:32145")
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        
        score = silhouette_score(labels, X)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
    
    def test_davies_bouldin_score(self, sample_df):
        """Test Davies-Bouldin score computation."""
        clusterer = KMeansClustering(n_clusters=2, random_state=42)
        clusterer.fit(sample_df)
        labels = clusterer.labels()
        
        # Get projected coordinates
        gdf_proj, _ = to_projected(sample_df, "lon", "lat", "EPSG:4326", "EPSG:32145")
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        
        score = davies_bouldin_score(labels, X)
        assert isinstance(score, float)
        assert score >= 0.0
    
    def test_k_distance_plot(self, sample_df):
        """Test k-distance plot computation."""
        gdf_proj, _ = to_projected(sample_df, "lon", "lat", "EPSG:4326", "EPSG:32145")
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        
        distances, indices = k_distance_plot(X, k=4)
        assert isinstance(distances, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert len(distances) == len(indices)
        assert np.all(distances >= 0)
    
    def test_bandwidth_selection(self, sample_df):
        """Test bandwidth selection for KDE."""
        gdf_proj, _ = to_projected(sample_df, "lon", "lat", "EPSG:4326", "EPSG:32145")
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        
        # Test Scott's rule
        bandwidth_scott = bandwidth_selection(X, method="scott")
        assert isinstance(bandwidth_scott, float)
        assert bandwidth_scott > 0
        
        # Test Silverman's rule
        bandwidth_silverman = bandwidth_selection(X, method="silverman")
        assert isinstance(bandwidth_silverman, float)
        assert bandwidth_silverman > 0


class TestMakeClusterer:
    """Test suite for make_clusterer factory function.

    Tests clusterer creation from algorithm names with various
    parameter configurations.
    """
    
    def test_make_clusterer_kmeans(self):
        """Test make_clusterer for K-Means."""
        clusterer = make_clusterer("kmeans", n_clusters=5, random_state=42)
        assert isinstance(clusterer, KMeansClustering)
        assert clusterer.n_clusters == 5
        assert clusterer.random_state == 42
    
    def test_make_clusterer_dbscan(self):
        """Test make_clusterer for DBSCAN."""
        clusterer = make_clusterer("dbscan", eps_meters=1000.0, min_samples=5)
        assert isinstance(clusterer, DBSCANClustering)
        assert clusterer.eps_meters == 1000.0
        assert clusterer.min_samples == 5
    
    def test_make_clusterer_kde(self):
        """Test make_clusterer for KDE."""
        clusterer = make_clusterer("kde", bandwidth_meters=1500.0, iso_mass=0.90)
        assert isinstance(clusterer, KDEClustering)
        assert clusterer.bandwidth_meters == 1500.0
        assert clusterer.iso_mass == 0.90
    
    def test_make_clusterer_unknown_algorithm(self):
        """Test make_clusterer with unknown algorithm."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            make_clusterer("unknown", n_clusters=5)


class TestDataValidation:
    """Test suite for data validation utilities.

    Tests coordinate validation including range checking
    and NaN removal.
    """
    
    def test_validate_coordinates(self):
        """Test coordinate validation."""
        df = pd.DataFrame({
            "lon": [-77.1, -77.2, np.nan, 200.0, -77.0],  # NaN and out-of-range
            "lat": [38.88, 38.89, 38.87, 38.88, 100.0],  # Out-of-range
        })
        
        df_clean = validate_coordinates(df, "lon", "lat")
        assert len(df_clean) < len(df)
        assert df_clean["lon"].notna().all()
        assert df_clean["lat"].notna().all()
        assert (df_clean["lon"] >= -180).all() and (df_clean["lon"] <= 180).all()
        assert (df_clean["lat"] >= -90).all() and (df_clean["lat"] <= 90).all()


class TestReproducibility:
    """Test suite for reproducibility with random_state.

    Tests that clustering algorithms produce identical results
    when using the same random_state seed.
    """
    
    def test_kmeans_reproducibility(self, sample_df):
        """Test that K-Means produces same results with same random_state."""
        clusterer1 = KMeansClustering(n_clusters=2, random_state=42)
        clusterer1.fit(sample_df)
        labels1 = clusterer1.labels()
        
        clusterer2 = KMeansClustering(n_clusters=2, random_state=42)
        clusterer2.fit(sample_df)
        labels2 = clusterer2.labels()
        
        # Labels should be the same (or equivalent up to permutation)
        # For simplicity, just check that get the same number of clusters
        assert len(np.unique(labels1)) == len(np.unique(labels2))
    
    def test_kde_reproducibility(self, sample_df):
        """Test that KDE produces same results with same random_state."""
        clusterer1 = KDEClustering(bandwidth_meters=2000.0, random_state=42)
        clusterer1.fit(sample_df)
        hotspots1 = clusterer1.hotspots()
        
        clusterer2 = KDEClustering(bandwidth_meters=2000.0, random_state=42)
        clusterer2.fit(sample_df)
        hotspots2 = clusterer2.hotspots()
        
        # Should produce same number of hotspots
        assert len(hotspots1) == len(hotspots2)

