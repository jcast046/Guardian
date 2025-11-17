"""Unified clustering interface for geographic coordinates.

Provides a consistent interface for K-Means, DBSCAN, and KDE clustering algorithms,
with support for coordinate transformations, hotspot extraction, and reproducible
parameter hashing.
"""

from clustering.base import Clusterer
from clustering.kmeans import KMeansClustering
from clustering.dbscan import DBSCANClustering
from clustering.kde import KDEClustering
from clustering.utils import (
    to_projected,
    to_geographic,
    validate_coordinates,
    choose_projected_crs,
    validate_crs_units,
    canonical_params_json,
    param_hash_from_json,
    HYPERPARAM_KEYS,
    load_points_df,
    gdf_to_points_json,
    points_json_to_gdf,
    M_PER_MILE,
)


def make_clusterer(name: str, **kwargs) -> Clusterer:
    """Factory function to create clusterer instances.
    
    Args:
        name: Algorithm name ("kmeans", "dbscan", or "kde").
        **kwargs: Algorithm-specific parameters.
        
    Returns:
        Clusterer instance.
        
    Raises:
        ValueError: If algorithm name is unknown.
        
    Examples:
        >>> clusterer = make_clusterer("kmeans", n_clusters=5, random_state=42)
        >>> clusterer = make_clusterer("dbscan", eps_meters=1000.0, min_samples=5)
        >>> clusterer = make_clusterer("kde", bandwidth_meters=1500.0, iso_mass=0.90)
    """
    if name == "kmeans":
        return KMeansClustering(**kwargs)
    elif name == "dbscan":
        return DBSCANClustering(**kwargs)
    elif name == "kde":
        return KDEClustering(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {name}. Must be one of: kmeans, dbscan, kde")


__all__ = [
    "Clusterer",
    "KMeansClustering",
    "DBSCANClustering",
    "KDEClustering",
    "make_clusterer",
    "to_projected",
    "to_geographic",
    "validate_coordinates",
    "choose_projected_crs",
    "validate_crs_units",
    "canonical_params_json",
    "param_hash_from_json",
    "HYPERPARAM_KEYS",
    "load_points_df",
    "gdf_to_points_json",
    "points_json_to_gdf",
    "M_PER_MILE",
]

