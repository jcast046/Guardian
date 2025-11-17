"""Clustering evaluation metrics and parameter selection helpers.

Provides evaluation metrics (silhouette score, Davies-Bouldin index),
parameter selection helpers (k-distance plot, bandwidth selection), and
stability analysis for clustering algorithms.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from sklearn.metrics import davies_bouldin_score as sklearn_davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# Import Clusterer type for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from clustering.base import Clusterer
else:
    Clusterer = Any


def silhouette_score(labels: np.ndarray, X: np.ndarray) -> float:
    """Compute silhouette score on projected coordinates.
    
    Args:
        labels: Cluster labels array (shape: (n_samples,)).
        X: Coordinate array in projected CRS (shape: (n_samples, n_features)).
        
    Returns:
        Silhouette score (higher is better, range: -1 to 1).
        
    Note:
        Only computes score for points with labels >= 0 (excludes noise points with -1).
    """
    # Filter out noise points (label == -1)
    mask = labels >= 0
    if mask.sum() < 2:
        # Need at least 2 points for silhouette score
        return -1.0
    
    labels_filtered = labels[mask]
    X_filtered = X[mask]
    
    # Need at least 2 clusters
    unique_labels = np.unique(labels_filtered)
    if len(unique_labels) < 2:
        return -1.0
    
    return sklearn_silhouette_score(X_filtered, labels_filtered)


def davies_bouldin_score(labels: np.ndarray, X: np.ndarray) -> float:
    """Compute Davies-Bouldin index on projected coordinates.
    
    Args:
        labels: Cluster labels array (shape: (n_samples,)).
        X: Coordinate array in projected CRS (shape: (n_samples, n_features)).
        
    Returns:
        Davies-Bouldin index (lower is better, range: 0 to infinity).
        
    Note:
        Only computes score for points with labels >= 0 (excludes noise points with -1).
    """
    # Filter out noise points (label == -1)
    mask = labels >= 0
    if mask.sum() < 2:
        # Need at least 2 points
        return np.inf
    
    labels_filtered = labels[mask]
    X_filtered = X[mask]
    
    # Need at least 2 clusters
    unique_labels = np.unique(labels_filtered)
    if len(unique_labels) < 2:
        return np.inf
    
    return sklearn_davies_bouldin_score(X_filtered, labels_filtered)


def k_distance_plot(
    X: np.ndarray, 
    k: int = 4, 
    sample_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k-distance for DBSCAN parameter selection.
    
    Args:
        X: Coordinate array in projected CRS (shape: (n_samples, n_features)).
        k: Number of neighbors (default: 4, range: 4-8 recommended).
        sample_size: Optional sample size for large datasets (default: None = use all).
        
    Returns:
        Tuple of (sorted_distances, indices) for plotting.
        - sorted_distances: Sorted k-distance values (ascending).
        - indices: Indices corresponding to sorted distances.
    """
    if sample_size is not None and sample_size < len(X):
        # Sample points for efficiency
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
        indices = np.arange(len(X))
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_sample)  # +1 because point is its own neighbor
    distances, _ = nbrs.kneighbors(X_sample)
    
    # Get k-distance (distance to k-th neighbor, excluding self)
    k_distances = distances[:, k]  # k-th neighbor (0-indexed, so k is the (k+1)-th point)
    
    # Sort distances
    sorted_indices = np.argsort(k_distances)
    sorted_distances = k_distances[sorted_indices]
    
    return sorted_distances, sorted_indices


def bandwidth_selection(
    X: np.ndarray, 
    method: str = "scott", 
    cv: bool = False
) -> float:
    """Select bandwidth for KDE.
    
    Args:
        X: Coordinate array in projected CRS (shape: (n_samples, n_features)).
        method: Bandwidth selection method - "scott", "silverman", or "cv" (default: "scott").
        cv: If True, use cross-validation (overrides method, default: False).
        
    Returns:
        Optimal bandwidth in meters (same units as X).
    """
    if cv:
        # Cross-validation approach
        # Use GridSearchCV to find optimal bandwidth
        bandwidths = np.logspace(-1, 2, 20)  # Range from 0.1 to 100 (adjust based on data scale)
        params = {'bandwidth': bandwidths}
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            params,
            cv=5
        )
        grid.fit(X)
        return grid.best_params_['bandwidth']
    
    elif method == "scott":
        # Scott's rule: h = n^(-1/(d+4)) * std
        n, d = X.shape
        std = np.std(X, axis=0).mean()  # Average standard deviation across dimensions
        h = n ** (-1.0 / (d + 4)) * std
        return float(h)
    
    elif method == "silverman":
        # Silverman's rule: h = (n*(d+2)/4)^(-1/(d+4)) * std
        n, d = X.shape
        std = np.std(X, axis=0).mean()  # Average standard deviation across dimensions
        h = (n * (d + 2) / 4) ** (-1.0 / (d + 4)) * std
        return float(h)
    
    else:
        raise ValueError(f"Unknown bandwidth selection method: {method}")


def bootstrap_stability(
    clusterer: Clusterer,
    df: Any,  # pd.DataFrame, but avoid pandas import in type hints
    n_iter: int = 10,
    sample_ratio: float = 0.85
) -> Dict[str, Any]:
    """Bootstrap stability analysis.
    
    Resamples data points and computes stability metrics across multiple runs.
    
    Args:
        clusterer: Clusterer instance (must be fitted or will be fitted on each iteration).
        df: DataFrame with coordinate columns (same format as fit() input).
        n_iter: Number of bootstrap iterations (default: 10).
        sample_ratio: Fraction of points to sample in each iteration (default: 0.85).
        
    Returns:
        Dictionary with stability metrics:
        - jaccard_overlap: Mean Jaccard overlap of hotspot polygons across iterations.
        - ari_scores: List of Adjusted Rand Index scores between iterations.
        - mean_ari: Mean ARI across all pairs of iterations.
        - std_ari: Standard deviation of ARI scores.
    """
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    # Get coordinate columns
    x_col = "lon"
    y_col = "lat"
    if hasattr(clusterer, 'x_col'):
        x_col = clusterer.x_col
    if hasattr(clusterer, 'y_col'):
        y_col = clusterer.y_col
    
    n_samples = len(df)
    sample_size = int(n_samples * sample_ratio)
    
    # Store results from each iteration
    hotspot_gdfs = []
    label_arrays = []
    
    for i in range(n_iter):
        # Sample data
        sampled_indices = np.random.choice(n_samples, size=sample_size, replace=False)
        df_sample = df.iloc[sampled_indices].reset_index(drop=True)
        
        # Create new clusterer instance with same parameters
        # We need to clone the clusterer
        import copy
        clusterer_copy = copy.deepcopy(clusterer)
        clusterer_copy.labels_ = None  # Reset labels
        
        # Fit on sampled data
        clusterer_copy.fit(df_sample, x_col=x_col, y_col=y_col)
        
        # Get hotspots and labels
        hotspots = clusterer_copy.hotspots()
        labels = clusterer_copy.labels()
        
        hotspot_gdfs.append(hotspots)
        label_arrays.append(labels)
    
    # Compute Jaccard overlap of hotspot polygons
    jaccard_overlaps = []
    for i in range(n_iter):
        for j in range(i + 1, n_iter):
            gdf1 = hotspot_gdfs[i]
            gdf2 = hotspot_gdfs[j]
            
            # Compute Jaccard overlap for each polygon pair
            # For simplicity, compute overlap of union of all polygons
            if len(gdf1) > 0 and len(gdf2) > 0:
                union1 = gdf1.geometry.unary_union
                union2 = gdf2.geometry.unary_union
                
                if union1.area > 0 and union2.area > 0:
                    intersection = union1.intersection(union2)
                    union = union1.union(union2)
                    jaccard = intersection.area / union.area if union.area > 0 else 0.0
                    jaccard_overlaps.append(jaccard)
    
    mean_jaccard = np.mean(jaccard_overlaps) if jaccard_overlaps else 0.0
    
    # Compute Adjusted Rand Index between label arrays
    # Note: This requires mapping labels back to original data indices
    # For now, compute ARI only if can align the samples
    # This is a simplified version - a full implementation would require
    # storing the original data indices for each bootstrap sample
    ari_scores = []
    # TODO: Implement proper ARI computation for bootstrapped samples
    # This would require:
    # 1. Storing original indices for each bootstrap sample
    # 2. Computing ARI on the intersection of indices
    # 3. Mapping labels back to original data space
    
    return {
        "jaccard_overlap": mean_jaccard,
        "ari_scores": ari_scores,
        "mean_ari": np.mean(ari_scores) if ari_scores else None,
        "std_ari": np.std(ari_scores) if ari_scores else None,
    }

