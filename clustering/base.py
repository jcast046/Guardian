"""Base clustering interface for unified clustering system.

Defines the abstract base class Clusterer that all clustering algorithms
must implement, providing a consistent interface for fit, labels, and
hotspots operations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd

from clustering.utils import (
    to_projected,
    to_geographic,
    validate_coordinates,
    canonical_params_json,
    param_hash_from_json,
    HYPERPARAM_KEYS,
)


class Clusterer(ABC):
    """Abstract base class for clustering algorithms.
    
    All clustering algorithms must implement this interface to provide
    consistent fit, labels, and hotspots operations.
    
    Attributes:
        crs_in: Input CRS (always EPSG:4326 at API edges).
        crs_proj: Projected CRS for internal calculations (must be meters-based).
        params: Dictionary of algorithm-specific parameters.
        labels_: Optional array of cluster labels (populated by fit()).
        proj_units: Units of projected CRS ("m" for meters).
        n_samples: Number of samples after fitting.
        data_bbox: Bounding box of input data (minx, miny, maxx, maxy).
    """
    
    def __init__(
        self, 
        crs_in: str = "EPSG:4326", 
        crs_proj: Optional[str] = None, 
        **params
    ):
        """Initialize clusterer.
        
        Args:
            crs_in: Input CRS (default: "EPSG:4326").
            crs_proj: Projected CRS for internal calculations (default: None = auto-select based on data).
            **params: Algorithm-specific parameters.
        """
        self.crs_in = crs_in
        self.crs_proj = crs_proj  # Will be set in fit() if None
        self.params = params
        self.labels_: Optional[np.ndarray] = None
        self.proj_units = "m"  # Always meters for projected CRS
        self.n_samples: Optional[int] = None
        self.data_bbox: Optional[Tuple[float, float, float, float]] = None
        
        # Store method name (set by subclasses)
        self.method: Optional[str] = None
    
    @abstractmethod
    def fit(
        self, 
        df: pd.DataFrame, 
        x_col: str = "lon", 
        y_col: str = "lat"
    ) -> "Clusterer":
        """Fit model to input data and populate self.labels_.
        
        Args:
            df: DataFrame with coordinate columns.
            x_col: Name of longitude/x column (default: "lon").
            y_col: Name of latitude/y column (default: "lat").
            
        Returns:
            self for method chaining.
        """
        pass
    
    def labels(self) -> np.ndarray:
        """Return stored cluster labels.
        
        Returns:
            Array of cluster labels (integers >= 0 for clusters, -1 for noise).
            
        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self.labels_ is None:
            raise RuntimeError("Model not fitted. Run .fit() first.")
        return self.labels_
    
    @abstractmethod
    def hotspots(
        self, 
        top_n: Optional[int] = None, 
        min_score: Optional[float] = None
    ) -> gpd.GeoDataFrame:
        """Extract hotspot centers/polygons.
        
        Args:
            top_n: Limit number of hotspots (optional).
            min_score: Filter by minimum score (optional).
            
        Returns:
            GeoDataFrame with columns:
                - geometry: Point or Polygon geometries (in crs_in)
                - score: Normalized score [0,1]
                - method: Algorithm name
                - params_hash: 10-character SHA-1 hash
                - params_json: Canonical JSON string
                - crs: CRS string (always EPSG:4326)
                - crs_proj: Original projected CRS
                - proj_units: Units of projected CRS
        """
        pass
    
    def predict(
        self, 
        df: pd.DataFrame, 
        x_col: str = "lon", 
        y_col: str = "lat"
    ) -> np.ndarray:
        """Predict cluster labels for new points.
        
        Args:
            df: DataFrame with coordinate columns.
            x_col: Name of longitude/x column (default: "lon").
            y_col: Name of latitude/y column (default: "lat").
            
        Returns:
            Array of cluster labels for new data points.
            
        Raises:
            NotImplementedError: If method doesn't support prediction.
        """
        raise NotImplementedError(f"predict() not implemented for {self.method}")
    
    def info(self) -> Dict[str, Any]:
        """Return clusterer information.
        
        Returns:
            Dictionary with method name, params, params_json, params_hash,
            crs_in, crs_proj, proj_units, n_samples, n_clusters, data_bbox, timestamp.
            
        Note:
            Can be called before fitting, but n_samples, n_clusters, and data_bbox
            will be None if not yet fitted.
        """
        if self.method is None:
            raise RuntimeError("Method name not set. This should not happen.")
        
        # Get hyperparameter keys for this method
        include = HYPERPARAM_KEYS.get(self.method, set())
        
        # Create canonical params JSON (works even if not fitted)
        params_json = canonical_params_json(self.method, self.params, include)
        params_hash = param_hash_from_json(params_json)
        
        # Get number of clusters (if applicable and fitted)
        n_clusters = None
        if self.labels_ is not None:
            unique_labels = np.unique(self.labels_)
            n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        
        # Get CRS (may be None if not fitted and not explicitly set)
        crs_proj = self.crs_proj if self.crs_proj is not None else "auto-select"
        
        return {
            "method": self.method,
            "params": self.params,
            "params_json": params_json,
            "params_hash": params_hash,
            "crs_in": self.crs_in,
            "crs_proj": crs_proj,
            "proj_units": self.proj_units,
            "n_samples": self.n_samples,
            "n_clusters": n_clusters,
            "data_bbox": self.data_bbox,
            "timestamp": datetime.now().isoformat(),
        }

