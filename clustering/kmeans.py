"""K-Means clustering implementation for geographic coordinates.

Provides K-Means clustering with automatic coordinate transformation,
hotspot extraction from cluster centers, and support for prediction
on new data points.
"""

from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans

from clustering.base import Clusterer
from clustering.utils import to_projected, to_geographic, validate_coordinates


class KMeansClustering(Clusterer):
    """K-Means clustering for geographic coordinates.
    
    Args:
        n_clusters: Number of clusters (default: 5).
        random_state: Random seed for reproducibility (default: 42).
        **kwargs: Additional arguments passed to Clusterer base class.
    """
    
    def __init__(
        self, 
        n_clusters: int = 5, 
        random_state: int = 42, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.method = "kmeans"
        self.model: Optional[KMeans] = None
        
        self.params.update({
            "n_clusters": n_clusters,
            "random_state": random_state,
        })
    
    def fit(
        self, 
        df: pd.DataFrame, 
        x_col: str = "lon", 
        y_col: str = "lat"
    ) -> "KMeansClustering":
        """Fit K-Means model to data.
        
        Args:
            df: DataFrame with coordinate columns.
            x_col: Name of longitude/x column.
            y_col: Name of latitude/y column.
            
        Returns:
            self for method chaining.
        """
        # Validate and clean coordinates
        df = validate_coordinates(df, x_col, y_col)
        
        # Store data bbox (in geographic CRS)
        self.data_bbox = (
            df[x_col].min(), df[y_col].min(),
            df[x_col].max(), df[y_col].max()
        )
        
        if self.crs_proj is None:
            from clustering.utils import choose_projected_crs
            self.crs_proj = choose_projected_crs(self.data_bbox)
        
        gdf_proj, units = to_projected(df, x_col, y_col, self.crs_in, self.crs_proj)
        self.proj_units = units
        
        self.gdf_proj_ = gdf_proj.copy()
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        self.X_proj_ = X
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.labels_ = self.model.fit_predict(X)
        self.n_samples = len(self.labels_)
        
        return self
    
    def hotspots(
        self, 
        top_n: Optional[int] = None, 
        min_score: Optional[float] = None
    ) -> gpd.GeoDataFrame:
        """Extract cluster centers as hotspots.
        
        Args:
            top_n: Limit number of hotspots (optional).
            min_score: Filter by minimum score (optional).
            
        Returns:
            GeoDataFrame with hotspot centers.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Run .fit() first.")
        
        centers = self.model.cluster_centers_
        n_clusters = len(centers)
        
        cluster_sizes = np.bincount(self.labels_, minlength=n_clusters)
        scores = cluster_sizes / len(self.labels_)
        
        geometry = [Point(x, y) for x, y in centers]
        gdf = gpd.GeoDataFrame(
            {
                "score": scores,
                "method": self.method,
            },
            geometry=geometry,
            crs=self.crs_proj
        )
        
        gdf = to_geographic(gdf, self.crs_in)
        
        info = self.info()
        gdf["params_hash"] = info["params_hash"]
        gdf["params_json"] = info["params_json"]
        gdf["crs"] = self.crs_in
        gdf["crs_proj"] = self.crs_proj
        gdf["proj_units"] = self.proj_units
        
        if top_n is not None:
            gdf = gdf.nlargest(top_n, "score")
        elif min_score is not None:
            gdf = gdf[gdf["score"] >= min_score]
        
        return gdf.reset_index(drop=True)
    
    def predict(
        self, 
        df: pd.DataFrame, 
        x_col: str = "lon", 
        y_col: str = "lat"
    ) -> np.ndarray:
        """Predict cluster labels for new points.
        
        Args:
            df: DataFrame with coordinate columns.
            x_col: Name of longitude/x column.
            y_col: Name of latitude/y column.
            
        Returns:
            Array of cluster labels for new data points.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Run .fit() first.")
        
        df = validate_coordinates(df, x_col, y_col)
        gdf_proj, _ = to_projected(df, x_col, y_col, self.crs_in, self.crs_proj)
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        
        labels = self.model.predict(X)
        return labels

