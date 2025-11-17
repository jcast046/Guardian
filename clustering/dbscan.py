"""DBSCAN clustering implementation for geographic coordinates.

Provides DBSCAN clustering with automatic coordinate transformation,
noise detection, and hotspot extraction as centroids or convex hulls.
"""

from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

from clustering.base import Clusterer
from clustering.utils import (
    to_projected,
    to_geographic,
    validate_coordinates,
    validate_crs_units,
)


class DBSCANClustering(Clusterer):
    """DBSCAN clustering for geographic coordinates.
    
    Args:
        eps_meters: Maximum distance in meters for points to be in same cluster (default: 1000.0).
        min_samples: Minimum number of points to form a cluster (default: 5).
        hotspot_mode: Hotspot geometry mode - "centroid" (point-weighted) or "hull" (convex hull) (default: "centroid").
        **kwargs: Additional arguments passed to Clusterer base class.
    """
    
    def __init__(
        self, 
        eps_meters: float = 1000.0, 
        min_samples: int = 5,
        hotspot_mode: str = "centroid",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps_meters = eps_meters
        self.min_samples = min_samples
        self.hotspot_mode = hotspot_mode
        self.method = "dbscan"
        self.model: Optional[DBSCAN] = None
        
        self.params.update({
            "eps_meters": eps_meters,
            "min_samples": min_samples,
            "hotspot_mode": hotspot_mode,
        })
    
    def fit(
        self, 
        df: pd.DataFrame, 
        x_col: str = "lon", 
        y_col: str = "lat"
    ) -> "DBSCANClustering":
        """Fit DBSCAN model to data.
        
        Args:
            df: DataFrame with coordinate columns.
            x_col: Name of longitude/x column.
            y_col: Name of latitude/y column.
            
        Returns:
            self for method chaining.
            
        Raises:
            ValueError: If CRS units are not meters when eps_meters is used.
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
        
        validate_crs_units(self.crs_proj, expected_units="m")
        
        gdf_proj, units = to_projected(df, x_col, y_col, self.crs_in, self.crs_proj)
        self.proj_units = units
        
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        self.gdf_proj_ = gdf_proj.copy()
        
        self.model = DBSCAN(
            eps=self.eps_meters,
            min_samples=self.min_samples,
            metric="euclidean"
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
            GeoDataFrame with hotspot centers or convex hulls.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Run .fit() first.")
        
        unique_labels = np.unique(self.labels_)
        cluster_labels = unique_labels[unique_labels >= 0]
        
        if len(cluster_labels) == 0:
            return gpd.GeoDataFrame(
                columns=["score", "method", "params_hash", "params_json", "crs", "crs_proj", "proj_units"],
                geometry=[],
                crs=self.crs_in
            )
        
        cluster_sizes = []
        for label in cluster_labels:
            size = np.sum(self.labels_ == label)
            cluster_sizes.append(size)
        cluster_sizes = np.array(cluster_sizes)
        scores = cluster_sizes / len(self.labels_)
        
        geometries = []
        hotspot_scores = []
        label_to_score = {label: score for label, score in zip(cluster_labels, scores)}
        
        for label in cluster_labels:
            mask = self.labels_ == label
            cluster_points = self.gdf_proj_[mask]
            
            if len(cluster_points) == 0:
                continue
            
            if self.hotspot_mode == "centroid":
                coords = np.array([[p.x, p.y] for p in cluster_points.geometry])
                mean_coords = coords.mean(axis=0)
                from shapely.geometry import Point
                centroid = Point(mean_coords[0], mean_coords[1])
                geometries.append(centroid)
            elif self.hotspot_mode == "hull":
                from shapely.geometry import MultiPoint
                points = MultiPoint(list(cluster_points.geometry))
                hull = points.convex_hull
                geometries.append(hull)
            else:
                raise ValueError(f"Unknown hotspot_mode: {self.hotspot_mode}")
            
            hotspot_scores.append(label_to_score[label])
        
        gdf = gpd.GeoDataFrame(
            {
                "score": hotspot_scores,
                "method": self.method,
            },
            geometry=geometries,
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

