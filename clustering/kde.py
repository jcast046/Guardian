"""KDE (Kernel Density Estimation) clustering implementation for geographic coordinates.

Provides kernel density estimation with automatic coordinate transformation,
isodensity polygon extraction, and density peak detection for hotspot identification.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.neighbors import KernelDensity

from clustering.base import Clusterer
from clustering.utils import (
    to_projected,
    to_geographic,
    validate_coordinates,
    validate_crs_units,
)


class KDEClustering(Clusterer):
    """KDE clustering for geographic coordinates using isodensity polygons.
    
    Args:
        bandwidth_meters: Bandwidth for KDE in meters (default: 1500.0).
        kernel: Kernel type for KDE (default: "gaussian").
        label_policy: Label assignment policy - "none" (all -1) or "threshold" (connected components) (default: "none").
        hotspot_mode: Hotspot geometry mode - "iso_polygons" (isodensity polygons) or "peaks" (density peaks) (default: "iso_polygons").
        iso_mass: Probability mass threshold for isodensity polygons (default: 0.90 = top 90%).
        grid_res_m: Grid cell size in meters for KDE evaluation (default: 250).
        min_area_m2: Minimum polygon area in m² for post-processing (default: 10000 = 0.01 km²).
        **kwargs: Additional arguments passed to Clusterer base class.
    """
    
    def __init__(
        self,
        bandwidth_meters: float = 1500.0,
        kernel: str = "gaussian",
        label_policy: str = "none",
        hotspot_mode: str = "iso_polygons",
        iso_mass: float = 0.90,
        grid_res_m: int = 250,
        min_area_m2: int = 10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bandwidth_meters = bandwidth_meters
        self.kernel = kernel
        self.label_policy = label_policy
        self.hotspot_mode = hotspot_mode
        self.iso_mass = iso_mass
        self.grid_res_m = grid_res_m
        self.min_area_m2 = min_area_m2
        self.method = "kde"
        self.model: Optional[KernelDensity] = None
        
        self.params.update({
            "bandwidth_meters": bandwidth_meters,
            "kernel": kernel,
            "label_policy": label_policy,
            "hotspot_mode": hotspot_mode,
            "iso_mass": iso_mass,
            "grid_res_m": grid_res_m,
            "min_area_m2": min_area_m2,
        })
    
    def fit(
        self,
        df: pd.DataFrame,
        x_col: str = "lon",
        y_col: str = "lat"
    ) -> "KDEClustering":
        """Fit KDE model to data.
        
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
        
        validate_crs_units(self.crs_proj, expected_units="m")
        
        gdf_proj, units = to_projected(df, x_col, y_col, self.crs_in, self.crs_proj)
        self.proj_units = units
        
        self.gdf_proj_ = gdf_proj.copy()
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        
        self.model = KernelDensity(
            bandwidth=self.bandwidth_meters,
            kernel=self.kernel
        )
        self.model.fit(X)
        self.n_samples = len(X)
        
        if self.label_policy == "none":
            self.labels_ = np.full(self.n_samples, -1, dtype=int)
        elif self.label_policy == "threshold":
            log_densities = self.model.score_samples(X)
            densities = np.exp(log_densities)
            threshold = np.percentile(densities, (1 - self.iso_mass) * 100)
            self.labels_ = np.where(densities >= threshold, 0, -1)
        else:
            raise ValueError(f"Unknown label_policy: {self.label_policy}")
        
        return self
    
    def _evaluate_kde_grid(
        self,
        bbox: Tuple[float, float, float, float],
        grid_res_m: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Evaluate KDE on a regular grid.
        
        Uses a pre-fitted model from fit() method and evaluates density on a
        resolution-based grid instead of fixed gridsize.
        
        Args:
            bbox: Bounding box as (minx, miny, maxx, maxy) in projected CRS.
            grid_res_m: Grid cell size in meters.
            
        Returns:
            Tuple containing:
            - X_grid: X coordinate grid array
            - Y_grid: Y coordinate grid array
            - Z: Density matrix (density per m²)
            - cell_area: Grid cell area in m²
        """
        minx, miny, maxx, maxy = bbox
        padding = self.bandwidth_meters * 2
        minx -= padding
        miny -= padding
        maxx += padding
        maxy += padding
        
        width = maxx - minx
        height = maxy - miny
        n_x = int(np.ceil(width / grid_res_m)) + 1
        n_y = int(np.ceil(height / grid_res_m)) + 1
        
        x_coords = np.linspace(minx, maxx, n_x)
        y_coords = np.linspace(miny, maxy, n_y)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        log_densities = self.model.score_samples(grid_points)
        densities = np.exp(log_densities)
        
        Z = densities.reshape(X_grid.shape)
        cell_area = grid_res_m * grid_res_m
        
        return X_grid, Y_grid, Z, cell_area
    
    def _find_isodensity_threshold(
        self,
        Z: np.ndarray,
        cell_area: float,
        iso_mass: float
    ) -> float:
        """Find isodensity threshold that captures specified probability mass.
        
        Args:
            Z: Density matrix (density per m²).
            cell_area: Area of each grid cell in m².
            iso_mass: Target probability mass (e.g., 0.90 for top 90%).
            
        Returns:
            Isodensity threshold value.
        """
        P = Z * cell_area
        P = P / P.sum()
        
        flat_Z = Z.flatten()
        flat_P = P.flatten()
        sorted_indices = np.argsort(flat_Z)[::-1]
        sorted_P = flat_P[sorted_indices]
        
        cumsum_P = np.cumsum(sorted_P)
        idx = np.searchsorted(cumsum_P, iso_mass)
        threshold = flat_Z[sorted_indices[idx]]
        
        return threshold
    
    def _polygonize_mask(
        self,
        mask: np.ndarray,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        grid_res_m: int
    ) -> list:
        """Polygonize binary mask into Shapely polygons.
        
        Args:
            mask: Binary mask (True/False or 1/0).
            X_grid: X coordinate grid.
            Y_grid: Y coordinate grid.
            grid_res_m: Grid cell size in meters.
            
        Returns:
            List of Shapely Polygon or MultiPolygon geometries.
            
        Note:
            Uses optional dependencies (rasterio, scikit-image, scipy) with fallbacks.
            Import errors are handled gracefully - install rasterio for best results.
        """
        try:
            import rasterio.features  # type: ignore[import-untyped, import]
            from rasterio.transform import from_bounds  # type: ignore[import-untyped, import]
            
            minx = X_grid[0, 0] - grid_res_m / 2
            maxx = X_grid[0, -1] + grid_res_m / 2
            miny = Y_grid[-1, 0] - grid_res_m / 2
            maxy = Y_grid[0, 0] + grid_res_m / 2
            
            transform = from_bounds(minx, miny, maxx, maxy, mask.shape[1], mask.shape[0])
            
            shapes = rasterio.features.shapes(
                mask.astype(np.uint8),
                transform=transform
            )
            polygons = []
            for shape, value in shapes:
                if value == 1:
                    coords = shape['coordinates'][0]
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception:
                        continue
            return polygons
        except ImportError:
            pass
        
        try:
            from skimage import measure  # type: ignore[import-untyped, import]
            contours = measure.find_contours(mask.astype(float), 0.5)
            polygons = []
            
            minx = X_grid[0, 0] - grid_res_m / 2
            miny = Y_grid[-1, 0] - grid_res_m / 2
            x_step = grid_res_m
            y_step = grid_res_m
            
            for contour in contours:
                coords = []
                for row, col in contour:
                    x = minx + col * x_step
                    y = miny + (mask.shape[0] - row) * y_step
                    coords.append((x, y))
                
                if len(coords) >= 3:
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception:
                        continue
            
            return polygons
        except ImportError:
            pass
        
        try:
            from scipy import ndimage  # type: ignore[import-untyped, import]
            labeled, num_features = ndimage.label(mask)
            polygons = []
            
            minx = X_grid[0, 0] - grid_res_m / 2
            miny = Y_grid[-1, 0] - grid_res_m / 2
            x_step = grid_res_m
            y_step = grid_res_m
            
            for i in range(1, num_features + 1):
                region_mask = labeled == i
                rows, cols = np.where(region_mask)
                if len(rows) > 0:
                    min_row, max_row = rows.min(), rows.max()
                    min_col, max_col = cols.min(), cols.max()
                    x_min = minx + min_col * x_step
                    y_min = miny + (mask.shape[0] - max_row - 1) * y_step
                    x_max = minx + (max_col + 1) * x_step
                    y_max = miny + (mask.shape[0] - min_row) * y_step
                    polygons.append(Polygon([
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max)
                    ]))
            return polygons
        except ImportError:
            raise ImportError(
                "Polygonization requires either rasterio or scikit-image. "
                "Please install one of them: pip install rasterio or pip install scikit-image"
            )
    
    def hotspots(
        self,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> gpd.GeoDataFrame:
        """Extract hotspot polygons or peaks.
        
        Args:
            top_n: Limit number of hotspots (optional).
            min_score: Filter by minimum score (optional).
            
        Returns:
            GeoDataFrame with hotspot polygons or points.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Run .fit() first.")
        
        if self.hotspot_mode == "peaks":
            X = np.column_stack([self.gdf_proj_.geometry.x, self.gdf_proj_.geometry.y])
            log_densities = self.model.score_samples(X)
            densities = np.exp(log_densities)
            
            top_indices = np.argsort(densities)[::-1]
            if top_n is not None:
                top_indices = top_indices[:top_n]
            elif min_score is not None:
                max_density = densities.max()
                threshold = min_score * max_density
                top_indices = top_indices[densities[top_indices] >= threshold]
            
            geometries = [self.gdf_proj_.geometry.iloc[i] for i in top_indices]
            scores = densities[top_indices] / densities.max()
            
            gdf = gpd.GeoDataFrame(
                {
                    "score": scores,
                    "method": self.method,
                },
                geometry=geometries,
                crs=self.crs_proj
            )
        else:
            bbox = self.gdf_proj_.total_bounds
            X_grid, Y_grid, Z, cell_area = self._evaluate_kde_grid(bbox, self.grid_res_m)
            threshold = self._find_isodensity_threshold(Z, cell_area, self.iso_mass)
            mask = Z >= threshold
            polygons = self._polygonize_mask(mask, X_grid, Y_grid, self.grid_res_m)
            
            if len(polygons) == 0:
                return gpd.GeoDataFrame(
                    columns=["score", "method", "params_hash", "params_json", "crs", "crs_proj", "proj_units"],
                    geometry=[],
                    crs=self.crs_in
                )
            
            processed_polygons = []
            for poly in polygons:
                if poly.area < self.min_area_m2:
                    continue
                processed_polygons.append(poly)
            
            if len(processed_polygons) > 1:
                merged = unary_union(processed_polygons)
                if isinstance(merged, Polygon):
                    processed_polygons = [merged]
                elif isinstance(merged, MultiPolygon):
                    processed_polygons = list(merged.geoms)
                else:
                    processed_polygons = [merged]
            
            scores = []
            total_mass = (Z * cell_area).sum()
            x_centers = X_grid[0, :]
            y_centers = Y_grid[:, 0]
            grid_points_flat = []
            for y in y_centers:
                for x in x_centers:
                    grid_points_flat.append(Point(x, y))
            
            for poly in processed_polygons:
                inside_mask = np.array([poly.contains(p) for p in grid_points_flat])
                inside_mask = inside_mask.reshape(Z.shape)
                
                if inside_mask.sum() == 0:
                    scores.append(0.0)
                    continue
                
                inside_Z = Z[inside_mask]
                inside_mass = (inside_Z * cell_area).sum()
                mass_fraction = inside_mass / total_mass if total_mass > 0 else 0.0
                scores.append(mass_fraction)
            
            gdf = gpd.GeoDataFrame(
                {
                    "score": scores,
                    "method": self.method,
                },
                geometry=processed_polygons,
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

