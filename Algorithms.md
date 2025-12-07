# Algorithms

This document provides a comprehensive overview of all algorithms used in the Guardian project for missing-child case analysis, synthetic data generation, and machine learning operations.

## Table of Contents

1. [Geographic & Distance Algorithms](#geographic--distance-algorithms)
2. [Graph & Network Algorithms](#graph--network-algorithms)
3. [Geospatial Analysis Algorithms](#geospatial-analysis-algorithms)
4. [Clustering Algorithms](#clustering-algorithms)
5. [Machine Learning Algorithms](#machine-learning-algorithms)
6. [Schema & Validation Algorithms](#schema--validation-algorithms)
7. [Reinforcement Learning Algorithms](#reinforcement-learning-algorithms)
8. [Mobility Forecasting Algorithms](#mobility-forecasting-algorithms)
9. [Clustering Evaluation Metrics](#clustering-evaluation-metrics)
10. [Optimization & Caching Algorithms](#optimization--caching-algorithms)
11. [Data Processing Algorithms](#data-processing-algorithms)
12. [Text Processing Algorithms](#text-processing-algorithms)

## Geographic & Distance Algorithms

### Haversine Distance

**Purpose**: Calculate the great-circle distance between two points on Earth using spherical geometry.

**Location**: `src/geography/distance.py`, `generate_cases.py`, `generate_cases_organized.py`

**Mathematical Formula**:
```
a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
c = 2 ⋅ atan2(√a, √(1−a))
d = R ⋅ c
```

Where:
- φ is latitude, λ is longitude
- R is Earth's radius (≈ 3959 miles)
- Δφ = φ2 - φ1, Δλ = λ2 - λ1

**Implementation**:
```python
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3959  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c
```

**Complexity**: O(1) time, O(1) space

**Usage**: Distance calculations, proximity searches, road finding, transit station discovery

**Performance**: ~0.000001 seconds per calculation

### Manhattan Distance

**Purpose**: Alternative distance metric for comparison with Haversine distance.

**Location**: `src/geography/distance.py`

**Formula**: |lat1 - lat2| + |lon1 - lon2|

**Usage**: Simplified distance calculations where spherical accuracy is not critical

## Graph & Network Algorithms

### Dijkstra's Algorithm (Bounded)

**Purpose**: Find shortest paths in transit networks with early termination for efficiency.

**Location**: `generate_cases.py`, `generate_cases_organized.py`

**Implementation**:
```python
def bounded_dijkstra(graph: Dict[int, List[Tuple[int, float]]], 
                     start_idx: int, 
                     cutoff: float) -> List[Tuple[int, float]]:
    dist = {start_idx: 0.0}
    pq = [(0.0, start_idx)]
    out = []
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > cutoff:
            break
        out.append((u, d))
        
        for v, weight in graph.get(u, []):
            new_dist = d + weight
            if new_dist <= cutoff and new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    return out
```

**Complexity**: O((V + E) log V) with early termination

**Key Features**:
- Priority queue-based implementation
- Early termination when distance exceeds cutoff
- Returns all nodes within specified distance
- Much faster than full Dijkstra's for bounded searches

**Usage**: Transit network pathfinding, nearby station discovery, road finding via transit connections

**Performance**: ~0.001-0.01 seconds for 2,359 stations

### Graph Construction

**Purpose**: Build transit network graphs from station coordinates.

**Location**: `generate_cases.py`, `src/transportation/networks.py`

**Algorithm**:
1. Extract station coordinates from transit data
2. Create connections between nearby stations (within max_connection_distance)
3. Build adjacency list representation
4. Cache graph for performance

**Complexity**: O(n²) where n is number of stations

**Implementation**:
```python
def build_transit_graph(transit_data: Dict[str, Any], max_connection_distance: float = 2.0):
    stations = []
    # Extract stations from transit data
    for feature in transit_data.get('features', []):
        if feature.get('geometry', {}).get('type') == 'Point':
            coords = feature['geometry']['coordinates']
            stations.append({'lat': coords[1], 'lon': coords[0]})
    
    # Build adjacency list
    graph = {}
    for i, station1 in enumerate(stations):
        graph[i] = []
        for j, station2 in enumerate(stations):
            if i != j:
                distance = haversine_distance(
                    station1['lat'], station1['lon'],
                    station2['lat'], station2['lon']
                )
                if distance <= max_connection_distance:
                    graph[i].append((j, distance))
    
    return graph, stations
```

**Usage**: Transit network analysis, station connectivity, pathfinding algorithms

## Geospatial Analysis Algorithms

### Point-in-Polygon

**Purpose**: Determine which Virginia region contains a given coordinate.

**Location**: `src/geography/regions.py`, `generate_cases.py`

**Implementation**: Bounding box check for efficiency
```python
def get_region_from_coordinates(lat: float, lon: float, regions_geojson: Dict[str, Any]) -> str:
    for feature in regions_geojson.get('features', []):
        if feature.get('geometry', {}).get('type') == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            
            # Simple bounding box check first
            min_lon = min(coord[0] for coord in coords)
            max_lon = max(coord[0] for coord in coords)
            min_lat = min(coord[1] for coord in coords)
            max_lat = max(coord[1] for coord in coords)
            
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                return feature.get('properties', {}).get('region_tag', 'Unknown')
    
    return 'Unknown'
```

**Complexity**: O(n) where n is number of regions

**Usage**: Regional classification, geographic validation, case location analysis

**Performance**: ~0.001 seconds for 5 regions

### Kernel Density Estimation (KDE)

**Purpose**: Create density heatmaps for spatial analysis of case distributions.

**Location**: `eda_hotspot.py`

**Implementation**: Gaussian kernels with scikit-learn
```python
def _kde_heat(xs: np.ndarray, ys: np.ndarray, bw: float, gridsize: int = 1000):
    # Create regular grid for KDE computation
    xgrid = np.linspace(x_min, x_max, gridsize)
    ygrid = np.linspace(y_min, y_max, gridsize)
    X, Y = np.meshgrid(xgrid, ygrid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    # Fit KDE model and compute density
    kde = KernelDensity(bandwidth=bw, kernel="gaussian")
    kde.fit(np.vstack([xs, ys]).T)
    Z = np.exp(kde.score_samples(grid_points)).reshape(gridsize, gridsize)
    return X, Y, Z
```

**Features**:
- Gaussian kernel density estimation
- Configurable bandwidth parameter
- Grid-based computation for visualization
- Log-transformed density values

**Usage**: Hotspot mapping, case distribution analysis, demographic visualization

**Parameters**:
- Bandwidth: 30,000 meters (approximately 18.64 miles) for state-scale analysis
- Grid size: 1000x1000 for high-resolution maps

### Raster Masking

**Purpose**: Clip density rasters to state boundaries for accurate visualization.

**Location**: `eda_hotspot.py`

**Implementation**: Shapely vectorized operations
```python
def _mask_raster_to_polygon(Z, X, Y, boundary_gdf):
    if boundary_gdf is None or boundary_gdf.empty:
        return Z
    
    b = boundary_gdf.to_crs(epsg=3857)
    poly = b.dissolve().geometry.unary_union.buffer(0)
    
    Zm = Z.copy()
    try:
        from shapely import vectorized
        mask = vectorized.covers(poly, X, Y)
    except Exception:
        # Fallback to point-by-point checking
        mask = np.zeros_like(Z, dtype=bool)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                mask[i, j] = poly.covers(Point(float(X[i, j]), float(Y[i, j])))
    
    Zm[~mask] = np.nan
    return Zm
```

**Usage**: Geographic boundary enforcement, clean map visualization

## Clustering Algorithms

### K-Means Clustering

**Purpose**: Partition-based clustering for hotspot detection from geographic coordinates.

**Location**: `clustering/kmeans.py`

**Implementation**:
```python
class KMeansClustering(Clusterer):
    def fit(self, df: pd.DataFrame, x_col: str = "lon", y_col: str = "lat"):
        # Validate and transform coordinates to projected CRS
        gdf_proj, units = to_projected(df, x_col, y_col, self.crs_in, self.crs_proj)
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        
        # Fit K-Means model
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def hotspots(self):
        # Extract cluster centers as hotspots
        centers = self.model.cluster_centers_
        cluster_sizes = np.bincount(self.labels_, minlength=len(centers))
        scores = cluster_sizes / len(self.labels_)
        return GeoDataFrame with hotspot centers
```

**Algorithm Steps**:
1. Transform geographic coordinates to projected CRS (meters) for accurate distance calculations
2. Fit K-Means model with specified number of clusters
3. Extract cluster centers as hotspot locations
4. Compute cluster scores based on point counts
5. Transform centers back to geographic CRS for output

**Complexity**: O(n × k × i × d) where n is number of points, k is clusters, i is iterations, d is dimensions

**Key Features**:
- Automatic coordinate transformation (geographic → projected → geographic)
- Cluster center extraction as point hotspots
- Score computation based on cluster size
- Support for prediction on new data points

**Parameters**:
- `n_clusters`: Number of clusters (default: 5)
- `random_state`: Random seed for reproducibility (default: 42)

**Usage**: Hotspot detection, spatial clustering, geographic pattern analysis

### DBSCAN Clustering

**Purpose**: Density-based clustering for irregular hotspot shapes with noise detection.

**Location**: `clustering/dbscan.py`

**Implementation**:
```python
class DBSCANClustering(Clusterer):
    def fit(self, df: pd.DataFrame, x_col: str = "lon", y_col: str = "lat"):
        # Transform to projected CRS (must be in meters for eps_meters)
        gdf_proj, units = to_projected(df, x_col, y_col, self.crs_in, self.crs_proj)
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        
        # Fit DBSCAN model
        self.model = DBSCAN(
            eps=self.eps_meters,
            min_samples=self.min_samples,
            metric="euclidean"
        )
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def hotspots(self):
        # Extract hotspots as centroids or convex hulls
        if self.hotspot_mode == "centroid":
            # Compute weighted centroid for each cluster
            coords = np.array([[p.x, p.y] for p in cluster_points.geometry])
            centroid = Point(coords.mean(axis=0))
        elif self.hotspot_mode == "hull":
            # Compute convex hull for each cluster
            points = MultiPoint(list(cluster_points.geometry))
            hull = points.convex_hull
        return GeoDataFrame with hotspot geometries
```

**Algorithm Steps**:
1. Transform coordinates to projected CRS (meters) for eps parameter
2. Fit DBSCAN model with eps (maximum distance) and min_samples parameters
3. Identify clusters and noise points (label = -1)
4. Extract hotspots as centroids or convex hulls based on mode
5. Compute cluster scores based on point counts

**Complexity**: O(n log n) with spatial indexing, O(n²) worst case

**Key Features**:
- Density-based clustering (no fixed number of clusters)
- Noise detection (points not in any cluster)
- Flexible hotspot extraction (centroids or convex hulls)
- Handles irregular cluster shapes

**Parameters**:
- `eps_meters`: Maximum distance in meters for points to be in same cluster (default: 1000.0)
- `min_samples`: Minimum number of points to form a cluster (default: 5)
- `hotspot_mode`: "centroid" (point-weighted) or "hull" (convex hull) (default: "centroid")

**Usage**: Density-based hotspot detection, irregular cluster shapes, noise filtering

### KDE Clustering (Isodensity Polygons)

**Purpose**: Kernel density estimation with isodensity polygon extraction for hotspot identification.

**Location**: `clustering/kde.py`

**Implementation**:
```python
class KDEClustering(Clusterer):
    def fit(self, df: pd.DataFrame, x_col: str = "lon", y_col: str = "lat"):
        # Transform to projected CRS
        gdf_proj, units = to_projected(df, x_col, y_col, self.crs_in, self.crs_proj)
        X = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
        
        # Fit KDE model
        self.model = KernelDensity(
            bandwidth=self.bandwidth_meters,
            kernel=self.kernel
        )
        self.model.fit(X)
        return self
    
    def hotspots(self):
        # Evaluate KDE on grid
        X_grid, Y_grid, Z, cell_area = self._evaluate_kde_grid(bbox, grid_res_m)
        
        # Find isodensity threshold for iso_mass
        threshold = self._find_isodensity_threshold(Z, cell_area, self.iso_mass)
        
        # Create binary mask
        mask = Z >= threshold
        
        # Polygonize mask into Shapely polygons
        polygons = self._polygonize_mask(mask, X_grid, Y_grid, grid_res_m)
        
        return GeoDataFrame with isodensity polygon hotspots
```

**Algorithm Steps**:
1. Transform coordinates to projected CRS (meters)
2. Fit KDE model with specified bandwidth and kernel
3. Evaluate density on regular grid (grid_res_m resolution)
4. Find isodensity threshold that captures iso_mass probability (e.g., top 90%)
5. Create binary mask from threshold
6. Polygonize mask into Shapely polygons
7. Filter polygons by minimum area

**Complexity**: O(n × m) where n is number of points, m is grid cells

**Key Features**:
- Grid-based density evaluation
- Isodensity polygon extraction (captures specified probability mass)
- Configurable bandwidth and grid resolution
- Polygon filtering by minimum area

**Parameters**:
- `bandwidth_meters`: KDE bandwidth in meters (default: 1500.0)
- `iso_mass`: Probability mass threshold for isodensity polygons (default: 0.90 = top 90%)
- `grid_res_m`: Grid cell size in meters (default: 250)
- `min_area_m2`: Minimum polygon area in m² (default: 10000 = 0.01 km²)
- `hotspot_mode`: "iso_polygons" (isodensity polygons) or "peaks" (density peaks)

**Usage**: Density-based hotspot detection, isodensity polygon extraction, spatial density analysis

**Note**: This is different from the visualization KDE in `eda_hotspot.py`. The clustering version extracts isodensity polygons for hotspot identification, while the visualization version creates heatmaps for display.

## Machine Learning Algorithms

### 4-bit Quantization (NF4)

**Purpose**: Memory-efficient model loading for 8GB VRAM systems.

**Location**: `guardian_llm/summarizer.py`, `guardian_llm/extractor.py`, `guardian_llm/weak_labeler.py`

**Implementation**:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id_or_dir,
    torch_dtype="auto",
    device_map=device_map,
    quantization_config=bnb_config,
    trust_remote_code=True
)
```

**Features**:
- 4-bit quantization using NF4 format
- Double quantization for additional compression
- bfloat16 compute dtype for efficiency
- Reduces model memory by ~75%

**Usage**: Memory-efficient model loading, RTX 4060 optimization

### QLoRA (Quantized Low-Rank Adaptation)

**Purpose**: Parameter-efficient fine-tuning with minimal memory requirements.

**Location**: `guardian_llm/finetune_qlora.py`

**Implementation**:
```python
class GuardianFineTuner:
    def setup_lora(self, r=16, lora_alpha=32, lora_dropout=0.1):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"]
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
```

**Parameters**:
- Rank (r): 16 - Low-rank adaptation dimension
- Alpha: 32 - LoRA scaling parameter
- Dropout: 0.1 - Regularization rate
- Target modules: All linear layers in transformer

**Features**:
- 4-bit quantization + LoRA adapters
- Enables fine-tuning on 8GB VRAM
- Minimal parameter updates (~1% of model parameters)
- Maintains performance while reducing memory

**Usage**: Model fine-tuning, task-specific adaptation

### Transformer Inference

**Purpose**: Efficient LLM text generation with hardware optimization.

**Features**:
- **SDPA (Scaled Dot-Product Attention)**: Fast attention implementation
- **Flash Attention 2**: Memory-efficient attention computation
- **TF32 Acceleration**: RTX 40xx GPU optimization
- **KV Caching**: Reuse computed key-value pairs
- **Early Stopping**: Terminate at specific markers

**Usage**: Text generation, entity extraction, summarization, classification

## Schema & Validation Algorithms

### JSON Schema Validation

**Purpose**: Automated validation of all JSON files against appropriate schemas.

**Location**: `build.py`

**Implementation**:
```python
def validate_json_file(file_path, schemas):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    
    schema_type = detect_schema_type(file_path, json_data)
    if not schema_type or schema_type not in schemas:
        return False, f"Schema '{schema_type}' not found"
    
    try:
        validate(instance=json_data, schema=schemas[schema_type])
        return True, None
    except ValidationError as e:
        return False, f"Schema error: {e.message}"
```

**Features**:
- Intelligent schema detection
- Comprehensive error reporting
- Support for multiple schema types
- Automated validation workflow

### Schema Detection Algorithm

**Purpose**: Automatically determine appropriate schema for JSON files.

**Algorithm**:
1. **Path-based detection**: Check file path patterns
   - `synthetic_cases/` → guardian schema
   - `templates/` → templates schema
   - `geo/` with gazetteer → gazetteer schema

2. **Content-based detection**: Analyze JSON structure
   - `case_id` + `demographic` → guardian schema
   - `synthetic_case_templates` → templates schema
   - `version` + `crs` + `entries` → gazetteer schema

3. **Field analysis**: Check for specific field patterns
   - `segmentId` + `localNames` → road_segment schema
   - `id` + `type` + `geometry` → transit schema

**Complexity**: O(1) for path detection, O(n) for content analysis

**Usage**: Build system validation, data quality assurance

## Reinforcement Learning Algorithms

### Priority Reweighting

**Purpose**: LLM-enhanced zone prioritization using weighted scoring.

**Location**: `zone_qa.py`

**Formula**:
```
score = α*original_priority + β*plausibility - γ*radius + risk_boost
priority = 1/(1 + exp(-3*(score - 0.5)))  # Sigmoid normalization
```

**Implementation**:
```python
def recompute_priority(zone: Dict[str, Any], qa_result: Dict[str, Any], 
                      reward_config: Dict[str, Any]) -> float:
    # Extract weights from active profile
    prof = reward_config.get("profiles", {}).get(reward_config.get("__active_profile__", "baseline"), {})
    w = prof.get("weights", {})
    
    alpha = float(w.get("alpha_orig", 0.6))      # Weight for original priority
    beta = float(w.get("beta_plaus", 0.8))       # Weight for LLM plausibility
    gamma = float(w.get("gamma_radius", 0.02))    # Penalty weight for zone radius
    
    # Extract input values
    orig = float(zone.get("priority", 0.5))
    plaus = float(qa_result.get("plausibility", 0.5))
    radius = float(zone.get("radius_miles", 3.11))
    
    # Calculate weighted score
    score = alpha * orig + beta * plaus - gamma * radius
    
    # Sigmoid normalization to [0, 1]
    return 1.0/(1.0 + math.exp(-3*(score-0.5)))
```

**Parameters**:
- α (alpha_orig): 0.6 - Weight for original priority
- β (beta_plaus): 0.8 - Weight for LLM plausibility
- γ (gamma_radius): 0.02 - Penalty for large radius

**Usage**: Zone prioritization, search effectiveness optimization

### Reward Calculation

**Purpose**: Multi-factor reward computation for search zone evaluation.

**Location**: `reinforcement_learning/search_reward_config.json`

**Distance-based Reward**:
```
reward = 1 / (1 + max(0, d_center_true_miles - radius_miles))
```

**Time-based Reward**:
```
reward = max(0, (window_end_hr - t_hit_hr) / (window_end_hr - window_start_hr))
```

**Hybrid Reward**:
```
reward = α * distance_component + β * time_factor + inside_bonus + corridor_bonus
```

**Parameters**:
- α: 0.7 - Distance component weight
- β: 0.3 - Time factor weight
- inside_threshold: 0.85 - Inside zone threshold
- inside_bonus: 0.1 - Bonus for inside zones
- corridor_bonus: 0.05 - Bonus for corridor zones

### Geo-hit@K Evaluation

**Purpose**: Search effectiveness metrics at different K values.

**Algorithm**:
1. For each case, rank zones by priority
2. Check if ground truth location falls within top-K zones
3. Calculate hit rate across all cases
4. Report metrics for K=1, 3, 5, 10

**Usage**: Performance evaluation, baseline vs LLM-enhanced comparison

## Mobility Forecasting Algorithms

### Markov Chain Propagation

**Purpose**: Probabilistic movement model for forecasting probability distributions over geographic space at future time horizons.

**Location**: `reinforcement_learning/movement_model.py`, `reinforcement_learning/forecast_api.py`

**Mathematical Formulation**:
```
p^(t+1) = P^T × p^(t)
```

Where:
- `p^(t)` is probability distribution at time step t (shape: (N,))
- `P` is transition matrix (shape: (N, N)) where P[i,j] = probability of moving from cell i to cell j
- `P^T` is transpose of transition matrix (row-stochastic: each row sums to 1)

**Implementation**:
```python
def risk_map(grid_xy, hotspots, road_cost, seclusion, t_hours, steps=3, init=None, alpha_prior=0.5):
    # Build KDE prior from hotspots
    prior = kde_prior(grid_xy, hotspots)
    
    # Build transition matrix
    P = build_transition(grid_xy, road_cost, seclusion)
    
    # Mix prior and case-specific seed
    if init is None:
        p = prior
    else:
        p = alpha_prior * prior + (1.0 - alpha_prior) * init
        p = p / (p.sum() + 1e-9)
    
    # Propagate through Markov chain
    for _ in range(steps):
        p = P.T @ p
    
    # Apply survival decay
    r = p * survival_factor(t_hours, profile="default")
    
    return r / (r.sum() + 1e-9)
```

**Algorithm Steps**:
1. Build initial probability distribution from KDE prior and/or case-specific seed
2. Build transition matrix P based on road costs, seclusion, and corridor proximity
3. Propagate distribution through Markov chain for specified number of steps
4. Apply survival decay based on elapsed time
5. Normalize to ensure probabilities sum to 1

**Sequential Propagation**:
For multiple horizons (e.g., 24h, 48h, 72h), forecasts build sequentially:
- 24h forecast: propagate from t=0 to t=24h
- 48h forecast: propagate from t=24h to t=48h (builds on 24h result)
- 72h forecast: propagate from t=48h to t=72h (builds on 48h result)

**Day/Night Transitions**:
- Different transition matrices for day vs night periods
- Day: 6 AM - 8 PM (higher corridor bias)
- Night: 8 PM - 6 AM (lower corridor bias, higher seclusion preference)

**Complexity**: O(N² × steps) where N is number of grid cells, steps is propagation steps

**Key Features**:
- Sequential propagation (forecasts build on previous horizons)
- Day/night transition modeling
- Profile-based survival decay
- Boundary enforcement (Virginia state mask)

**Parameters**:
- `steps`: Number of Markov chain propagation steps (default: 3 per 24 hours)
- `alpha_prior`: Mixing weight for KDE prior vs case seed (0.0 = only seed, 1.0 = only prior)
- `profile`: Survival profile type ("default", "runaway", "abduction")

**Usage**: Mobility forecasting, risk distribution prediction, search zone optimization

### Survival Analysis (Exponential Decay)

**Purpose**: Temporal decay models for modeling survival probability over time since disappearance.

**Location**: `reinforcement_learning/movement_model.py`

**Mathematical Formula**:
```
S(t) = exp(-λ × t)
```

Where:
- `S(t)` is survival probability at time t
- `λ = ln(2) / half_life` is decay rate
- `half_life` is time at which survival probability = 0.5

**Implementation**:
```python
def survival_curve_exponential(t_hours: float, half_life: float = 24.0) -> float:
    lam = np.log(2.0) / max(half_life, 1e-6)
    return np.exp(-lam * t_hours)

def survival_factor(t_hours: float, profile: str = "default") -> float:
    if profile == "default":
        return survival_curve_exponential(t_hours, half_life=24.0)
    elif profile == "runaway":
        return survival_curve_exponential(t_hours, half_life=48.0)
    elif profile == "abduction":
        return survival_curve_exponential(t_hours, half_life=12.0)
```

**Profile Types**:
- **default**: Standard case (half_life = 24 hours)
- **runaway**: Runaway case (half_life = 48 hours) - slower decay
- **abduction**: Abduction case (half_life = 12 hours) - faster decay

**Complexity**: O(1) time, O(1) space

**Key Features**:
- Exponential decay model
- Profile-based half-life parameters
- Scales probability distributions by survival factor

**Usage**: Temporal decay in mobility forecasting, risk distribution scaling

### Transition Matrix Construction

**Purpose**: Build probability transition matrices for Markov chain movement models based on geographic features.

**Location**: `reinforcement_learning/movement_model.py`

**Mathematical Formulation**:
```
P[i,j] = exp(-β_cost × rc[j] + β_secl × sc[j] + β_corr × cc[j]) / Z
```

Where:
- `rc[j]` is normalized road cost at cell j (lower = easier movement)
- `sc[j]` is normalized seclusion score at cell j (higher = better hiding)
- `cc[j]` is normalized corridor score at cell j (higher = closer to highways)
- `β_cost`, `β_secl`, `β_corr` are weighting coefficients
- `Z` is normalization constant (ensures row sums to 1)

**Implementation**:
```python
def build_transition(grid_xy, road_cost, seclusion, k=12, beta_cost=1.0, beta_secl=0.5, 
                     corridor_score=None, beta_corr=0.0):
    # Find k nearest neighbors for each grid cell
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(grid_xy)
    idx = nbrs.kneighbors(return_distance=False)
    
    # Normalize features
    rc = (road_cost - np.min(road_cost)) / (np.ptp(road_cost) + 1e-9)
    sc = (seclusion - np.min(seclusion)) / (np.ptp(seclusion) + 1e-9)
    if corridor_score is not None:
        cc = (corridor_score - np.min(corridor_score)) / (np.ptp(corridor_score) + 1e-9)
    else:
        cc = np.zeros_like(road_cost)
    
    # Build transition matrix
    N = len(grid_xy)
    P = np.zeros((N, N), float)
    
    for i in range(N):
        js = idx[i]  # k nearest neighbors
        # Compute transition weights
        w = np.exp(-beta_cost * rc[js] + beta_secl * sc[js] + beta_corr * cc[js])
        P[i, js] = w
        P[i, i] += 1e-6  # Small self-transition probability
        P[i] /= (P[i].sum() + 1e-12)  # Normalize row
    
    return P
```

**Algorithm Steps**:
1. Find k nearest neighbors for each grid cell using KD-tree
2. Normalize road cost, seclusion, and corridor scores to [0, 1]
3. Compute transition weights using exponential function
4. Normalize each row to ensure probabilities sum to 1
5. Validate transition matrix (row sums, non-negativity)
6. Cache result to avoid recomputation

**Complexity**: O(N × k × log N) for neighbor search, O(N × k) for matrix construction

**Key Features**:
- K-nearest neighbors for sparse transitions (only k neighbors per cell)
- Road cost weighting (prefer low-cost paths)
- Seclusion scoring (prefer hidden locations)
- Corridor bias (prefer highway proximity)
- Matrix caching for performance

**Parameters**:
- `k`: Number of nearest neighbors (default: 12)
- `beta_cost`: Road cost penalty coefficient (default: 1.0)
- `beta_secl`: Seclusion reward coefficient (default: 0.5)
- `beta_corr`: Corridor reward coefficient (default: 0.0, optional)

**Usage**: Movement model construction, transition probability computation, Markov chain setup

## Clustering Evaluation Metrics

### Silhouette Score

**Purpose**: Measure cluster quality by computing how similar points are to their own cluster compared to other clusters.

**Location**: `metrics/clustering.py`

**Mathematical Formula**:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- `a(i)` is average distance from point i to other points in same cluster
- `b(i)` is minimum average distance from point i to points in other clusters
- Score ranges from -1 (poor clustering) to +1 (excellent clustering)

**Implementation**:
```python
def silhouette_score(labels: np.ndarray, X: np.ndarray) -> float:
    # Filter out noise points (label == -1)
    mask = labels >= 0
    labels_filtered = labels[mask]
    X_filtered = X[mask]
    
    # Need at least 2 clusters
    unique_labels = np.unique(labels_filtered)
    if len(unique_labels) < 2:
        return -1.0
    
    return sklearn_silhouette_score(X_filtered, labels_filtered)
```

**Complexity**: O(n² × d) where n is number of points, d is dimensions

**Usage**: Cluster quality assessment, parameter selection, algorithm comparison

### Davies-Bouldin Index

**Purpose**: Measure cluster quality by computing average similarity ratio between clusters.

**Location**: `metrics/clustering.py`

**Mathematical Formula**:
```
DB = (1/k) × Σ max(i≠j) [(σ_i + σ_j) / d(c_i, c_j)]
```

Where:
- `k` is number of clusters
- `σ_i` is average distance from points in cluster i to cluster center
- `d(c_i, c_j)` is distance between cluster centers i and j
- Lower values indicate better clustering

**Implementation**:
```python
def davies_bouldin_score(labels: np.ndarray, X: np.ndarray) -> float:
    # Filter out noise points (label == -1)
    mask = labels >= 0
    labels_filtered = labels[mask]
    X_filtered = X[mask]
    
    # Need at least 2 clusters
    unique_labels = np.unique(labels_filtered)
    if len(unique_labels) < 2:
        return np.inf
    
    return sklearn_davies_bouldin_score(X_filtered, labels_filtered)
```

**Complexity**: O(n × k + k² × d) where n is points, k is clusters, d is dimensions

**Usage**: Cluster quality assessment, parameter selection, algorithm comparison

### Bootstrap Stability Analysis

**Purpose**: Evaluate cluster stability across multiple bootstrap iterations by resampling data.

**Location**: `metrics/clustering.py`

**Algorithm**:
1. For each bootstrap iteration:
   - Sample fraction of data points (default: 85%)
   - Fit clustering model on sampled data
   - Extract hotspots and cluster labels
2. Compute stability metrics:
   - **Jaccard Overlap**: Overlap of hotspot polygons across iterations
   - **Adjusted Rand Index (ARI)**: Agreement of cluster labels across iterations
   - **Mean/Std ARI**: Average and standard deviation of ARI scores

**Implementation**:
```python
def bootstrap_stability(clusterer, df, n_iter=10, sample_ratio=0.85):
    n_samples = len(df)
    sample_size = int(n_samples * sample_ratio)
    
    hotspot_gdfs = []
    label_arrays = []
    
    for i in range(n_iter):
        # Sample data
        sampled_indices = np.random.choice(n_samples, size=sample_size, replace=False)
        df_sample = df.iloc[sampled_indices]
        
        # Fit clusterer on sampled data
        clusterer_copy = copy.deepcopy(clusterer)
        clusterer_copy.fit(df_sample)
        
        # Get hotspots and labels
        hotspots = clusterer_copy.hotspots()
        labels = clusterer_copy.labels()
        
        hotspot_gdfs.append(hotspots)
        label_arrays.append(labels)
    
    # Compute Jaccard overlap of hotspot polygons
    jaccard_overlaps = []
    for i in range(n_iter):
        for j in range(i + 1, n_iter):
            union1 = hotspot_gdfs[i].geometry.unary_union
            union2 = hotspot_gdfs[j].geometry.unary_union
            intersection = union1.intersection(union2)
            union = union1.union(union2)
            jaccard = intersection.area / union.area if union.area > 0 else 0.0
            jaccard_overlaps.append(jaccard)
    
    # Compute ARI scores
    ari_scores = []
    for i in range(n_iter):
        for j in range(i + 1, n_iter):
            # Compute ARI on common points
            common_indices = np.intersect1d(sampled_indices_list[i], sampled_indices_list[j])
            labels_i = label_arrays[i][common_indices]
            labels_j = label_arrays[j][common_indices]
            ari = adjusted_rand_score(labels_i, labels_j)
            ari_scores.append(ari)
    
    return {
        "jaccard_overlap": np.mean(jaccard_overlaps),
        "ari_scores": ari_scores,
        "mean_ari": np.mean(ari_scores),
        "std_ari": np.std(ari_scores)
    }
```

**Complexity**: O(n_iter × (n × clustering_time + n² × comparison_time))

**Key Features**:
- Resampling-based stability evaluation
- Jaccard overlap of hotspot polygons
- Adjusted Rand Index for label agreement
- Statistical summary (mean, std) across iterations

**Parameters**:
- `n_iter`: Number of bootstrap iterations (default: 10)
- `sample_ratio`: Fraction of points to sample (default: 0.85)

**Usage**: Cluster stability assessment, parameter robustness evaluation, algorithm reliability testing

## Optimization & Caching Algorithms

### Graph Caching

**Purpose**: Persistent caching of transit network graphs to avoid repeated construction.

**Location**: `generate_cases.py`, `src/transportation/networks.py`

**Implementation**:
```python
# Global caches for performance optimization
_GRAPH_CACHE = None
_STATIONS_CACHE = None

def get_cached_graph_and_stations(transit_data):
    global _GRAPH_CACHE, _STATIONS_CACHE
    
    if _GRAPH_CACHE is None or _STATIONS_CACHE is None:
        _GRAPH_CACHE, _STATIONS_CACHE = build_transit_graph(transit_data)
    
    return _GRAPH_CACHE, _STATIONS_CACHE
```

**Features**:
- Global cache variables
- Lazy initialization
- Memory-efficient storage
- Performance optimization

**Usage**: Transit network analysis, road finding algorithms

### Lazy Loading

**Purpose**: On-demand data loading to minimize memory usage.

**Implementation**:
- Models loaded only when first needed
- Gazetteer data loaded on demand
- Transit networks cached after first use
- Configuration files loaded as needed

**Usage**: Memory optimization, startup performance

### Memory Management

**Purpose**: Automatic garbage collection during large-scale operations.

**Implementation**:
```python
# Automatic garbage collection every 50 cases
if case_count % 50 == 0:
    import gc
    gc.collect()
```

**Features**:
- Periodic garbage collection
- Memory monitoring
- Performance tracking
- Large-scale generation support

**Usage**: 500+ case generation, memory optimization

## Data Processing Algorithms

### Entity Extraction

**Purpose**: Hybrid rule-based + LLM entity extraction for structured output.

**Algorithm**:
1. **Rule-based extraction**: Use regex patterns for common entities
2. **LLM backfill**: Fill missing fields with LLM inference
3. **Validation**: Ensure JSON structure compliance
4. **Completion**: Fill gaps with intelligent defaults

**Features**:
- Deterministic scaffolding
- LLM enhancement
- Regex validation
- Structured JSON output

**Usage**: Case data extraction, entity recognition

### Risk Assessment

**Purpose**: Rule-based + LLM hybrid risk classification.

**Location**: `guardian_llm/weak_labeler.py`

**Algorithm**:
1. **Rule-based heuristics**: Check for high-risk keywords and patterns
2. **LLM classification**: Use model for nuanced assessment
3. **Hybrid scoring**: Combine rule-based and LLM scores
4. **Calibration**: Adjust scores based on historical data

**Implementation**:
```python
def assess_risk(narrative: str) -> str:
    # Rule-based risk assessment
    rule_risk = _risk_by_rules(narrative)
    
    # LLM-based assessment
    llm_risk = _llm_risk_assessment(narrative)
    
    # Hybrid scoring
    if rule_risk == "high":
        return "high"
    elif llm_risk == "high" and rule_risk != "low":
        return "high"
    else:
        return llm_risk
```

**Usage**: Case prioritization, risk-based zone weighting

### Batch Processing

**Purpose**: Efficient multi-case processing with memory optimization.

**Features**:
- Configurable batch sizes
- Memory management
- Progress tracking
- Error handling

**Usage**: Large-scale case processing, LLM inference optimization

## Text Processing Algorithms

### Token Decoding

**Purpose**: Clean LLM output extraction without prompt echo.

**Features**:
- Prompt removal
- Early stopping at markers
- Clean token extraction
- Output formatting

**Usage**: LLM output processing, text generation

### Regex Backfill

**Purpose**: Pattern-based field completion for missing JSON fields.

**Algorithm**:
1. Identify missing fields in JSON structure
2. Apply regex patterns to extract values
3. Fill missing fields with extracted data
4. Validate completed structure

**Implementation**:
```python
def _fallback_parse(text: str) -> dict:
    # Extract movement pattern using regex
    movement_match = re.search(r'(walking|driving|public_transit|unknown)', text, re.I)
    movement = movement_match.group(1).lower() if movement_match else "unknown"
    
    # Extract risk level using regex
    risk_match = re.search(r'(low|medium|high)', text, re.I)
    risk = risk_match.group(1).lower() if risk_match else "medium"
    
    return {"movement": movement, "risk": risk}
```

**Usage**: JSON completion, data validation, error recovery

## Algorithm Integration

The Guardian project integrates these algorithms through:

- **Geographic Processing**: Haversine distance + point-in-polygon for location analysis
- **Network Analysis**: Dijkstra's algorithm + graph construction for transit networks
- **Clustering**: K-Means, DBSCAN, and KDE for multi-method hotspot detection
- **Mobility Forecasting**: Markov chain propagation + survival analysis + transition matrices for risk distribution prediction
- **Machine Learning**: Quantization + QLoRA for efficient model operations
- **Data Validation**: Schema detection + JSON validation for data quality
- **Reinforcement Learning**: Priority reweighting + reward calculation for optimization
- **Evaluation**: Silhouette score + Davies-Bouldin index + bootstrap stability for cluster quality assessment
- **Performance**: Caching + lazy loading + memory management for scalability

This comprehensive algorithm ecosystem enables efficient processing of missing-child case data, synthetic generation, machine learning operations, and mobility forecasting while maintaining high performance and accuracy.
