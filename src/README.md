# Guardian Source Code Organization

This directory contains the organized source code for the Guardian synthetic case generation system, separated into logical modules for better maintainability and code organization.

## Directory Structure

```
src/
├── algorithms/          # Pathfinding and optimization algorithms
│   ├── __init__.py
│   ├── dijkstra.py     # Dijkstra's algorithm implementations
│   ├── astar.py        # A* search with Manhattan distance
│   ├── hub_labeling.py # Hub Labeling for ultra-fast queries
│   └── bounded_search.py # Bounded search algorithms
├── geography/          # Geographic utilities and validation
│   ├── __init__.py
│   ├── distance.py     # Distance calculation functions
│   ├── regions.py      # Regional classification
│   └── validation.py   # Geographic validation utilities
├── transportation/     # Transportation network utilities
│   ├── __init__.py
│   ├── networks.py     # Network construction and caching
│   ├── roads.py        # Road finding and analysis
│   └── transit.py      # Transit hub finding
└── validation/         # Data validation utilities
    ├── __init__.py
    └── schema.py       # Schema validation functions
```

## Module Descriptions

### Algorithms Module (`src/algorithms/`)

Contains all pathfinding and optimization algorithms:

- **`dijkstra.py`**: Dijkstra's algorithm implementations including bounded Dijkstra's for range queries
- **`astar.py`**: A* search algorithm with Manhattan distance heuristic for informed search
- **`hub_labeling.py`**: Hub Labeling (Pruned Landmark Labeling) for ultra-fast distance queries
- **`bounded_search.py`**: Bounded search algorithms optimized for "all nodes within D miles" queries

### Geography Module (`src/geography/`)

Contains geographic utilities and validation:

- **`distance.py`**: Haversine distance and Manhattan distance calculations
- **`regions.py`**: Regional classification using GeoJSON boundaries
- **`validation.py`**: Geographic validation for road accuracy

### Transportation Module (`src/transportation/`)

Contains transportation network utilities:

- **`networks.py`**: Network construction, caching, and management
- **`roads.py`**: Road finding using comprehensive multi-source analysis
- **`transit.py`**: Transit hub finding using network pathfinding

## Key Features

### Algorithm Optimization
- **Dijkstra's Algorithm**: O((V + E) log V) with early termination for range queries
- **A* Search**: Informed search with Manhattan distance heuristic
- **Hub Labeling**: O(k) query time after O(k * (V + E) log V) preprocessing
- **Bounded Search**: Early termination for optimal range query performance

### Geographic Accuracy
- **Multi-source validation**: Uses gazetteer, regions, roads, transit, and summary data
- **Regional classification**: Accurate region determination using GeoJSON boundaries
- **Distance calculations**: Haversine formula for great-circle distances
- **Road validation**: Geographic context validation for road accuracy

### Performance Optimization
- **Global caching**: One-time graph building with O(1) subsequent access
- **Early termination**: Bounded algorithms for range queries
- **Memory efficiency**: Optimized data structures for large networks
- **Typical performance**: ~0.001-0.01 seconds for 2,359 stations

## Usage

The organized modules can be imported and used as follows:

```python
# Import specific algorithms
from algorithms import bounded_dijkstra, astar_shortest_paths

# Import geographic utilities
from geography import haversine_distance, get_region_from_coordinates

# Import transportation functions
from transportation import find_nearby_roads, find_nearby_transit
```

## Benefits of Organization

1. **Maintainability**: Clear separation of concerns makes code easier to maintain
2. **Reusability**: Modular design allows components to be reused in different contexts
3. **Testability**: Individual modules can be tested independently
4. **Scalability**: Easy to add new algorithms or geographic utilities
5. **Documentation**: Each module has comprehensive documentation
6. **Performance**: Optimized algorithms with caching for production use

## Migration from Monolithic Structure

The original `generate_cases.py` (2,085 lines) has been reorganized into:

- **Algorithms**: 4 focused modules for different pathfinding approaches
- **Geography**: 3 modules for geographic utilities and validation
- **Transportation**: 3 modules for transportation network analysis
- **Main script**: Simplified `generate_cases_organized.py` (200 lines)

This reduces complexity and improves code organization while maintaining all functionality.
