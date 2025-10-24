"""Geography module for Guardian case generation.

This module contains geographic utilities for distance calculations,
regional classification, and spatial analysis for the Guardian missing
person case management system.

Modules:
    distance: Distance calculation functions (Haversine, Manhattan)
    regions: Regional classification and nearby place finding
    validation: Geographic accuracy validation for roads and locations

Functions:
    haversine_distance: Calculate great circle distance between points
    manhattan_distance: Calculate Manhattan distance between points
    get_region_from_coordinates: Determine Virginia region from coordinates
    find_nearby_places: Find nearby cities and landmarks
    is_geographically_accurate_road: Validate road geographic accuracy
    is_major_road_in_region: Validate major road regional appropriateness
"""

from .distance import (
    haversine_distance,
    manhattan_distance
)

from .regions import (
    get_region_from_coordinates,
    find_nearby_places
)

from .validation import (
    is_geographically_accurate_road,
    is_major_road_in_region
)

__all__ = [
    'haversine_distance',
    'manhattan_distance',
    'get_region_from_coordinates',
    'find_nearby_places',
    'is_geographically_accurate_road',
    'is_major_road_in_region'
]
