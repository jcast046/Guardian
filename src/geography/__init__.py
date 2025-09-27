"""
Geography module for Guardian case generation.

This module contains geographic utilities for distance calculations,
regional classification, and spatial analysis.
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
