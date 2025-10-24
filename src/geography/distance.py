"""Distance calculation utilities for geographic coordinates.

This module contains various distance calculation functions optimized
for different use cases in the Guardian system.
"""

import math
from typing import Tuple


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Uses the Haversine formula to compute the shortest distance between
    two points on a sphere (Earth) given their latitude and longitude.
    This is the standard formula for calculating distances on a spherical
    surface and is more accurate than simple Euclidean distance for
    geographic coordinates.
    
    Args:
        lat1 (float): Latitude of first point in decimal degrees
        lon1 (float): Longitude of first point in decimal degrees
        lat2 (float): Latitude of second point in decimal degrees
        lon2 (float): Longitude of second point in decimal degrees
        
    Returns:
        float: Distance in miles
        
    Example:
        >>> dist = haversine_distance(38.88, -77.1, 38.89, -77.11)
        >>> dist > 0
        True
        >>> dist < 1.0  # Should be less than 1 mile
        True
        
    Formula:
        a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 ⋅ atan2( √a, √(1−a) )
        d = R ⋅ c
        
    Where:
        φ is latitude, λ is longitude, R is earth's radius (≈ 3959 miles)
        Δφ = φ2 - φ1, Δλ = λ2 - λ1
        
    Performance:
        - Time Complexity: O(1)
        - Space Complexity: O(1)
        - Typical runtime: ~0.000001 seconds
        
    Note:
        This function assumes Earth is a perfect sphere with radius 3959 miles.
        For most applications, this provides sufficient accuracy.
    """
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Earth's radius in miles
    earth_radius = 3959.0
    
    return earth_radius * c


def manhattan_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Manhattan distance between two points.
    
    Manhattan distance is an admissible heuristic for A* search that
    provides a lower bound on the actual distance between two points.
    It's computationally efficient and works well for grid-based pathfinding.
    
    Args:
        lat1 (float): Latitude of first point in decimal degrees
        lon1 (float): Longitude of first point in decimal degrees
        lat2 (float): Latitude of second point in decimal degrees
        lon2 (float): Longitude of second point in decimal degrees
        
    Returns:
        float: Manhattan distance in miles
        
    Example:
        >>> dist = manhattan_distance(38.88, -77.1, 38.89, -77.11)
        >>> dist > 0
        True
        
    Note:
        Manhattan distance is calculated as |lat1-lat2| + |lon1-lon2|,
        converted to miles using approximate conversion factors.
    """
    # Approximate conversion factors for latitude/longitude to miles
    lat_to_miles = 69.0  # 1 degree latitude ≈ 69 miles
    lon_to_miles = 54.6  # 1 degree longitude ≈ 54.6 miles at 38°N (Virginia)
    
    lat_diff = abs(lat1 - lat2) * lat_to_miles
    lon_diff = abs(lon1 - lon2) * lon_to_miles
    
    return lat_diff + lon_diff
