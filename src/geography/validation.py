"""Geographic validation utilities for road accuracy.

This module contains functions for validating road accuracy based on
geographic context and regional expectations.

Author: Joshua Castillo
"""

from typing import Dict, List, Any


def is_geographically_accurate_road(road_name: str, region: str, nearby_places: List[Dict[str, Any]], lat: float, lon: float) -> bool:
    """Determine if a road is geographically accurate based on location context.
    
    This function uses nearby places and regional context to validate
    whether a road should be included in the results.
    
    Args:
        road_name: Name of the road to validate
        region: Regional classification of the road
        nearby_places: Nearby places from gazetteer
        lat: Latitude of the location
        lon: Longitude of the location
        
    Returns:
        True if the road is geographically accurate
        
    Example:
        >>> places = [{'name': 'Alexandria', 'distance': 2.0}]
        >>> is_accurate = is_geographically_accurate_road('I-395', 'Northern Virginia', places, 38.8, -77.0)
        >>> is_accurate
        True
    """
    road_lower = road_name.lower()
    
    # Filter out obviously incorrect roads
    if is_obviously_incorrect_road(road_name, region):
        return False
    
    # Check if road name matches nearby places
    for place in nearby_places:
        place_name = place['name'].lower()
        if place_name in road_lower or road_lower in place_name:
            return True
    
    # Check for major interstates that should be nearby
    major_interstates = ['i-95', 'i-81', 'i-64', 'i-66', 'i-395', 'i-495']
    if any(interstate in road_lower for interstate in major_interstates):
        return is_major_road_in_region(road_name, region, lat, lon)
    
    # For local roads, be more permissive
    return True


def is_major_road_in_region(road_name: str, region: str, lat: float, lon: float) -> bool:
    """Determine if a major road should be included based on region and location.
    
    This function validates major roads (interstates, US routes) based on
    their expected presence in different regions of Virginia.
    
    Args:
        road_name: Name of the road to validate
        region: Regional classification
        lat: Latitude of the location
        lon: Longitude of the location
        
    Returns:
        True if the road should be included
        
    Example:
        >>> is_valid = is_major_road_in_region('I-95', 'Northern Virginia', 38.8, -77.0)
        >>> is_valid
        True
    """
    road_lower = road_name.lower()
    
    # I-95 runs through most of Virginia
    if 'i-95' in road_lower:
        return True
    
    # I-81 runs through western Virginia
    if 'i-81' in road_lower:
        return region in ['Valley', 'Southwest', 'Central Virginia']
    
    # I-64 runs through central and eastern Virginia
    if 'i-64' in road_lower:
        return region in ['Tidewater', 'Central Virginia', 'Northern Virginia']
    
    # I-66 runs through Northern Virginia
    if 'i-66' in road_lower:
        return region == 'Northern Virginia'
    
    # I-395 and I-495 are Northern Virginia specific
    if any(interstate in road_lower for interstate in ['i-395', 'i-495']):
        return region == 'Northern Virginia'
    
    # I-264 and I-564 are Tidewater specific
    if any(interstate in road_lower for interstate in ['i-264', 'i-564']):
        return region == 'Tidewater'
    
    # US routes are generally more widespread
    if road_name.startswith('US-'):
        return True
    
    # State routes are region-specific
    if road_name.startswith('VA-'):
        return True
    
    return False


def is_obviously_incorrect_road(road_name: str, region: str) -> bool:
    """
    Filter out roads that are clearly misclassified by region.
    
    This function identifies roads that are obviously in the wrong region
    based on their names and expected geographic distribution.
    
    Args:
        road_name (str): Name of the road to validate
        region (str): Regional classification of the road
        
    Returns:
        bool: True if the road should be filtered out
        
    Example:
        >>> is_incorrect = is_obviously_incorrect_road('Natural Bridge', 'Northern Virginia')
        >>> is_incorrect
        True
        >>> is_incorrect = is_obviously_incorrect_road('I-395', 'Northern Virginia')
        >>> is_incorrect
        False
        
    Algorithm:
        1. Check for western Virginia indicators in Northern Virginia
        2. Check for beach/coastal indicators in inland regions
        3. Check for mountain/valley indicators in flat regions
        4. Return filtering decision
        
    Performance:
        - Time Complexity: O(1)
        - Space Complexity: O(1)
        - Typical runtime: ~0.00001 seconds
        
    Note:
        This function helps maintain geographic accuracy by filtering out
        roads that are clearly misclassified based on their names and
        expected regional distribution.
    """
    road_lower = road_name.lower()
    
    # Roads that are clearly in western Virginia but might be misclassified
    western_va_indicators = [
        'natural bridge', 'high bridge', 'shenandoah', 'mudlick',
        'stone road', 'hawkins mill', 'riverside', 'valley',
        'mudlickrd', 'riversidedr', 'shenandoah ave', 'stone roadhawkins mill'
    ]
    
    # If it's a Northern Virginia region but has western VA indicators, it's likely wrong
    if region == 'Northern Virginia' and any(indicator in road_lower for indicator in western_va_indicators):
        return True
    
    # Roads that are clearly in other regions but might be misclassified
    if region == 'Northern Virginia' and any(indicator in road_lower for indicator in ['beach', 'terminal', 'portsmouth']):
        return True
    
    # Filter out roads with obviously incorrect names for the region
    if region == 'Northern Virginia' and any(indicator in road_lower for indicator in [
        'shenandoah', 'valley', 'mountain', 'ridge', 'gap', 'natural bridge', 'high bridge'
    ]):
        return True
    
    # Filter out roads with concatenated names that are clearly wrong
    if region == 'Northern Virginia' and any(indicator in road_lower for indicator in [
        'mudlickrd', 'riversidedr', 'shenandoah ave', 'stone roadhawkins mill'
    ]):
        return True
    
    # Filter out roads that are clearly too far from Northern Virginia
    if region == 'Northern Virginia' and any(indicator in road_lower for indicator in [
        'natural bridge', 'high bridge trail', 'mudlick', 'shenandoah'
    ]):
        return True
    
    return False
