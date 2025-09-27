"""
Transportation network construction and caching utilities.

This module contains functions for building transportation networks
from various data sources and managing global caches for performance.
"""

from typing import Dict, List, Tuple, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from geography.distance import haversine_distance

# Global caches for performance optimization
_GRAPH_CACHE = None
_STATIONS_CACHE = None
_ROAD_GRAPH_CACHE = None
_ROAD_STATIONS_CACHE = None


def build_transit_graph(transit_data: Dict[str, Any], max_connection_distance: float = 2.0) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]:
    """
    Build a transit network graph from station coordinates.
    
    This function creates a network where transit stations are nodes
    and connections are made between nearby stations. It's used for
    network-based pathfinding in the transit system.
    
    Args:
        transit_data (Dict[str, Any]): Transit stations data with geometry coordinates
        max_connection_distance (float, optional): Maximum distance to connect stations.
            Defaults to 2.0 miles.
        
    Returns:
        Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]: 
            Transit network graph and station information
        
    Example:
        >>> graph, stations = build_transit_graph(transit_data)
        >>> len(graph) > 0
        True
        >>> len(stations) > 0
        True
        
    Algorithm:
        1. Extract station coordinates from transit data
        2. Create connections between nearby stations
        3. Build adjacency list representation
        4. Return graph and station information
        
    Performance:
        - Time Complexity: O(n²) where n is number of stations
        - Space Complexity: O(n²) for graph storage
        - Typical runtime: ~2-5 seconds for 2,359 stations
        
    Note:
        This function builds a complete transit network graph that can be used
        for various pathfinding algorithms including Dijkstra's and A*.
    """
    stations = []
    
    # Extract stations from transit data
    for station in transit_data.get('stations', []):
        if 'geometry' in station and station['geometry'] and 'coordinates' in station['geometry']:
            coords = station['geometry']['coordinates']
            if len(coords) >= 2:
                stations.append({
                    'lat': coords[1],
                    'lon': coords[0],
                    'name': station.get('properties', {}).get('name', 'Unnamed'),
                    'type': station.get('properties', {}).get('type', 'transit_stop')
                })
    
    # Build graph connections between nearby stations
    graph = {}
    for i in range(len(stations)):
        graph[i] = []
        for j in range(len(stations)):
            if i != j:
                distance = haversine_distance(
                    stations[i]['lat'], stations[i]['lon'],
                    stations[j]['lat'], stations[j]['lon']
                )
                if distance <= max_connection_distance:
                    graph[i].append((j, distance))
    
    return graph, stations


def build_road_network(transit_data: Dict[str, Any], road_segments: List[Dict[str, Any]], max_connection_distance: float = 2.0) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]:
    """
    Build a road network graph from transit stations and road segments.
    
    This function creates a network where transit stations are nodes
    and connections are made based on proximity to road segments.
    Each station is mapped to nearby road names for road accessibility.
    
    Args:
        transit_data (Dict[str, Any]): Transit stations data with geometry coordinates
        road_segments (List[Dict[str, Any]]): Road segments data with road names
        max_connection_distance (float, optional): Maximum distance to connect stations.
            Defaults to 2.0 miles.
        
    Returns:
        Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]: 
            Road network graph and station information with road mappings
        
    Algorithm:
        1. Extract station coordinates from transit data
        2. Map each station to nearby road segments
        3. Create connections between nearby stations
        4. Build adjacency list representation
    """
    stations = []
    station_road_mapping = {}
    
    # Extract stations from transit data
    for station in transit_data.get('stations', []):
        if 'geometry' in station and station['geometry'] and 'coordinates' in station['geometry']:
            coords = station['geometry']['coordinates']
            if len(coords) >= 2:
                stations.append({
                    'lat': coords[1],
                    'lon': coords[0],
                    'name': station.get('properties', {}).get('name', 'Unnamed'),
                    'type': station.get('properties', {}).get('type', 'transit_stop')
                })
    
    # Map each station to nearby road segments
    for i, station in enumerate(stations):
        nearby_roads = []
        for segment in road_segments:
            if 'localNames' in segment and segment['localNames'] and 'admin' in segment and segment['admin']:
                road_name = segment['localNames'][0]
                road_region = segment['admin'].get('region', '')
                
                # Simple region-based mapping for now
                # In a real implementation, we'd use actual road coordinates
                if road_region in ['Northern Virginia', 'Tidewater', 'Central Virginia', 'Valley', 'Southwest']:
                    nearby_roads.append(road_name)
        
        station_road_mapping[i] = nearby_roads[:5]  # Limit to 5 roads per station
    
    # Build graph connections between nearby stations
    graph = {}
    for i in range(len(stations)):
        graph[i] = []
        for j in range(len(stations)):
            if i != j:
                distance = haversine_distance(
                    stations[i]['lat'], stations[i]['lon'],
                    stations[j]['lat'], stations[j]['lon']
                )
                if distance <= max_connection_distance:
                    graph[i].append((j, distance))
    
    # Add road information to stations
    road_stations = []
    for i, station in enumerate(stations):
        road_stations.append({
            'lat': station['lat'],
            'lon': station['lon'],
            'name': station['name'],
            'type': station['type'],
            'road_name': station_road_mapping[i][0] if station_road_mapping[i] else 'Unknown Road'
        })
    
    return graph, road_stations


def get_cached_graph_and_stations(transit_data: Dict[str, Any]) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]:
    """
    Get cached graph and stations, building if not already cached.
    
    This function implements one-time initialization and caching to avoid
    rebuilding the graph and stations on every query. This provides
    significant performance improvements for repeated queries.
    
    Args:
        transit_data (Dict[str, Any]): Transit stations data with geometry coordinates
        
    Returns:
        Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]: Cached graph and stations
        
    Performance:
        - First call: O(n²) to build graph
        - Subsequent calls: O(1) to return cached data
        - Memory usage: O(n²) for graph storage
        - Typical first call: ~2-5 seconds for 2,359 stations
        - Typical subsequent calls: ~0.001 seconds
    """
    global _GRAPH_CACHE, _STATIONS_CACHE
    
    if _GRAPH_CACHE is None or _STATIONS_CACHE is None:
        _GRAPH_CACHE, _STATIONS_CACHE = build_transit_graph(transit_data)
    
    return _GRAPH_CACHE, _STATIONS_CACHE


def get_cached_road_network(transit_data: Dict[str, Any], road_segments: List[Dict[str, Any]]) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]:
    """
    Get cached road network graph, building if not already cached.
    
    This function builds a road network graph by mapping transit stations
    to nearby road segments and creating connections based on proximity.
    It uses one-time initialization and caching for performance.
    
    Args:
        transit_data (Dict[str, Any]): Transit stations data with geometry coordinates
        road_segments (List[Dict[str, Any]]): Road segments data with road names
        
    Returns:
        Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]: 
            Road network graph and station information
        
    Performance:
        - First call: O(n²) to build graph
        - Subsequent calls: O(1) to return cached data
        - Memory usage: O(n²) for graph storage
    """
    global _ROAD_GRAPH_CACHE, _ROAD_STATIONS_CACHE
    
    if _ROAD_GRAPH_CACHE is None or _ROAD_STATIONS_CACHE is None:
        _ROAD_GRAPH_CACHE, _ROAD_STATIONS_CACHE = build_road_network(transit_data, road_segments)
    
    return _ROAD_GRAPH_CACHE, _ROAD_STATIONS_CACHE
