"""
Consolidated Guardian modules for organized code structure.

This module contains all the organized functions from the original
generate_cases.py, providing a clean interface for the main script.
"""

import heapq
import random
from typing import Dict, List, Tuple, Any
from pathlib import Path

BASE = Path(".")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth."""
    import math
    
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


def get_region_from_coordinates(lat: float, lon: float, regions_geojson: Dict[str, Any]) -> str:
    """Determine which region a location falls into using GeoJSON boundaries."""
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


def find_nearby_places(lat: float, lon: float, gazetteer_data: Dict[str, Any], max_distance: float = 10.0) -> List[Dict[str, Any]]:
    """Find nearby cities and landmarks from gazetteer data."""
    nearby_places = []
    
    for entry in gazetteer_data.get('entries', []):
        if 'lat' in entry and 'lon' in entry:
            place_lat = entry['lat']
            place_lon = entry['lon']
            
            distance = haversine_distance(lat, lon, place_lat, place_lon)
            if distance <= max_distance:
                nearby_places.append({
                    'name': entry.get('name', 'Unknown'),
                    'type': entry.get('type', 'Unknown'),
                    'region_tag': entry.get('region_tag', 'Unknown'),
                    'lat': place_lat,
                    'lon': place_lon,
                    'distance': distance
                })
    
    # Sort by distance
    nearby_places.sort(key=lambda x: x['distance'])
    
    return nearby_places


def is_geographically_accurate_road(road_name: str, region: str, nearby_places: List[Dict[str, Any]], lat: float, lon: float) -> bool:
    """Determine if a road is geographically accurate based on location context."""
    road_lower = road_name.lower()
    
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
    """Determine if a major road should be included based on region and location."""
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


def clean_road_name(road_name: str) -> str:
    """Clean up concatenated road names and remove duplicates."""
    # Split on common separators and take the first meaningful part
    separators = ['.', 'Rd', 'Ave', 'St', 'Dr', 'Blvd', 'Pkwy', 'Way']
    
    # Find the first separator and take everything before it
    for separator in separators:
        if separator in road_name:
            parts = road_name.split(separator)
            if parts[0].strip():
                return parts[0].strip()
    
    # If no separator found, return the original name
    return road_name.strip()


def find_nearby_roads(lat: float, 
                     lon: float, 
                     road_segments: List[Dict[str, Any]], 
                     regions_geojson: Dict[str, Any], 
                     transit_data: Dict[str, Any], 
                     max_distance: float = 10.0) -> List[str]:
    """Find nearby roads using comprehensive multi-source analysis."""
    # Load additional data sources
    gazetteer_data = load(BASE / "data" / "geo" / "va_gazetteer.json")
    transportation_summary = load(BASE / "data" / "transportation" / "va_transportation_summary.json")
    
    # Determine location's region
    location_region = get_region_from_coordinates(lat, lon, regions_geojson)
    
    # Find nearby cities/landmarks from gazetteer
    nearby_places = find_nearby_places(lat, lon, gazetteer_data, max_distance)
    
    # Map GeoJSON region names to road data region names
    region_mapping = {
        'NoVA': 'Northern Virginia',
        'Tidewater': 'Tidewater', 
        'Piedmont': 'Central Virginia',
        'Shenandoah': 'Valley',
        'Appalachia': 'Southwest'
    }
    
    road_region_name = region_mapping.get(location_region, location_region)
    
    # Get comprehensive road list from transportation summary
    all_roads = []
    if 'summary' in transportation_summary:
        summary = transportation_summary['summary']
        all_roads.extend(summary.get('interstates', {}).get('items', []))
        all_roads.extend(summary.get('us_routes', {}).get('items', []))
        all_roads.extend(summary.get('state_routes', {}).get('items', []))
    
    # Find roads from road segments that match the region
    nearby_roads = []
    seen_roads = set()
    
    for segment in road_segments:
        if 'localNames' in segment and segment['localNames'] and 'admin' in segment and segment['admin']:
            road_name = segment['localNames'][0]
            road_region = segment['admin'].get('region', '')
            
            # Match based on region
            if road_region_name == road_region:
                # Enhanced filtering for geographic accuracy
                if is_geographically_accurate_road(road_name, road_region_name, nearby_places, lat, lon):
                    cleaned_road_name = clean_road_name(road_name)
                    
                    if cleaned_road_name not in seen_roads:
                        seen_roads.add(cleaned_road_name)
                        nearby_roads.append(cleaned_road_name)
    
    # Add major roads from transportation summary if they're in the region
    for road in all_roads:
        if is_major_road_in_region(road, road_region_name, lat, lon):
            if road not in seen_roads:
                seen_roads.add(road)
                nearby_roads.append(road)
    
    return nearby_roads[:15]  # Limit to 15 roads for practical use


def find_nearby_transit(lat: float, lon: float, transit_data: Dict[str, Any], max_distance: float = 15.0) -> List[str]:
    """Find nearby transit hubs using network pathfinding."""
    # Get cached graph and stations
    graph, stations = get_cached_graph_and_stations(transit_data)
    
    # Find nearby stations using bounded Dijkstra's algorithm
    nearby_stations = bounded_shortest_paths(graph, stations, lat, lon, max_distance)
    
    # Convert to formatted transit hub strings
    transit_hubs = []
    seen_locations = set()
    
    for item in nearby_stations:
        station = item['station']
        distance = item['distance']
        
        # Create location key to avoid duplicates
        location_key = (round(station['lat'], 4), round(station['lon'], 4))
        if location_key in seen_locations:
            continue
        seen_locations.add(location_key)
        
        # Format station name
        station_name = station.get('name', 'Unnamed Transit Stop')
        if station_name == 'Unnamed':
            station_name = f"Transit Stop at {station.get('region', 'Unknown Location')}"
        
        # Create detailed transit hub entry as formatted string
        transit_hub_string = f"{station_name} (lat: {station['lat']:.4f}, lon: {station['lon']:.4f}, distance: {distance:.2f}mi, type: {station.get('type', 'transit_stop')})"
        transit_hubs.append(transit_hub_string)
    
    return transit_hubs


def find_nearby_pois(lat: float, lon: float, gazetteer_data: Dict[str, Any], max_distance: float = 10.0) -> List[str]:
    """Find nearby points of interest using gazetteer data."""
    nearby_places = find_nearby_places(lat, lon, gazetteer_data, max_distance)
    
    # Convert to POI strings
    pois = []
    for place in nearby_places[:5]:  # Limit to 5 POIs
        pois.append(f"{place['name']} ({place['type']})")
    
    return pois


# Global caches for performance optimization
_GRAPH_CACHE = None
_STATIONS_CACHE = None


def get_cached_graph_and_stations(transit_data: Dict[str, Any]) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]:
    """Get cached graph and stations, building if not already cached."""
    global _GRAPH_CACHE, _STATIONS_CACHE
    
    if _GRAPH_CACHE is None or _STATIONS_CACHE is None:
        _GRAPH_CACHE, _STATIONS_CACHE = build_transit_graph(transit_data)
    
    return _GRAPH_CACHE, _STATIONS_CACHE


def build_transit_graph(transit_data: Dict[str, Any], max_connection_distance: float = 2.0) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Dict[str, Any]]]:
    """Build a transit network graph from station coordinates."""
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


def bounded_shortest_paths(graph: Dict[int, List[Tuple[int, float]]], 
                          stations: List[Dict[str, Any]], 
                          start_lat: float, 
                          start_lon: float, 
                          max_distance: float = 10.0) -> List[Dict[str, Any]]:
    """Find all stations within max_distance using bounded Dijkstra's algorithm."""
    # Find the closest station to the start point
    start_station_idx = None
    min_distance = float('inf')
    
    for i, station in enumerate(stations):
        distance = haversine_distance(start_lat, start_lon, station['lat'], station['lon'])
        if distance < min_distance:
            min_distance = distance
            start_station_idx = i
    
    if start_station_idx is None:
        return []
    
    # Use bounded Dijkstra's to find all stations within max_distance
    results = bounded_dijkstra(graph, start_station_idx, max_distance)
    
    # Convert to formatted results
    nearby_stations = []
    for vertex_idx, distance in results:
        nearby_stations.append({
            'station': stations[vertex_idx],
            'distance': distance
        })
    
    return nearby_stations


def bounded_dijkstra(graph: Dict[int, List[Tuple[int, float]]], 
                     start_idx: int, 
                     cutoff: float) -> List[Tuple[int, float]]:
    """Bounded Dijkstra's algorithm for finding all nodes within cutoff distance."""
    dist = {start_idx: 0.0}
    pq = [(0.0, start_idx)]
    out = []
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > cutoff: 
            break
        out.append((u, d))
        
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')) and nd <= cutoff:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    
    return out


def load(p: Path) -> Dict[str, Any]:
    """Load JSON file with comprehensive error handling."""
    import json
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {p}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {p}: {e}")
    except Exception as e:
        raise Exception(f"Error loading {p}: {e}")
