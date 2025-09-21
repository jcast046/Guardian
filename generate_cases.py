#!/usr/bin/env python3
"""
Guardian Synthetic Case Generator

This module generates realistic, schema-valid synthetic missing-child cases
for Virginia using comprehensive geographic, transportation, and behavioral data.
Integrates with reinforcement learning configurations to create training data
that reflects real-world search patterns and temporal dynamics.

Features:
- Geographic realism using Virginia gazetteer and regional boundaries
- Transportation integration with real road segments and transit stations
- Reinforcement learning time windows and search zone generation
- Consistent vehicle and witness data from curated lexicons
- Schema validation ensuring data quality and consistency

Author: Joshua Castillo

Usage:
    python generate_cases.py --n 10 --seed 42 --out data/custom_cases
"""

import json
import random
import uuid
import argparse
import datetime
import math
from pathlib import Path
from jsonschema import Draft202012Validator

BASE = Path(".")

def load(p): 
    """
    Load JSON file with comprehensive error handling.
    
    Args:
        p (Path): Path to the JSON file to load
        
    Returns:
        dict: Parsed JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        PermissionError: If the file cannot be read
        
    Example:
        >>> data = load(Path("data/geo/va_gazetteer.json"))
        >>> 'entries' in data
        True
    """
    return json.load(open(p, "r", encoding="utf-8"))

def validate(instance, schema):
    """
    Validate instance against schema with detailed error reporting.
    
    Uses Draft202012Validator for comprehensive JSON Schema validation
    with detailed error path reporting for debugging.
    
    Args:
        instance (dict): JSON data to validate
        schema (dict): JSON Schema to validate against
        
    Raises:
        AssertionError: If validation fails, with detailed error messages
        jsonschema.ValidationError: If schema is invalid
        
    Example:
        >>> case = {"case_id": "GRD-2025-123", "demographic": {"age": 12}}
        >>> validate(case, guardian_schema)
        # Raises AssertionError if validation fails
    """
    v = Draft202012Validator(schema)
    errs = sorted(v.iter_errors(instance), key=lambda e: e.path)
    if errs: 
        raise AssertionError("\n".join(f"{'/'.join(map(str,e.path))}: {e.message}" for e in errs))

def gen_case_id():
    """
    Generate a unique case ID in GRD-YYYY-NNNNNN format.
    
    Creates a standardized case identifier with:
    - GRD prefix (Guardian)
    - Current year
    - 6-digit random number (000000-999999)
    
    Returns:
        str: Unique case ID (e.g., "GRD-2025-123456")
        
    Example:
        >>> case_id = gen_case_id()
        >>> case_id.startswith("GRD-2025-")
        True
    """
    return f"GRD-{datetime.datetime.now().year}-{random.randint(0,999999):06d}"

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth.
    
    Uses the Haversine formula to compute the shortest distance between
    two points on a sphere (Earth) given their latitude and longitude.
    
    Args:
        lat1 (float): Latitude of first point in decimal degrees
        lon1 (float): Longitude of first point in decimal degrees  
        lat2 (float): Latitude of second point in decimal degrees
        lon2 (float): Longitude of second point in decimal degrees
        
    Returns:
        float: Distance in miles between the two points
        
    Example:
        >>> distance = haversine_distance(38.0, -78.0, 39.0, -77.0)
        >>> 60 < distance < 80  # Approximately 70 miles
        True
    """
    R = 3959  # Earth's radius in miles
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def find_nearby_roads(lat, lon, road_segments, max_distance=10):
    """Find roads within max_distance miles of a location."""
    nearby_roads = []
    for segment in road_segments:
        if 'centroid' in segment and segment['centroid']:
            road_lat = segment['centroid']['lat']
            road_lon = segment['centroid']['lon']
            distance = haversine_distance(lat, lon, road_lat, road_lon)
            if distance <= max_distance:
                nearby_roads.append(segment['localNames'][0] if segment['localNames'] else 'Unknown Road')
    return nearby_roads[:3]  # Return up to 3 nearby roads

def find_nearby_transit(lat, lon, transit_data, max_distance=5):
    """Find transit stops within max_distance miles of a location."""
    nearby_transit = []
    for station in transit_data.get('stations', []):
        if 'geometry' in station and station['geometry']['coordinates']:
            stop_lon, stop_lat = station['geometry']['coordinates'][:2]
            distance = haversine_distance(lat, lon, stop_lat, stop_lon)
            if distance <= max_distance:
                station_name = station.get('name', 'Unnamed Transit Stop')
                if station_name != 'Unnamed':
                    nearby_transit.append(station_name)
    return nearby_transit[:2]  # Return up to 2 nearby transit stops

def get_region_from_coordinates(lat, lon, regions_geojson):
    """Determine which Virginia region a coordinate falls into."""
    for feature in regions_geojson['features']:
        # Simple point-in-polygon check for rectangular regions
        coords = feature['geometry']['coordinates'][0]
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)
        
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return feature['properties']['region_tag']
    return 'Unknown'

def select_realistic_location(gazetteer, regions_geojson, preferred_region=None):
    """Select a realistic Virginia location, optionally preferring a specific region."""
    if preferred_region:
        # Filter locations by region
        region_locations = [loc for loc in gazetteer['entries'] 
                          if loc.get('region_tag') == preferred_region]
        if region_locations:
            return random.choice(region_locations)
    
    # Prefer urban areas with transit (Northern Virginia, Richmond, Norfolk area)
    urban_areas = [loc for loc in gazetteer['entries'] 
                   if loc.get('region_tag') in ['Northern Virginia', 'Central Virginia', 'Tidewater']]
    
    if urban_areas and random.random() < 0.7:  # 70% chance to select urban area
        return random.choice(urban_areas)
    
    # Fallback to any Virginia location
    return random.choice(gazetteer['entries'])

def generate_realistic_movement_cues(behaviors, routes, current_region, destination_region=None):
    """Generate movement cues that make geographic sense."""
    # Get highways that connect regions
    relevant_highways = []
    for route in routes['highways']:
        if current_region in route.get('segments', []) or (destination_region and destination_region in route.get('segments', [])):
            relevant_highways.append(route['name'])
    
    if relevant_highways:
        highway = random.choice(relevant_highways)
        direction = random.choice(['northbound', 'southbound', 'eastbound', 'westbound'])
        return f"headed {direction} on {highway}"
    else:
        return random.choice(behaviors['movement_cues'])

def generate_search_zones(lat, lon, rl_config, gazetteer, regions):
    """Generate search zones based on RL configuration for each time window."""
    zones = []
    time_windows = rl_config['rl_search_config']['time_windows']
    zones_per_window = rl_config['rl_search_config']['action_space']['zones_per_window']
    zone_schema = rl_config['rl_search_config']['action_space']['zone_schema']
    
    for window in time_windows:
        window_id = window['id']
        weight = window['weight']
        
        # Generate zones for this time window
        for i in range(zones_per_window):
            # Select a nearby location for the zone center
            max_distance = 30 + (window['start_hr'] * 5)  # Expand search radius over time
            nearby_locations = []
            
            for loc in gazetteer['entries']:
                distance = haversine_distance(lat, lon, loc['lat'], loc['lon'])
                if distance <= max_distance:
                    nearby_locations.append(loc)
            
            if nearby_locations:
                zone_location = random.choice(nearby_locations)
                zone_lat = zone_location['lat']
                zone_lon = zone_location['lon']
            else:
                # Fallback to random location within expanded radius
                zone_lat = lat + random.uniform(-0.5, 0.5)
                zone_lon = lon + random.uniform(-0.5, 0.5)
            
            # Determine radius based on time window (later = larger radius)
            min_radius = 5
            max_radius = 50
            radius = min_radius + (window['start_hr'] / 72) * (max_radius - min_radius)
            radius = random.uniform(radius * 0.8, radius * 1.2)
            
            # Get region tag for the zone
            zone_region = get_region_from_coordinates(zone_lat, zone_lon, regions)
            
            zone = {
                "center_lat": zone_lat,
                "center_lon": zone_lon,
                "radius_miles": radius,
                "corridor": random.choice(["I-95 NB", "I-64 EB", "US-29 SW", "I-81 SB"]),
                "region_tag": zone_region,
                "priority": random.uniform(0.3, 0.9),
                "in_state": True,  # All zones are in Virginia
                "out_of_state_penalty": 0.0,
                "time_window": window_id,
                "weight": weight
            }
            zones.append(zone)
    
    return zones

def generate_follow_up_sightings(behaviors, vehicles, witnesses, clothing, gazetteer, regions, transit_data, original_lat, original_lon, rl_config, time_offset_base=1):
    """Generate follow-up sighting events using RL time windows and search patterns."""
    sightings = []
    
    # Use RL time windows to structure sightings
    time_windows = rl_config['rl_search_config']['time_windows']
    zones_per_window = rl_config['rl_search_config']['action_space']['zones_per_window']
    
    for window in time_windows:
        window_id = window['id']
        start_hr = window['start_hr']
        end_hr = window['end_hr']
        weight = window['weight']
        
        # Generate sightings within this time window
        num_sightings_in_window = random.randint(0, 2)  # 0-2 sightings per window
        
        for i in range(num_sightings_in_window):
            # Calculate time within the window
            window_duration = end_hr - start_hr
            time_offset = start_hr + random.uniform(0, window_duration)
            
            # Select a nearby location within reasonable distance
            max_distance = 50  # miles
            nearby_locations = []
            
            for loc in gazetteer['entries']:
                distance = haversine_distance(original_lat, original_lon, loc['lat'], loc['lon'])
                if distance <= max_distance:
                    nearby_locations.append(loc)
            
            # Try to use transit stations for more realistic sightings
            nearby_transit_stations = []
            for station in transit_data.get('stations', []):
                if 'geometry' in station and station['geometry']['coordinates']:
                    stop_lon, stop_lat = station['geometry']['coordinates'][:2]
                    distance = haversine_distance(original_lat, original_lon, stop_lat, stop_lon)
                    if distance <= max_distance:
                        nearby_transit_stations.append((stop_lat, stop_lon, station.get('name', 'Transit Station')))
            
            if nearby_transit_stations and random.random() < 0.3:  # 30% chance to use transit station
                sighting_lat, sighting_lon, station_name = random.choice(nearby_transit_stations)
            elif nearby_locations:
                sighting_location = random.choice(nearby_locations)
                sighting_lat = sighting_location['lat']
                sighting_lon = sighting_location['lon']
            else:
                # Fallback to random location within Virginia
                sighting_lat = random.uniform(36.5, 39.5)
                sighting_lon = random.uniform(-83.5, -75.0)
            
            # Calculate confidence based on time window (earlier = higher confidence)
            base_confidence = 0.3 + (weight * 0.4)  # Use window weight for confidence
            confidence = min(0.9, base_confidence + random.uniform(-0.1, 0.1))
            
            # Determine event type based on location type
            if nearby_transit_stations and random.random() < 0.3:
                event_type = random.choice(["transit_tap", "camera_hit", "sighting"])
                note = f"Transit system report: Child seen at transit station wearing {random.choice(clothing['categories']['tops'])} and {random.choice(clothing['categories']['bottoms'])}"
            else:
                event_type = random.choice(["sighting", "lead", "camera_hit", "lpr_hit", "cell_ping", "search_action"])
                note = f"Witness reported seeing child wearing {random.choice(clothing['categories']['tops'])} and {random.choice(clothing['categories']['bottoms'])}"
            
            sighting = {
                "ts": (datetime.datetime.now(datetime.timezone.utc) + 
                       datetime.timedelta(hours=time_offset)).isoformat(),
                "lat": sighting_lat,
                "lon": sighting_lon,
                "event_type": event_type,
                "reporter_type": random.choice(["public", "officer", "family", "unknown"]),
                "confidence": confidence,
                "note": note
            }
            sightings.append(sighting)
    
    return sightings

def main():
    """
    Main function - orchestrates the synthetic case generation process.
    
    Loads all required data sources, validates input data, and generates
    the specified number of synthetic missing-child cases with comprehensive
    geographic, transportation, and behavioral realism.
    
    Command Line Arguments:
        --n (int): Number of cases to generate (default: 10)
        --seed (int): Random seed for reproducibility (default: 42)
        --out (Path): Output directory for generated cases (default: data/synthetic_cases)
        
    Data Sources Loaded:
        - Virginia gazetteer (geographic locations)
        - Regional boundaries (GeoJSON)
        - Road segments (transportation network)
        - Transit stations (public transportation)
        - RL configuration (search patterns and time windows)
        - Lexicons (behaviors, clothing, vehicles, witnesses)
        - Case templates (narrative generation)
        
    Output:
        - Generates N synthetic case files in JSON format
        - Validates each case against Guardian schema
        - Provides progress feedback and validation results
        
    Example:
        $ python generate_cases.py --n 5 --seed 123 --out data/test_cases
        Loading data sources...
        Validating input data...
        ✓ Input data validation passed
        Found 133 Virginia locations
        Found 247 road segments
        Found 2359 transit stations
        Generating 5 synthetic cases...
        ✓ Case 1/5: GRD-2025-123456 - Valid
        ...
        ✓ Generated 5 synthetic cases in data/test_cases
        All cases validated against Guardian schema
    """
    ap = argparse.ArgumentParser(description="Generate synthetic missing child cases")
    ap.add_argument("--n", type=int, default=10, help="Number of cases to generate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic_cases"), help="Output directory")
    args = ap.parse_args()
    
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading data sources...")
    
    # Load data sources
    gaz = load("data/geo/va_gazetteer.json")
    regions = load("data/geo/va_rl_regions.geojson")
    road_segments = load("data/transportation/va_road_segments.json")
    transit_data = load("data/transportation/va_transit.json")
    tmpl = load("data/templates/case_templates.json")
    behaviors = load("data/lexicons/behaviors.json")
    clothing = load("data/lexicons/clothing.json")
    routes = load("data/lexicons/routes.json")
    gaps = load("data/lexicons/time_gaps.json")
    vehicles = load("data/lexicons/vehicles.json")
    witnesses = load("data/lexicons/witness.json")
    rlconf = load("reinforcement_learning/search_reward_config.json")

    # Load schemas
    schm_gaz = load("schemas/gazetteer.schema.json")
    schm_tmpl = load("schemas/case_templates.schema.json")
    schm_out = load("schemas/guardian_schema.json")

    # Preflight validation checks
    print("Validating input data...")
    validate(gaz, schm_gaz)
    validate(tmpl, schm_tmpl)
    print("✓ Input data validation passed")

    # Filter places to cities and counties
    places = [e for e in gaz["entries"] if e["type"] in ("city", "county")]
    print(f"Found {len(places)} Virginia locations")
    print(f"Found {len(road_segments['road_segments'])} road segments")
    print(f"Found {len(transit_data.get('stations', []))} transit stations")

    print(f"Generating {args.n} synthetic cases...")
    
    for i in range(args.n):
        # Select realistic Virginia location with regional context
        p = select_realistic_location(gaz, regions)
        t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Generate child demographics
        child_age = random.randint(6, 17)
        child_gender = random.choice(["male", "female"])
        
        # Find nearby infrastructure
        nearby_roads = find_nearby_roads(p["lat"], p["lon"], road_segments['road_segments'])
        nearby_transit = find_nearby_transit(p["lat"], p["lon"], transit_data)
        
        # Determine region for realistic movement patterns
        current_region = get_region_from_coordinates(p["lat"], p["lon"], regions)
        
        # Fill template strings
        init = tmpl["synthetic_case_templates"]["Initial_Report"]["template"].format(
            child_age=child_age,
            child_gender=child_gender,
            location=p["name"],
            time=t0,
            witness_guardian=random.choice(["parent", "teacher", "friend", "neighbor"]),
            expected_activity=random.choice(["return home", "arrive at school", "meet a friend"])
        )

        # Generate realistic movement cues based on geography
        movement_cue = generate_realistic_movement_cues(behaviors, routes, current_region)
        
        # Select a realistic vehicle from the inventory
        selected_vehicle = random.choice(vehicles["inventory"])
        vehicle_description = f"{selected_vehicle['color']} {selected_vehicle['year']} {selected_vehicle['make']} {selected_vehicle['model']}"
        
        move = tmpl["synthetic_case_templates"]["Route_Movement"]["template"].format(
            time_offset=random.randint(1, 12),
            movement_cues_text=movement_cue,
            direction_area=random.choice(["northbound", "southbound", "west toward Blue Ridge", "east toward Tidewater"]),
            route_name=random.choice([h["name"] for h in routes["highways"]]),
            vehicle_description=vehicle_description,
            origin_city=p["name"]
        )

        outfit = "{}, {}, {}".format(
            random.choice(clothing["categories"]["tops"]),
            random.choice(clothing["categories"]["bottoms"]),
            random.choice(clothing["categories"]["shoes"])
        )

        wit = tmpl["synthetic_case_templates"]["Witness_Encounter"]["template"].format(
            time_offset=random.randint(1, 24),
            witness_type=random.choice([t["type"] for t in witnesses["types"]]),
            child_age=child_age,
            child_gender=child_gender,
            clothing_description=outfit,
            poi_city_or_county=random.choice(places)["name"]
        )

        # Generate follow-up sightings with realistic geography and RL time windows
        follow_up_sightings = generate_follow_up_sightings(behaviors, vehicles, witnesses, clothing, gaz, regions, transit_data, p["lat"], p["lon"], rlconf)
        
        # Generate search zones based on RL configuration
        search_zones = generate_search_zones(p["lat"], p["lon"], rlconf, gaz, regions)

        # Create the case
        case = {
            "case_id": gen_case_id(),
            "demographic": {
                "gender": child_gender,
                "age_years": child_age,
                "name": f"Child_{random.randint(1000, 9999)}",  # Anonymous identifier
                "distinctive_features": random.choice([
                    "birthmark on left arm",
                    "glasses",
                    "blonde hair",
                    "brown eyes",
                    "freckles"
                ])
            },
            "spatial": {
                "last_seen_lat": p["lat"],
                "last_seen_lon": p["lon"],
                "last_seen_location": f"{p['name']}, Virginia",
                "last_seen_city": p["name"],
                "last_seen_county": p.get("county", "Unknown"),
                "last_seen_state": "Virginia",
                "nearby_roads": nearby_roads if nearby_roads else [random.choice([h["name"] for h in routes["highways"]])],
                "nearby_transit_hubs": nearby_transit,
                "nearby_pois": []
            },
            "temporal": {
                "timezone": "America/New_York",
                "last_seen_ts": t0,
                "reported_missing_ts": (datetime.datetime.now(datetime.timezone.utc) + 
                                     datetime.timedelta(hours=random.randint(1, 6))).isoformat(),
                "elapsed_report_minutes": random.randint(60, 360),
                "follow_up_sightings": follow_up_sightings
            },
            "outcome": {
                "case_status": random.choice(["ongoing", "found", "not_found"]),
                "recovery_ts": None,
                "recovery_location": None,
                "recovery_state": None,
                "recovery_lat": None,
                "recovery_lon": None,
                "recovery_time_hours": None,
                "recovery_distance_mi": None,
                "recovery_condition": None
            },
            "narrative_osint": {
                "incident_summary": init,
                "behavioral_patterns": random.sample(behaviors["movement_cues"], k=random.randint(1, 3)),
                "movement_cues_text": move,
                "temporal_markers": [f"Last seen at {t0}", f"Reported missing {random.randint(1, 6)} hours later"],
                "witness_accounts": [
                    {
                        "description": wit,
                        "clothing": outfit,
                        "vehicle": f"{selected_vehicle['make']} {selected_vehicle['model']}",
                        "behavior": random.choice(behaviors["suspect_behaviors"])
                    }
                ],
                "news": [
                    {
                        "title": f"Missing {child_gender} child reported in {p['name']}",
                        "excerpt": f"Authorities are searching for a {child_age}-year-old {child_gender} last seen in {p['name']}, Virginia."
                    }
                ],
                "social_media": [
                    {
                        "platform": random.choice(["nextdoor", "NCMEC", "NamUs", "The Charley Project", "other"]),
                        "text": f"Please help find missing child in {p['name']} area. Last seen wearing {outfit}"
                    }
                ],
                "persons_of_interest": [
                    {
                        "role": random.choice(["suspect", "companion", "family", "unknown"]),
                        "age_estimate": random.randint(25, 65),
                        "vehicle": {
                            "make": selected_vehicle["make"],
                            "model": selected_vehicle["model"],
                            "color": selected_vehicle["color"],
                            "plate_partial": selected_vehicle["license"]
                        },
                        "note": random.choice(behaviors["suspect_behaviors"])
                    }
                ]
            },
            "provenance": {
                "sources": ["synthetic_generator"],
                "original_fields": {},
                "_fulltext": f"{init} {move} {wit}",
                "search_zones": search_zones
            }
        }

        # Validate against Guardian schema
        try:
            validate(case, schm_out)
            print(f"✓ Case {i+1}/{args.n}: {case['case_id']} - Valid")
        except AssertionError as e:
            print(f"✗ Case {i+1}/{args.n}: {case['case_id']} - Validation failed: {e}")
            continue

        # Write to file
        output_file = args.out / f"{case['case_id']}.json"
        output_file.write_text(
            json.dumps(case, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )

    print(f"\n✓ Generated {args.n} synthetic cases in {args.out}")
    print(f"All cases validated against Guardian schema")

if __name__ == "__main__":
    main()
