#!/usr/bin/env python3
"""
Guardian LLM Analysis Script

This script provides a comprehensive LLM analysis pipeline for processing missing person cases
in the Guardian project. It supports both full LLM processing and minimal deterministic modes,
optimized for different phases of the project lifecycle.

Key Features:
- Multi-stage LLM processing (summarizer, extractor, weak labeler)
- Minimal deterministic mode (no LLM dependencies)
- Geographic data processing with Virginia gazetteer integration
- Comprehensive case validation and quality assurance
- EDA (Exploratory Data Analysis) pipeline for data insights
- Performance optimization with stage-by-stage processing

Architecture:
- Phase switches control LLM usage (PHASE_MINIMAL, USE_SUMMARIZER, etc.)
- Geographic normalization using Virginia gazetteer data
- Deterministic entity extraction for minimal mode
- Comprehensive case narrative generation
- Multi-format output support (JSON, JSONL, structured data)

Usage:
    python run_all_llms.py                    # Run minimal EDA pipeline
    python run_all_llms.py --reasoned         # Run with LLM enhancement
    python run_all_llms.py --do-summary       # Include summarization
    python run_all_llms.py --fallback-extractor # Use fallback extraction

Author: Joshua Castillo
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any
import traceback

# =============================================================================
# PHASE CONFIGURATION AND FEATURE FLAGS
# =============================================================================
# These flags control the behavior of the Guardian analysis pipeline,
# allowing for different operational modes based on project phase and requirements.

# -------- LLM PROCESSING SWITCHES --------
# Core phase control - determines whether to use LLM processing or deterministic methods
PHASE_MINIMAL = False      

# Individual component switches for fine-grained control
USE_SUMMARIZER = True      # enable bullet point summaries for richer analysis
USE_LABELER   = True      # enable movement/risk assessment for case analysis
USE_LLM_EXTRACTOR = True   # enable LLM-based entity extraction for enhanced data
WRITE_VERBOSE_NARRATIVE = True  # enable verbose narrative generation

# =============================================================================
# CONDITIONAL IMPORTS AND MODULE INITIALIZATION
# =============================================================================
# Dynamically import LLM modules based on phase configuration.
# In minimal mode, LLM functions are replaced with no-op stubs to avoid
# dependency issues and improve performance for deterministic processing.

if not PHASE_MINIMAL:
    # Full LLM processing mode - import all Guardian LLM modules
    from guardian_llm import (
        summarize,           # Case summarization using Llama-3.2-3B-Instruct
        label_case,          # Combined movement/risk labeling using Qwen2.5-3B-Instruct
        classify_movement,    # Movement pattern classification using Qwen2.5-3B-Instruct
        assess_risk          # Risk level assessment using Qwen2.5-3B-Instruct
    )
    from guardian_llm.extractor import extract_json  # JSON entity extraction using Qwen2.5-3B-Instruct
    from guardian_llm.summarizer import release as release_sum  # Model cleanup
    from guardian_llm.extractor import release as release_ext   # Model cleanup
    from guardian_llm.weak_labeler import release as release_lbl  # Model cleanup
    
    
    try:
        from guardian_llm.extractor import batch_extract_json
    except ImportError:
        batch_extract_json = None
else:
    # Minimal mode - use no-op stubs to avoid LLM dependencies
    summarize = None
    label_case = None
    classify_movement = None
    assess_risk = None
    extract_json = None
    release_sum = lambda: None  # No-op cleanup functions
    release_ext = lambda: None
    release_lbl = lambda: None

# =============================================================================
# CORE UTILITY FUNCTIONS AND CONSTANTS
# =============================================================================
# Essential helper functions for data processing, formatting, and validation
# used throughout the Guardian analysis pipeline.

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Visual separator for formatted output
SECTION = "—" * 56

def _fmt_mins(m: int) -> str:
    """
    Format minutes as human-readable time string.
    
    Converts minutes to hours and minutes format (e.g., 90 minutes -> "1h 30m").
    Handles None values gracefully by returning "unknown".
    
    Args:
        m (int): Number of minutes to format
        
    Returns:
        str: Formatted time string (e.g., "1h 30m", "45m", "unknown")
        
    Examples:
        >>> _fmt_mins(90)
        '1h 30m'
        >>> _fmt_mins(45)
        '45m'
        >>> _fmt_mins(None)
        'unknown'
    """
    if m is None:
        return "unknown"
    h, r = divmod(int(m), 60)
    return f"{h}h {r}m" if h else f"{r}m"

def _fmt_dt(ts_iso: str, tz_name: str = "America/New_York") -> tuple[str, str]:
    """
    Format ISO timestamp to UTC and local timezone strings.
    
    Converts an ISO timestamp to both UTC and local timezone representations.
    Handles timezone conversion and provides fallback for invalid timestamps.
    
    Args:
        ts_iso (str): ISO format timestamp string
        tz_name (str): Target timezone name (default: "America/New_York")
        
    Returns:
        tuple[str, str]: (UTC_iso_string, local_formatted_string)
        
    Examples:
        >>> _fmt_dt("2024-01-15T14:30:00Z")
        ('2024-01-15T14:30:00+00:00', '2024-01-15 09:30 EST')
        >>> _fmt_dt("")
        ('', '')
    """
    if not ts_iso:
        return ("", "")
    try:
        # Parse ISO timestamp and convert to UTC
        dt_utc = datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        # Convert to local timezone
        dt_local = dt_utc.astimezone(ZoneInfo(tz_name))
        return (dt_utc.isoformat(), dt_local.strftime("%Y-%m-%d %H:%M %Z"))
    except Exception:
        # Fallback to original string if parsing fails
        return (ts_iso, "")

def _abbr_state(s: str | None) -> str | None:
    """
    Normalize state names to standard abbreviations.
    
    Converts various forms of Virginia state names to the standard "VA" abbreviation.
    Handles case variations and returns original string if not recognized.
    
    Args:
        s (str | None): State name string to normalize
        
    Returns:
        str | None: Normalized state abbreviation or original string
        
    Examples:
        >>> _abbr_state("Virginia")
        'VA'
        >>> _abbr_state("va")
        'VA'
        >>> _abbr_state("California")
        'California'
        >>> _abbr_state(None)
        None
    """
    if not s: return None
    s = s.strip()
    return {"Virginia":"VA","va":"VA","Va":"VA"}.get(s, s)

def _pick(*vals, default=None):
    """
    Select first non-empty value from a list of candidates.
    
    Iterates through provided values and returns the first one that is not
    None, empty string, empty list, or empty dictionary. Useful for data
    extraction with fallback values.
    
    Args:
        *vals: Variable number of candidate values to check
        default: Default value to return if all candidates are empty
        
    Returns:
        Any: First non-empty value or default
        
    Examples:
        >>> _pick(None, "", "valid", "backup")
        'valid'
        >>> _pick(None, "", [], {})
        None
        >>> _pick(None, "", [], {}, default="fallback")
        'fallback'
    """
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return default

# =============================================================================
# MINIMAL MODE DATA PROCESSING HELPERS
# =============================================================================
# Specialized functions for deterministic data processing in minimal mode.
# These functions handle geographic normalization, entity extraction, and
# data validation without requiring LLM dependencies.

import re

# --- Virginia Independent Cities Registry ---
# These cities have county-level status in Virginia and should be treated
# as both city and county for geographic processing purposes.
_INDEP_CITIES = {
    "Alexandria","Bedford","Bristol","Buena Vista","Charlottesville","Chesapeake",
    "Colonial Heights","Covington","Danville","Emporia","Fairfax","Falls Church",
    "Franklin","Fredericksburg","Galax","Hampton","Harrisonburg","Hopewell",
    "Lexington","Lynchburg","Manassas","Manassas Park","Martinsville","Newport News",
    "Norfolk","Norton","Petersburg","Poquoson","Portsmouth","Radford","Richmond",
    "Roanoke","Salem","Staunton","Suffolk","Virginia Beach","Waynesboro","Williamsburg",
    "Winchester"
}

def _norm_city(s: str | None) -> str | None:
    """
    Normalize city names to standard format.
    
    Converts city names to proper case format and filters out invalid values.
    Handles common placeholder values and empty strings.
    
    Args:
        s (str | None): Raw city name string
        
    Returns:
        str | None: Normalized city name or None if invalid
        
    Examples:
        >>> _norm_city("richmond")
        'Richmond'
        >>> _norm_city("NEW YORK")
        'New York'
        >>> _norm_city("unknown")
        None
        >>> _norm_city("")
        None
    """
    if not s or str(s).strip().lower() in {"unknown","n/a","na","none",""}:
        return None
    return " ".join(w.capitalize() for w in str(s).strip().split())

def _norm_county(c: str | None) -> str | None:
    """
    Normalize county names to standard format.
    
    Removes "county" suffix, converts to proper case, and handles Virginia
    independent cities that function as counties.
    
    Args:
        c (str | None): Raw county name string
        
    Returns:
        str | None: Normalized county name or None if invalid
        
    Examples:
        >>> _norm_county("fairfax county")
        'Fairfax'
        >>> _norm_county("richmond")
        'Richmond'
        >>> _norm_county("unknown")
        None
    """
    if not c or str(c).strip().lower() in {"unknown","n/a","na","none",""}:
        return None
    # Remove "county" suffix (case-insensitive)
    c = re.sub(r"\bcounty\b", "", str(c), flags=re.I).strip()
    c = " ".join(w.capitalize() for w in c.split())
    # If someone passed an independent city into "county", just keep it.
    if c in _INDEP_CITIES:
        return c
    return c

def _load_gazetteer_unified(path="va_gazetteer.json"):
    """
    Load and normalize Virginia gazetteer data from JSON file.
    
    Supports two gazetteer formats:
    Format A: {"entries":[{"name":"Richmond","lat":..., "lon":..., "type":"city"}, ...]}
    Format B: {"Richmond":{"lat":...,"lon":...,"type":"city","aliases":[...]}, ...}
    
    Creates a unified lookup table with lowercase keys and alias support.
    Handles missing files gracefully by returning empty dictionary.
    
    Args:
        path (str): Path to gazetteer JSON file (default: "va_gazetteer.json")
        
    Returns:
        dict: Unified lookup table with structure:
              {name_lower: {"lat": float, "lon": float, "type": str}}
              
    Examples:
        >>> gaz = _load_gazetteer_unified("va_gazetteer.json")
        >>> gaz["richmond"]["lat"]
        37.5407
        >>> gaz["alexandria"]["type"]
        'city'
    """
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        gz = json.load(f)
    lut = {}
    
    # Format A: entries array format
    if isinstance(gz, dict) and "entries" in gz and isinstance(gz["entries"], list):
        for e in gz["entries"]:
            name = e.get("name")
            if not name: 
                continue
            key = name.lower()
            lut[key] = {"lat": e.get("lat"), "lon": e.get("lon"), "type": e.get("type", "")}
            # Add aliases to lookup table
            for a in (e.get("aliases") or []):
                lut[str(a).lower()] = lut[key]
    
    # Format B: direct object format
    elif isinstance(gz, dict):
        for name, rec in gz.items():
            key = str(name).lower()
            lut[key] = {
                "lat": rec.get("lat") or rec.get("centroid_lat") or (rec.get("centroid") or [None,None])[0],
                "lon": rec.get("lon") or rec.get("centroid_lon") or (rec.get("centroid") or [None,None])[1],
                "type": rec.get("type","")
            }
            # Add aliases to lookup table
            for a in (rec.get("aliases") or []):
                lut[str(a).lower()] = lut[key]
    return lut

# Global gazetteer lookup table - initialized once for performance
_GAZ = _load_gazetteer_unified()  

def _geocode_from_gaz(city: str | None, county: str | None):
    """
    Geocode location using Virginia gazetteer data.
    
    Attempts to find coordinates for a location using the gazetteer lookup table.
    Prioritizes county-level geocoding, then falls back to city-level.
    For Virginia independent cities, county and city are the same, so county lookup succeeds.
    
    Args:
        city (str | None): City name to geocode
        county (str | None): County name to geocode
        
    Returns:
        tuple[float, float] | tuple[None, None]: (latitude, longitude) or (None, None) if not found
        
    Examples:
        >>> _geocode_from_gaz("Richmond", "Richmond City")
        (37.5407, -77.4360)
        >>> _geocode_from_gaz("Unknown City", None)
        (None, None)
    """
    # Try county first (higher priority for Virginia independent cities)
    if county:
        hit = _GAZ.get(str(county).lower())
        if hit and isinstance(hit.get("lat"), (int,float)) and isinstance(hit.get("lon"), (int,float)):
            return hit["lat"], hit["lon"]
    
    # Fall back to city if county lookup failed
    if city:
        hit = _GAZ.get(str(city).lower())
        if hit and isinstance(hit.get("lat"), (int,float)) and isinstance(hit.get("lon"), (int,float)):
            return hit["lat"], hit["lon"]
    
    return None, None

def _gender_to_MF(g):
    """
    Normalize gender values to standard M/F format.
    
    Converts various gender representations to standardized M/F codes.
    Handles case variations and partial matches.
    
    Args:
        g: Gender value to normalize (any type)
        
    Returns:
        str | None: "M", "F", or None if unrecognized
        
    Examples:
        >>> _gender_to_MF("male")
        'M'
        >>> _gender_to_MF("female")
        'F'
        >>> _gender_to_MF("m")
        'M'
        >>> _gender_to_MF("unknown")
        None
    """
    if g is None: return None
    g = str(g).strip().lower()
    if g.startswith("m"): return "M"
    if g.startswith("f"): return "F"
    return None

def _iso_utc(ts):
    """
    Normalize timestamp to ISO UTC format.
    
    Converts timestamps to standard ISO UTC format with Z suffix.
    Handles various input formats and provides fallback for invalid timestamps.
    
    Args:
        ts: Timestamp to normalize (any type)
        
    Returns:
        str | None: ISO UTC timestamp string or None if invalid
        
    Examples:
        >>> _iso_utc("2024-01-15T14:30:00+00:00")
        '2024-01-15T14:30:00Z'
        >>> _iso_utc("2024-01-15T14:30:00.123Z")
        '2024-01-15T14:30:00Z'
        >>> _iso_utc(None)
        None
    """
    if not ts:
        return None
    try:
        # pass through if already ISO-ish; drop subseconds for neatness
        return str(ts).replace("+00:00", "Z")
    except Exception:
        return None

def to_min_record(case, gaz_lut):
    """
    Build minimal case record for Phase processing.
    
    Extracts essential fields from a case dictionary and creates a standardized
    minimal record suitable for EDA and geographic analysis. Handles missing
    data gracefully and performs geographic normalization.
    
    Args:
        case (dict): Full case dictionary with nested structure
        gaz_lut (dict): Gazetteer lookup table for geocoding
        
    Returns:
        dict: Minimal case record with keys:
              - case_id: Unique case identifier
              - age: Person's age in years (int)
              - gender: Gender code ("M" or "F")
              - county: County name (normalized)
              - state: State abbreviation (default "VA")
              - city: City name (normalized)
              - lat: Latitude coordinate (float)
              - lon: Longitude coordinate (float)
              - date_reported: ISO UTC timestamp string
              
    Examples:
        >>> case = {"case_id": "GRD-001", "demographic": {"age_years": 15, "gender": "male"}}
        >>> to_min_record(case, {})
        {'case_id': 'GRD-001', 'age': 15, 'gender': 'M', ...}
    """
    cid = case.get("case_id") or case.get("id") or "UNKNOWN"

    demo = case.get("demographic", {}) or {}
    age = demo.get("age_years")
    try:
        age = int(round(age)) if age is not None else None
    except Exception:
        age = None
    gender = _gender_to_MF(demo.get("gender"))

    sp = case.get("spatial", {}) or {}
    city   = _norm_city(sp.get("last_seen_city"))
    county = _norm_county(sp.get("last_seen_county"))
    state  = "VA"  # keep canonical

    # if county is missing but city is an independent city, use city as county
    if not county and city in _INDEP_CITIES:
        county = city

    lat = sp.get("last_seen_lat")
    lon = sp.get("last_seen_lon")
    lat = float(lat) if isinstance(lat, (int, float)) else None
    lon = float(lon) if isinstance(lon, (int, float)) else None

    if (lat is None or lon is None) and (city or county):
        glat, glon = _geocode_from_gaz(city, county)
        lat = lat if lat is not None else glat
        lon = lon if lon is not None else glon

    tm = case.get("temporal", {}) or {}
    date_reported = _iso_utc(tm.get("reported_missing_ts"))

    return {
        "case_id": cid,
        "age": age,
        "gender": gender,
        "county": county,
        "state": state,
        "city": city,
        "lat": lat,
        "lon": lon,
        "date_reported": date_reported
    }

def validate_min_record(rec):
    """
    Validate minimal case record for data quality.
    
    Checks essential fields in a minimal case record and returns a list
    of validation issues. Used for data quality assurance in the EDA pipeline.
    
    Args:
        rec (dict): Minimal case record to validate
        
    Returns:
        list[str]: List of validation issues found (empty if valid)
        
    Examples:
        >>> rec = {"case_id": "GRD-001", "age": 15, "gender": "M", "county": "Fairfax", "lat": 38.8, "lon": -77.1}
        >>> validate_min_record(rec)
        []
        >>> rec = {"case_id": "", "age": None, "gender": "X"}
        >>> validate_min_record(rec)
        ['missing case_id', 'missing age', 'missing/invalid gender', 'missing county', 'missing lat/lon']
    """
    issues = []
    if not rec.get("case_id"): issues.append("missing case_id")
    if rec.get("age") is None: issues.append("missing age")
    if rec.get("gender") not in ("M","F"): issues.append("missing/invalid gender")
    if not rec.get("county"): issues.append("missing county")
    if (rec.get("lat") is None) or (rec.get("lon") is None):
        issues.append("missing lat/lon")
    return issues

def eda_counts(min_records):
    """
    Generate EDA (Exploratory Data Analysis) counts for minimal records.
    
    Creates frequency counts for key demographic and geographic fields
    to support data quality analysis and visualization. Handles missing
    values by categorizing them as "Unknown".
    
    Args:
        min_records (list[dict]): List of minimal case records
        
    Returns:
        dict: Counts dictionary with structure:
              {
                  "age": {age: count, ...},
                  "gender": {"M": count, "F": count, ...},
                  "county": {county_name: count, ...},
                  "city": {city_name: count, ...},
                  "state": {state_name: count, ...}
              }
              
    Examples:
        >>> records = [{"age": 15, "gender": "M", "county": "Fairfax"}]
        >>> eda_counts(records)
        {'age': {15: 1}, 'gender': {'M': 1}, 'county': {'Fairfax': 1}, ...}
    """
    from collections import Counter
    age_c = Counter()
    gen_c = Counter()
    cty_c = Counter()
    city_c = Counter()
    state_c = Counter()
    
    for r in min_records:
        # Age counts
        if r.get("age") is not None: age_c[r["age"]] += 1
        
        # Gender counts
        if r.get("gender") in ("M","F"): gen_c[r["gender"]] += 1
        
        # County counts
        county = r.get("county")
        if county is None or county == "null":
            cty_c["Unknown"] += 1
        elif county:
            cty_c[county] += 1
            
        # City counts
        city = r.get("city")
        if city is None or city == "null":
            city_c["Unknown"] += 1
        elif city:
            city_c[city] += 1
            
        # State counts
        state = r.get("state")
        if state is None or state == "null":
            state_c["Unknown"] += 1
        elif state:
            state_c[state] += 1
    
    return {
        "age": dict(sorted(age_c.items())),
        "gender": dict(sorted(gen_c.items())),
        "county": dict(sorted(cty_c.items())),
        "city": dict(sorted(city_c.items())),
        "state": dict(sorted(state_c.items())),
    }
# ---------- helpers ----------

def build_clean_narrative(case: dict) -> dict:
    """
    Build comprehensive case narrative from structured case data.
    
    Creates a well-formatted markdown narrative from a case dictionary with
    nested structure (demographic/spatial/temporal/narrative_osint). Includes
    all relevant case information in organized sections.
    
    Args:
        case (dict): Case dictionary with nested structure containing:
                    - demographic: age, gender, name, features
                    - spatial: location data, coordinates
                    - temporal: timestamps, timezone
                    - narrative_osint: incident details, witnesses, etc.
        
    Returns:
        dict: Narrative dictionary with keys:
              - text: Full markdown narrative with sections
              - short: One-line summary for UI display
              
    Examples:
        >>> case = {"case_id": "GRD-001", "demographic": {"name": "John", "age_years": 15}}
        >>> build_clean_narrative(case)
        {'text': '### Case GRD-001\\n...', 'short': 'John (male, 15y) last seen...'}
    """
    cid = case.get("case_id", "UNKNOWN")

    d = case.get("demographic", {}) or {}
    s = case.get("spatial", {}) or {}
    t = case.get("temporal", {}) or {}
    n = case.get("narrative_osint", {}) or {}

    # Subject
    name   = _pick(d.get("name"), "Unknown child")
    age    = d.get("age_years")
    gender = _pick(d.get("gender"), "").lower()
    gender_str = {"female":"female", "male":"male"}.get(gender, gender or "unknown")
    feats  = d.get("distinctive_features") or "—"

    # Location
    city   = _pick(s.get("last_seen_city"), s.get("last_seen_location"), "Unknown")
    county = _pick(s.get("last_seen_county"), "Unknown")
    state  = _abbr_state(_pick(s.get("last_seen_state"), "VA")) or "VA"
    lat    = s.get("last_seen_lat")
    lon    = s.get("last_seen_lon")

    # Timeline
    tz_name = t.get("timezone", "America/New_York")
    last_seen_iso = t.get("last_seen_ts")
    reported_iso  = t.get("reported_missing_ts")
    last_seen_utc, last_seen_local = _fmt_dt(last_seen_iso, tz_name)
    reported_utc,  reported_local  = _fmt_dt(reported_iso, tz_name)
    ttr_mins = t.get("elapsed_report_minutes")
    ttr_str  = _fmt_mins(ttr_mins)

    # Incident details
    summary   = n.get("incident_summary") or ""
    behaviors = n.get("behavioral_patterns") or []
    cues_text = n.get("movement_cues_text") or ""
    witnesses = n.get("witness_accounts") or []
    news = n.get("news") or []
    social_media = n.get("social_media") or []
    persons_of_interest = n.get("persons_of_interest") or []
    temporal_markers = n.get("temporal_markers") or []

    # Outcome information
    outcome = case.get("outcome", {}) or {}
    case_status = outcome.get("case_status", "")
    recovery_ts = outcome.get("recovery_ts")
    recovery_location = outcome.get("recovery_location")
    recovery_time_hours = outcome.get("recovery_time_hours")
    recovery_distance_mi = outcome.get("recovery_distance_mi")
    recovery_condition = outcome.get("recovery_condition")

    # Follow-up sightings
    follow_up_sightings = t.get("follow_up_sightings", [])

    # Nearby infrastructure
    nearby_roads = s.get("nearby_roads", [])
    nearby_transit = s.get("nearby_transit_hubs", [])
    nearby_pois = s.get("nearby_pois", [])

    # Extract up to first 2 highways from cues text
    import re
    highways = re.findall(r'\b(I-\d{1,3}|US-\d{1,3}|VA-\d{1,3})\b', cues_text)[:2]

    # First witness details 
    w0 = witnesses[0] if witnesses else {}
    w_clothing = w0.get("clothing")
    w_vehicle  = w0.get("vehicle")
    w_behavior = w0.get("behavior")

    # One-line short summary 
    short_bits = [f"{name} ({gender_str}{(',' if age is None else f', {age}y')})",
                  f"last seen {city}, {state}"]
    if last_seen_local: short_bits.append(f"@ {last_seen_local}")
    if w_vehicle:       short_bits.append(f"vehicle: {w_vehicle}")
    if highways:        short_bits.append(f"cues: {', '.join(highways)}")
    short = "; ".join(short_bits)
    if len(short) > 220: short = short[:217] + "..."

    # Clean, sectioned narrative
    lines = []
    lines.append(f"### Case {cid}")
    lines.append(SECTION)
    lines.append("")
    
    # Subject section
    lines.append("**Subject**")
    lines.append(f"- Name/ID: {name}")
    lines.append(f"- Age/Gender: {age if age is not None else '—'} / {gender_str}")
    lines.append(f"- Distinctive features: {feats}")
    lines.append("")

    # Location section
    lines.append("**Last known location**")
    lines.append(f"- City/County/State: {city}, {county}, {state}")
    lines.append(f"- Coordinates: {lat}, {lon}")
    lines.append("")

    # Timeline section
    lines.append("**Timeline**")
    lines.append(f"- Last seen (local): {last_seen_local or '—'}")
    lines.append(f"- Reported missing (local): {reported_local or '—'}")
    lines.append(f"- Time to report: {ttr_str}")
    lines.append("")

    # Incident summary section
    if summary:
        lines.append("**Incident summary**")
        lines.append(f"- {summary}")
        lines.append("")

    # Behavioral indicators section
    if behaviors:
        lines.append("**Behavioral indicators**")
        for b in behaviors:
            lines.append(f"- {b}")
        lines.append("")

    # Movement cues section
    if cues_text:
        lines.append("**Movement cues**")
        if highways: 
            lines.append(f"- Key corridors: {', '.join(highways)}")
        lines.append(f"- Details: {cues_text}")
        lines.append("")

    # Witness reports section
    if witnesses:
        lines.append("**Witness reports**")
        for i, w in enumerate(witnesses, 1):
            desc = w.get("description") or "—"
            lines.append(f"- Witness {i}: {desc}")
            if w.get("clothing"): 
                lines.append(f"  - Clothing: {w['clothing']}")
            if w.get("vehicle"):  
                lines.append(f"  - Vehicle: {w['vehicle']}")
            if w.get("behavior"): 
                lines.append(f"  - Behavior: {w['behavior']}")
        lines.append("")

    # Follow-up sightings section
    if follow_up_sightings:
        lines.append("**Follow-up sightings**")
        for i, sighting in enumerate(follow_up_sightings[:5], 1):  # Limit to first 5
            ts = sighting.get("ts", "")
            lat = sighting.get("lat", "")
            lon = sighting.get("lon", "")
            event_type = sighting.get("event_type", "")
            reporter_type = sighting.get("reporter_type", "")
            confidence = sighting.get("confidence", "")
            note = sighting.get("note", "")
            
            lines.append(f"- Sighting {i}: {event_type} at {lat}, {lon}")
            if ts:
                sighting_utc, sighting_local = _fmt_dt(ts, tz_name)
                lines.append(f"  - Time: {sighting_local or ts}")
            lines.append(f"  - Reporter: {reporter_type}, Confidence: {confidence:.2f}")
            if note:
                lines.append(f"  - Note: {note}")
        lines.append("")

    # News reports section
    if news:
        lines.append("**News reports**")
        for i, article in enumerate(news, 1):
            title = article.get("title", "")
            excerpt = article.get("excerpt", "")
            lines.append(f"- News {i}: {title}")
            if excerpt:
                lines.append(f"  - Excerpt: {excerpt}")
        lines.append("")

    # Social media section
    if social_media:
        lines.append("**Social media**")
        for i, post in enumerate(social_media, 1):
            platform = post.get("platform", "")
            text = post.get("text", "")
            lines.append(f"- Social Media {i} ({platform}): {text}")
        lines.append("")

    # Persons of interest section
    if persons_of_interest:
        lines.append("**Persons of interest**")
        for i, poi in enumerate(persons_of_interest, 1):
            role = poi.get("role", "")
            age_estimate = poi.get("age_estimate", "")
            note = poi.get("note", "")
            vehicle_info = poi.get("vehicle", {})
            
            lines.append(f"- POI {i}: {role}")
            if age_estimate:
                lines.append(f"  - Age Estimate: ~{age_estimate}")
            if note:
                lines.append(f"  - Note: {note}")
            
            if vehicle_info:
                make = vehicle_info.get("make", "")
                model = vehicle_info.get("model", "")
                color = vehicle_info.get("color", "")
                plate = vehicle_info.get("plate_partial", "")
                if make or model or color:
                    vehicle_desc = f"  - Vehicle: {color} {make} {model}"
                    if plate:
                        vehicle_desc += f" (plate: {plate})"
                    lines.append(vehicle_desc)
        lines.append("")

    # Nearby infrastructure section
    if nearby_roads or nearby_transit or nearby_pois:
        lines.append("**Nearby infrastructure**")
        if nearby_roads:
            lines.append(f"- Roads: {', '.join(nearby_roads[:10])}")  # Limit to first 10
        if nearby_transit:
            lines.append(f"- Transit: {len(nearby_transit)} transit stops within range")
        if nearby_pois:
            lines.append(f"- Points of interest: {len(nearby_pois)} POIs")
        lines.append("")

    # Search zones section
    search_zones = case.get("provenance", {}).get("search_zones", [])
    if search_zones:
        lines.append("**Search zones**")
        for i, zone in enumerate(search_zones[:3], 1):  # Limit to first 3
            center_lat = zone.get("center_lat", "")
            center_lon = zone.get("center_lon", "")
            radius_miles = zone.get("radius_miles", "")
            corridor = zone.get("corridor", "")
            region_tag = zone.get("region_tag", "")
            priority = zone.get("priority", "")
            time_window = zone.get("time_window", "")
            
            lines.append(f"- Zone {i}: {corridor} in {region_tag}")
            lines.append(f"  - Center: {center_lat}, {center_lon}")
            lines.append(f"  - Radius: {radius_miles:.1f} miles")
            lines.append(f"  - Priority: {priority:.2f}, Time: {time_window}")
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    return {"text": text, "short": short}

def load_synthetic_cases(data_dir: str = "data/synthetic_cases") -> List[Dict[str, Any]]:
    """
    Load all synthetic case JSON files from the data directory.
    
    Scans the specified directory for JSON files and loads them into memory.
    Handles file loading errors gracefully and provides progress feedback.
    Used for batch processing of synthetic case data.
    
    Args:
        data_dir (str): Path to synthetic cases directory (default: "data/synthetic_cases")
        
    Returns:
        List[Dict[str, Any]]: List of case dictionaries loaded from JSON files
        
    Examples:
        >>> cases = load_synthetic_cases("data/synthetic_cases")
        >>> len(cases)
        500
        >>> cases[0]["case_id"]
        'GRD-001'
    """
    cases = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} not found!")
        return cases
    
    json_files = list(data_path.glob("*.json"))
    print(f"Found {len(json_files)} synthetic case files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
                cases.append(case_data)
                print(f"Loaded: {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    return cases

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SECTION = "—" * 56

def _fmt_mins(m: int) -> str:
    if m is None:
        return "unknown"
    h, r = divmod(int(m), 60)
    return f"{h}h {r}m" if h else f"{r}m"

def _fmt_dt(ts_iso: str, tz_name: str = "America/New_York") -> tuple[str, str]:
    """
    Returns (iso_utc, local_str) — ISO timestamp (UTC) and a pretty local time string.
    """
    if not ts_iso:
        return ("", "")
    try:
        dt_utc = datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        dt_local = dt_utc.astimezone(ZoneInfo(tz_name))
        return (dt_utc.isoformat(), dt_local.strftime("%Y-%m-%d %H:%M %Z"))
    except Exception:
        return (ts_iso, "")  # fall back to raw

def build_narrative(case: dict) -> dict:
    """
    Returns:
      {
        "text": "<markdown narrative>",
        "short": "<one-line summary>"
      }
    """
    cid = case.get("case_id", "UNKNOWN")
    name = case.get("name") or case.get("person_id") or "Unknown"
    age  = case.get("age")
    gender = (case.get("gender") or "").upper()
    gender_str = {"F":"female", "M":"male"}.get(gender, gender or "unknown")
    features = case.get("features") or case.get("distinctive_features") or []
    features_str = ", ".join(features) if isinstance(features, list) else str(features) if features else "—"

    # Location
    loc = case.get("location") or {}
    city   = loc.get("city") or "Unknown"
    county = loc.get("county") or "Unknown"
    state  = loc.get("state") or "VA"
    lat    = case.get("lat") or case.get("latitude")
    lon    = case.get("lon") or case.get("longitude")

    # Timeline
    last_seen_iso = case.get("last_seen_ts")
    reported_iso  = case.get("report_ts")
    last_seen_utc, last_seen_local = _fmt_dt(last_seen_iso)
    reported_utc,  reported_local  = _fmt_dt(reported_iso)
    ttr_mins = case.get("time_to_report_mins")
    ttr_str  = _fmt_mins(ttr_mins)

    # Clothing / Vehicle
    clothing = case.get("clothing") or []
    clothing_str = ", ".join(clothing) if clothing else "—"
    vehicle = case.get("vehicle") or {}
    vehicle_str = " ".join(str(vehicle.get(k)) for k in ("color","year","make","model") if vehicle.get(k)) or "—"
    plate = vehicle.get("plate") or "—"

    # Movement + behaviors + witnesses
    movement_cues = case.get("movement_cues") or []
    behaviors     = case.get("behaviors") or case.get("behavioral_patterns") or []
    witnesses     = case.get("witnesses") or []
    reporter      = case.get("reporter") or case.get("reported_by") or "Unknown"

    # Compact short summary (1 line)
    short = (f"{name} ({gender_str}, {age}y)" if age else f"{name} ({gender_str})")
    short += f" last seen {city}, {state}"
    if last_seen_local:
        short += f" @ {last_seen_local}"
    if vehicle_str != "—":
        short += f"; possible vehicle: {vehicle_str}"
    if movement_cues:
        short += f"; cues: {', '.join(movement_cues[:3])}"
    # Reduce length for dashboards:
    if len(short) > 220:
        short = short[:217] + "..."

    # Markdown narrative 
    lines = []
    lines += [f"### Case {cid}", SECTION]
    lines += [f"**Subject**: {name} — {age} years old, {gender_str}",
              f"**Distinctive features**: {features_str}", ""]
    lines += [f"**Last known location**: {city}, {county}, {state}",
              f"**Coordinates**: {lat}, {lon}",
              f"**Last seen (local)**: {last_seen_local or '—'}",
              f"**Reported missing (local)**: {reported_local or '—'}",
              f"**Time to report**: {ttr_str}", ""]
    lines += [f"**Clothing**: {clothing_str}",
              f"**Vehicle**: {vehicle_str}",
              f"**Plate/partial**: {plate}", ""]
    if movement_cues:
        lines += ["**Movement cues**:"]
        lines += [f"- {cue}" for cue in movement_cues]
        lines.append("")
    if behaviors:
        lines += ["**Behavioral indicators**:"]
        lines += [f"- {b}" for b in behaviors]
        lines.append("")
    if witnesses:
        lines += ["**Witness reports**:"]
        for i, w in enumerate(witnesses, 1):
            w_loc = w.get("location") or {}
            w_city = w_loc.get("city") or w_loc.get("name") or w_loc or "Unknown"
            w_desc = w.get("desc") or w.get("description") or ""
            w_time = _fmt_dt(w.get("ts") or "")[1]
            lines += [f"- Witness {i} ({w.get('type','unknown')}), {w_time or 'time unknown'} — {w_city}"]
            if w_desc:
                lines += [f"  - {w_desc}"]
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    return {"text": text, "short": short}

def backfill_entities(entities: dict, case: dict) -> dict:
    """Fill missing fields deterministically from the synthetic (nested) case JSON."""
    out = dict(entities or {})
    out.setdefault("aliases", [])

    d = case.get("demographic", {}) or {}
    s = case.get("spatial", {}) or {}
    t = case.get("temporal", {}) or {}
    n = case.get("narrative_osint", {}) or {}

    # Name / Age / Gender
    out["name"]   = out.get("name")   or d.get("name")
    out["age"]    = out.get("age")    or d.get("age_years")
    g = (out.get("gender") or d.get("gender") or "").upper()
    out["gender"] = "F" if g.startswith("F") else ("M" if g.startswith("M") else None)

    # Location
    out.setdefault("location", {})
    loc = out["location"]
    loc["city"]   = loc.get("city")   or s.get("last_seen_city") or s.get("last_seen_location")
    loc["county"] = loc.get("county") or s.get("last_seen_county")
    loc["state"]  = loc.get("state")  or _abbr_state(s.get("last_seen_state")) or "VA"

    # Coordinates
    out["lat"] = out.get("lat") or s.get("last_seen_lat")
    out["lon"] = out.get("lon") or s.get("last_seen_lon")

    # Date reported
    out["date_reported"] = out.get("date_reported") or t.get("reported_missing_ts")

    # Movement cues (regex highways + any extractor output)
    import re
    cues = set(out.get("movement_cues") or [])
    cues_text = n.get("movement_cues_text") or ""
    cues.update(re.findall(r'\b(I-\d{1,3}|US-\d{1,3}|VA-\d{1,3})\b', cues_text))
    out["movement_cues"] = sorted(cues) if cues else []

    # Additional entity extraction from new categories
    # Follow-up sightings
    follow_up_sightings = t.get("follow_up_sightings", [])
    if follow_up_sightings:
        out["follow_up_sightings"] = follow_up_sightings[:5]  # Limit to first 5

    # News reports
    news = n.get("news", [])
    if news:
        out["news_reports"] = news

    # Social media
    social_media = n.get("social_media", [])
    if social_media:
        out["social_media"] = social_media

    # Persons of interest
    persons_of_interest = n.get("persons_of_interest", [])
    if persons_of_interest:
        out["persons_of_interest"] = persons_of_interest

    # Nearby infrastructure
    nearby_roads = s.get("nearby_roads", [])
    if nearby_roads:
        out["nearby_roads"] = nearby_roads[:10]  # Limit to first 10

    nearby_transit = s.get("nearby_transit_hubs", [])
    if nearby_transit:
        out["nearby_transit"] = len(nearby_transit)

    nearby_pois = s.get("nearby_pois", [])
    if nearby_pois:
        out["nearby_pois"] = len(nearby_pois)

    # Search zones
    search_zones = case.get("provenance", {}).get("search_zones", [])
    if search_zones:
        out["search_zones"] = search_zones[:3]  # Limit to first 3
        out["search_zones_count"] = len(search_zones)
        # Extract key corridors from search zones
        corridors = [zone.get("corridor", "") for zone in search_zones if zone.get("corridor")]
        if corridors:
            out["search_corridors"] = corridors[:5]  # Limit to first 5

    # Risk factors 
    rf = set(out.get("risk_factors") or [])
    fulltext = " ".join([
        n.get("incident_summary",""), 
        cues_text,
        " ".join(n.get("behavioral_patterns") or []),
        " ".join([w.get("description", "") for w in n.get("witness_accounts", [])]),
        " ".join([poi.get("note", "") for poi in n.get("persons_of_interest", [])])
    ]).lower()

    if (out.get("age") is not None and isinstance(out["age"], (int, float)) and out["age"] <= 12):
        rf.add("minor_under_13")
    if any(k in fulltext for k in ["entered vehicle", "vehicle"]):
        rf.add("involved_vehicle")
    if any(k in fulltext for k in ["i-95","i-81","i-64","us-58"]):
        rf.add("interstate_corridor")
    if any(k in fulltext for k in ["offered a ride", "unknown adult", "asking the child questions"]):
        rf.add("adult_engagement")
    if t.get("elapsed_report_minutes") and t["elapsed_report_minutes"] >= 360:
        rf.add("delayed_report_6h_plus")
    if follow_up_sightings:
        rf.add("has_follow_up_sightings")
    if persons_of_interest:
        rf.add("persons_of_interest_identified")
    if any(k in fulltext for k in ["transit", "bus", "station"]):
        rf.add("transit_related")
    if search_zones:
        rf.add("has_search_zones")
        # Check for high-priority search zones
        high_priority_zones = [z for z in search_zones if z.get("priority", 0) > 0.7]
        if high_priority_zones:
            rf.add("high_priority_search_zones")

    out["risk_factors"] = sorted(rf) if rf else []
    return out

def make_hints(case_json: dict) -> dict:
    """Build hints from case JSON for extractor."""
    return {
        "spatial": {
            "last_seen_city":   case_json.get("spatial", {}).get("last_seen_city"),
            "last_seen_county": case_json.get("spatial", {}).get("last_seen_county"),
            "last_seen_state":  case_json.get("spatial", {}).get("last_seen_state") or "VA",
            "last_seen_lat":    case_json.get("spatial", {}).get("last_seen_lat"),
            "last_seen_lon":    case_json.get("spatial", {}).get("last_seen_lon"),
        },
        "temporal": {
            "reported_missing_ts": case_json.get("temporal", {}).get("reported_missing_ts"),
        },
        "narrative_osint": {
            "movement_cues_text": case_json.get("narrative_osint", {}).get("movement_cues_text", ""),
        }
    }

def build_narratives(case: dict) -> tuple[str, str]:
    """Return (sum_text, ext_text). ext_text keeps the fields the extractor/labeler need."""
    case_id = case.get("case_id", "Unknown")
    d, s, t, n = case.get("demographic", {}), case.get("spatial", {}), case.get("temporal", {}), case.get("narrative_osint", {})

    name  = d.get("name", "Unknown child")
    age   = d.get("age_years", "Unknown")
    gender= d.get("gender", "Unknown")
    feats = d.get("distinctive_features", "")

    # -- "full-ish" narrative with the fields the extractor/labeler need --
    parts = []
    parts.append(f"=== CASE {case_id} ===")
    parts.append(f"Missing Person: {name}, {age}-year-old {gender}")
    if feats: parts.append(f"Distinctive Features: {feats}")

    # Location block (include city/county/state + coords)
    city, county, state = s.get("last_seen_city",""), s.get("last_seen_county",""), s.get("last_seen_state","")
    locline = "Last Seen: "
    if city:   locline += city
    if county: locline += (", " if city else "") + county
    if state:  locline += (", " if (city or county) else "") + state
    elif s.get("last_seen_location"): locline += s["last_seen_location"]
    parts.append(locline)

    if s.get("last_seen_lat") and s.get("last_seen_lon"):
        parts.append(f"Coordinates: {s['last_seen_lat']}, {s['last_seen_lon']}")

    # Timeline (keep both timestamps + elapsed minutes)
    if t.get("last_seen_ts"):        parts.append(f"Last Seen: {t['last_seen_ts']} ({t.get('timezone','')})")
    if t.get("reported_missing_ts"): parts.append(f"Reported Missing: {t['reported_missing_ts']}")
    if t.get("elapsed_report_minutes") is not None:
        parts.append(f"Time to Report: {t['elapsed_report_minutes']} minutes")

    # Incident details (keep movement cues text and behavioral patterns)
    if n.get("incident_summary"): parts.append(f"Summary: {n['incident_summary']}")
    if n.get("behavioral_patterns"):
        parts.append("Behavioral Patterns: " + "; ".join(n["behavioral_patterns"]))
    if n.get("movement_cues_text"): parts.append("Movement Cues: " + n["movement_cues_text"])

    # One witness with clothing/vehicle/behavior (these feed risk/risk_factors)
    w = n.get("witness_accounts", [])
    if w:
        w0 = w[0]
        desc = w0.get("description","")
        clothing = w0.get("clothing","")
        vehicle  = w0.get("vehicle","")
        behavior = w0.get("behavior","")
        parts.append(f"Witness 1: {desc}")
        if clothing: parts.append(f"Clothing: {clothing}")
        if vehicle:  parts.append(f"Vehicle: {vehicle}")
        if behavior: parts.append(f"Behavior: {behavior}")

    # Final rich string for extractor + labeler 
    ext_text = "\n".join(parts)
    if len(ext_text) > 2000:  # Increased from 1600
        ext_text = ext_text[:2000] + "..."

    # Super-stable, concise narrative for LLMs
    sum_parts = []
    sum_parts.append(f"=== CASE {case_id} ===")
    sum_parts.append(f"Person: {name}, {age}-year-old {gender}; features: {feats}")
    sum_parts.append(f"Last seen: {city}, {county}, {state} @ {t.get('last_seen_ts', '')} {t.get('timezone', '')}")
    if s.get("last_seen_lat") and s.get("last_seen_lon"):
        sum_parts.append(f"Coords: {s['last_seen_lat']}, {s['last_seen_lon']}")
    
    # First 1-2 highway/route cues
    if n.get("movement_cues_text"):
        cues = n["movement_cues_text"]
        # Extract first 1-2 highway patterns
        import re
        highways = re.findall(r'\b(I-\d{1,3}|US-\d{1,3}|VA-\d{1,3})\b', cues)
        if highways:
            sum_parts.append(f"Movement: {', '.join(highways[:2])}")
    
    # First witness brief
    if w:
        w0 = w[0]
        witness_parts = []
        if w0.get("clothing"): witness_parts.append(f"clothing: {w0['clothing']}")
        if w0.get("vehicle"): witness_parts.append(f"vehicle: {w0['vehicle']}")
        if w0.get("behavior"): witness_parts.append(f"behavior: {w0['behavior']}")
        if witness_parts:
            sum_parts.append(f"Witness: {'; '.join(witness_parts)}")
    
    sum_text = "\n".join(sum_parts)
    if len(sum_text) > 800:  
        sum_text = sum_text[:800] + "..."

    return sum_text, ext_text

# Speed optimization flags
DO_SUMMARY = True  # summaries needed for UI and analysis

def run_llm_analysis_stage_by_stage(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run all LLM models stage-by-stage for optimal performance.
    
    This approach loads each model once and processes all cases through it,
    avoiding the reload/compile overhead that occurs with case-by-case processing.
    Significantly faster than individual case processing for large datasets.
    
    Processing stages:
    1. Pre-extract narratives (both short and rich formats)
    2. Run summarizer on all cases (if enabled)
    3. Build minimal entities (deterministic, no LLM)
    4. Skip weak labeler 
    5. Assemble final results
    
    Args:
        cases (List[Dict[str, Any]]): List of case data dictionaries
        
    Returns:
        List[Dict[str, Any]]: Results from all LLM models for all cases
        
    Examples:
        >>> cases = [{"case_id": "GRD-001", "demographic": {"age_years": 15}}]
        >>> results = run_llm_analysis_stage_by_stage(cases)
        >>> len(results)
        1
        >>> results[0]["case_id"]
        'GRD-001'
    """
    print(f"\n{'='*80}")
    print(f"STAGE-BY-STAGE LLM PROCESSING")
    print(f"{'='*80}")
    print(f"Processing {len(cases)} cases...")
    
    # Pre-extract narratives once (both short and rich)
    print(f"\n1. Pre-extracting narratives...")
    trimmed = []  # list of (cid, sum_text, ext_text, short_fallback)
    for case in cases:
        cid = case.get("case_id", "Unknown")
        built = build_clean_narrative(case)
        # summarizer gets a compact context; extractor/labeler get the sectioned narrative
        sum_text = (built["short"] + "\n\n" + built["text"])[:900]
        ext_text = built["text"]
        trimmed.append((cid, sum_text, ext_text, built["short"]))
        print(f"   Extracted: {cid}")
    
    all_results = []
    
    # 2. Summaries in one shot (use short text) - SKIP if DO_SUMMARY is False
    if DO_SUMMARY:
        print(f"\n2. Running Summarizer (Llama-3.2-3B-Instruct) on all cases...")
        from guardian_llm.summarizer import summarize, release as release_sum
        sum_results = {}
        for cid, sum_text, _, short_fallback in trimmed:
            try:
                start_time = time.time()
                summary = summarize(sum_text)
                summary_time = time.time() - start_time
                # Create short summary from bullet points
                short = ' '.join(
                    line.lstrip("•*- ").strip()
                    for line in (summary or "").splitlines()
                    if line.strip().startswith(("•","-","*"))
                )[:220] or short_fallback
                
                sum_results[cid] = {
                    "text": summary,
                    "short_summary": short,
                    "processing_time": summary_time
                }
                print(f"   {cid}: {summary_time:.2f}s")
            except Exception as e:
                print(f"   {cid}: FAILED - {e}")
                sum_results[cid] = {"error": str(e)}
        release_sum()
    else:
        print(f"\n2. Skipping Summarizer (DO_SUMMARY=False)")
        sum_results = {cid: {"text": "Summary skipped for speed", "processing_time": 0.0} for cid, _, _ in trimmed}
    
    # 3. Minimal "extraction" (no LLM) – build tiny schema deterministically
    print(f"\n3. Building minimal entities (no LLM extractor)...")
    from guardian_llm.extractor import minimal_entities_from_case
    ext_results = {}
    for i, (cid, _, _ext_text, _short) in enumerate(trimmed):
        m = minimal_entities_from_case(cases[i])
        # count non-null top-levels + nested location keys
        non_null = sum(1 for v in m.values() if v not in (None, "", [], {}))
        loc_non_null = sum(1 for v in (m.get("location") or {}).values() if v not in (None, "", [], {}))
        ext_results[cid] = {
            "data": m,
            "processing_time": 0.0,
            "field_count": non_null + loc_non_null
        }
        print(f"   {cid}: age={m['age']} gender={m['gender']} "
              f"loc={m['location']['city']}, {m['location']['county']}, {m['location']['state']} "
              f"lat/lon={m['lat']},{m['lon']} date_reported={bool(m['date_reported'])}")
    
    # LLM JSON extraction (Qwen) in batch — fills richer entities
    print(f"\n3b. Running LLM JSON extractor (Qwen2.5-3B-Instruct) on all cases...")
    llm_ext_results = {}
    try:
        # Prefer batch if available; else fall back to per-item
        texts_for_ext = [ext_text for (_cid, _sum, ext_text, _short) in trimmed]

        # Build per-case hints 
        hints_list = [make_hints(c) for c in cases]

        start_ex = time.time()
        if batch_extract_json:
            extracted = batch_extract_json(texts_for_ext, hints_list=hints_list)
        else:
            extracted = []
            for t, h in zip(texts_for_ext, hints_list):
                try:
                    extracted.append(extract_json(t, hints=h))
                except Exception:
                    extracted.append({})

        # Backfill with deterministic extras and count fields
        for (cid, _sum, _ext_t, _short), case, data in zip(trimmed, cases, extracted):
            merged = backfill_entities(data or {}, case)
            field_count = sum(1 for v in merged.values() if v not in (None, "", [], {}))
            llm_ext_results[cid] = {
                "data": merged,
                "processing_time": (time.time() - start_ex) / max(1, len(extracted)),
                "field_count": field_count
            }
        # Release extractor model memory
        release_ext()
        print(f"   Extracted entities for {len(llm_ext_results)} cases")
    except Exception as e:
        print(f"   LLM extractor failed: {e}")
        # Fall back: wrap the minimal entities already built
        for cid, payload in ext_results.items():
            llm_ext_results[cid] = payload

    # 4. Run weak labeler in batch mode
    print(f"\n4. Running Weak Labeler (Qwen2.5-3B-Instruct) in batch mode...")
    try:
        from guardian_llm.weak_labeler import load_weak_labeler, weak_label_batch
        
        # Load configuration
        try:
            with open("guardian.config.json", "r") as f:
                config = json.load(f)
            wl_model_id = config["models"]["weak_labeler"]
            device_map = config.get("inference", {}).get("device_map", "auto")
        except:
            wl_model_id = "C:/Users/N0Cir/CS698/Guardian/models/Qwen2.5-3B-Instruct"
            device_map = "auto"
        
        # Load the weak labeler model once
        wl_pipe = load_weak_labeler(wl_model_id, device_map=device_map)
        
        # Prepare texts for labeling (use summary text preferred, fallback to narrative)
        case_ids = []
        texts_for_lbl = []
        for case_id, summary_text, ext_text, short_fallback in trimmed:
            case_ids.append(case_id)
            txt = (summary_text or short_fallback or ext_text or "").strip()
            texts_for_lbl.append(txt)
        
        # Batch size from config  (defaults to 16)
        bs = config.get("batching", {}).get("weak_labeler_bs", 16)
        
        # Run labeling in batches and collect results
        lbl_results = {}
        for i in range(0, len(texts_for_lbl), bs):
            chunk_ids = case_ids[i:i+bs]
            chunk_txt = texts_for_lbl[i:i+bs]
            labels = weak_label_batch(wl_pipe, chunk_txt)
            for cid, lab in zip(chunk_ids, labels):
                lbl_results[cid] = {
                    "data": {
                        "movement": lab["movement"], 
                        "risk": lab["risk"]
                    }, 
                    "processing_time": lab.get("time", 0.0)
                }
            print(f"   Processed batch {i//bs + 1}: {len(chunk_ids)} cases")
        
        print(f"   Completed weak labeling for {len(lbl_results)} cases")
        
    except Exception as e:
        print(f"   Weak Labeler failed: {e}")
        # Fallback to default values
        lbl_results = {cid: {"data": {"movement": "Unknown", "risk": "Unknown"}, "processing_time": 0.0}
                       for cid, *_ in trimmed}
    
    # 5. Assemble final results
    print(f"\n5. Assembling final results...")
    for i, (case_id, sum_text, ext_text, short_fallback) in enumerate(trimmed):
        result = {
            "case_id": case_id,
            "narrative": ext_text,
            "narrative_short": short_fallback,   # always present
            "llm_results": {
                "summary": sum_results.get(case_id, {"error": "Not processed"}),
                "entities": llm_ext_results.get(case_id, {"error": "Not processed"}),
                "labels": lbl_results.get(case_id, {"error": "Not processed"})
            }
        }
        all_results.append(result)
    
    return all_results

def run_llm_analysis(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all LLM models on a single case (legacy method).
    
    Processes a single case through all available LLM models:
    - Summarizer (Llama-3.2-3B-Instruct) for case summarization
    - JSON Extractor (Qwen2.5-3B-Instruct) for entity extraction
    - Movement Classifier (Qwen2.5-3B-Instruct) for movement patterns
    - Risk Assessor (Qwen2.5-3B-Instruct) for risk evaluation
    - Combined Labeler (Qwen2.5-3B-Instruct) for comprehensive labeling
    
    Note: This method is slower than stage-by-stage processing due to
    model reload overhead. Use run_llm_analysis_stage_by_stage for batch processing.
    
    Args:
        case (Dict[str, Any]): Case data dictionary
        
    Returns:
        Dict[str, Any]: Results from all LLM models with structure:
                       {
                           "case_id": str,
                           "narrative": str,
                           "narrative_short": str,
                           "llm_results": {
                               "summary": {...},
                               "entities": {...},
                               "labels": {...}
                           }
                       }
                       
    Examples:
        >>> case = {"case_id": "GRD-001", "demographic": {"age_years": 15}}
        >>> result = run_llm_analysis(case)
        >>> result["case_id"]
        'GRD-001'
    """
    case_id = case.get("case_id", "Unknown")
    print(f"\n{'='*60}")
    print(f"Processing Case: {case_id}")
    print(f"{'='*60}")
    
    # Extract narrative
    built = build_narrative(case)
    narrative = built["text"]
    print(f"\nCase Narrative:")
    print(f"{'-'*40}")
    print(narrative)
    print(f"{'-'*40}")
    
    results = {
        "case_id": case_id,
        "narrative": narrative,
        "narrative_short": built["short"],  # Add short summary
        "llm_results": {}
    }
    
    try:
        # 1. Summarization
        print(f"\n Running Summarizer (Llama-3.2-3B-Instruct)...")
        start_time = time.time()
        summary = summarize(narrative)
        summary_time = time.time() - start_time
        results["llm_results"]["summary"] = {
            "text": summary,
            "processing_time": summary_time
        }
        print(f" Summary ({summary_time:.2f}s):")
        print(summary)
        
        # Release summarizer model
        release_sum()
        
    except Exception as e:
        print(f" Summarizer failed: {e}")
        results["llm_results"]["summary"] = {"error": str(e)}
    
    try:
        # 2. JSON Extraction
        print(f"\n Running JSON Extractor (Qwen2.5-3B-Instruct)...")
        start_time = time.time()
        json_data = extract_json(narrative)
        json_data = backfill_entities(json_data, case)  # backfill missing fields
        extractor_time = time.time() - start_time
        # Count non-null fields as "entities"
        total_entities = sum(1 for v in json_data.values() if v is not None and v != [] and v != {})
        results["llm_results"]["entities"] = {
            "data": json_data,
            "processing_time": extractor_time,
            "field_count": total_entities
        }
        print(f" JSON Data ({extractor_time:.2f}s, {total_entities} fields):")
        print(json.dumps(json_data, indent=2))
        
        # Release extractor model
        release_ext()
        
    except Exception as e:
        print(f" JSON Extractor failed: {e}")
        results["llm_results"]["entities"] = {"error": str(e)}
    
    
    try:
        # 3. Movement Classification
        print(f"\n Running Movement Classifier (Qwen2.5-3B-Instruct)...")
        start_time = time.time()
        movement = classify_movement(narrative)
        movement_time = time.time() - start_time
        results["llm_results"]["movement"] = {
            "classification": movement,
            "processing_time": movement_time
        }
        print(f" Movement Classification ({movement_time:.2f}s): {movement}")
        
        # Release weak labeler model
        release_lbl()
        
    except Exception as e:
        print(f" Movement Classifier failed: {e}")
        results["llm_results"]["movement"] = {"error": str(e)}
    
    try:
        # 4. Risk Assessment
        print(f"\n Running Risk Assessor (Qwen2.5-3B-Instruct)...")
        start_time = time.time()
        risk = assess_risk(narrative)
        risk_time = time.time() - start_time
        results["llm_results"]["risk"] = {
            "assessment": risk,
            "processing_time": risk_time
        }
        print(f" Risk Assessment ({risk_time:.2f}s): {risk}")
        
        # Release weak labeler model
        release_lbl()
        
    except Exception as e:
        print(f" Risk Assessor failed: {e}")
        results["llm_results"]["risk"] = {"error": str(e)}
    
    try:
        # 5. Combined Labeling
        print(f"\n Running Combined Labeler (Qwen2.5-3B-Instruct)...")
        start_time = time.time()
        labels = label_case(narrative)
        label_time = time.time() - start_time
        results["llm_results"]["labels"] = {
            "data": labels,
            "processing_time": label_time
        }
        print(f" Combined Labels ({label_time:.2f}s):")
        print(json.dumps(labels, indent=2))
        
        # Release weak labeler model
        release_lbl()
        
    except Exception as e:
        print(f" Combined Labeler failed: {e}")
        results["llm_results"]["labels"] = {"error": str(e)}
    
    return results

def save_results(all_results: List[Dict[str, Any]], output_file: str = "llm_analysis_results.json"):
    """
    Save all analysis results to a JSON file.
    
    Writes the complete analysis results to a JSON file with proper formatting.
    Handles encoding issues and provides feedback on successful save operations.
    
    Args:
        all_results (List[Dict[str, Any]]): All analysis results to save
        output_file (str): Output file path (default: "llm_analysis_results.json")
        
    Examples:
        >>> results = [{"case_id": "GRD-001", "llm_results": {...}}]
        >>> save_results(results, "output.json")
        Results saved to: output.json
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n Results saved to: {output_file}")
    except Exception as e:
        print(f" Failed to save results: {e}")

def print_summary(all_results: List[Dict[str, Any]]):
    """
    Print comprehensive summary of all analysis results.
    
    Displays processing statistics, success rates, and distribution analysis
    for movement classifications and risk assessments. Provides insights
    into data quality and processing effectiveness.
    
    Args:
        all_results (List[Dict[str, Any]]): All analysis results to summarize
        
    Examples:
        >>> results = [{"case_id": "GRD-001", "llm_results": {...}}]
        >>> print_summary(results)
        ANALYSIS SUMMARY
        ================
        Total cases processed: 1
        Successful summaries: 1/1
        ...
    """
    print(f"\n{'='*80}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total cases processed: {len(all_results)}")
    
    # Count successful analyses
    successful_summaries = sum(1 for r in all_results if "summary" in r["llm_results"] and "error" not in r["llm_results"]["summary"])
    successful_entities = sum(1 for r in all_results if "entities" in r["llm_results"] and "error" not in r["llm_results"]["entities"])
    
    # Fix movement and risk counting
    valid_moves = {"Stationary","Local","Regional","Interstate","International","Unknown"}
    valid_risks = {"Low","Medium","High","Critical","Unknown"}
    
    successful_movements = sum(1 for r in all_results if "labels" in r["llm_results"] and "error" not in r["llm_results"]["labels"] and str(r["llm_results"]["labels"].get("data", {}).get("movement", "")) in valid_moves)
    successful_risks = sum(1 for r in all_results if "labels" in r["llm_results"] and "error" not in r["llm_results"]["labels"] and str(r["llm_results"]["labels"].get("data", {}).get("risk", "")) in valid_risks)
    
    print(f"Successful summaries: {successful_summaries}/{len(all_results)}")
    print(f"Successful entity extractions: {successful_entities}/{len(all_results)}")
    print(f"Successful movement classifications: {successful_movements}/{len(all_results)}")
    print(f"Successful risk assessments: {successful_risks}/{len(all_results)}")
    
    # Show movement and risk distributions
    movements = {}
    risks = {}
    
    for result in all_results:
        if "labels" in result["llm_results"] and "error" not in result["llm_results"]["labels"]:
            labels = result["llm_results"]["labels"].get("data", {})
            movement = str(labels.get("movement", ""))
            risk = str(labels.get("risk", ""))
            
            if movement in valid_moves:
                movements[movement] = movements.get(movement, 0) + 1
            if risk in valid_risks:
                risks[risk] = risks.get(risk, 0) + 1
    
    if movements:
        print(f"\nMovement Classifications:")
        for movement, count in sorted(movements.items()):
            print(f"  {movement}: {count}")
    
    if risks:
        print(f"\nRisk Assessments:")
        for risk, count in sorted(risks.items()):
            print(f"  {risk}: {count}")

def run_minimal_pipeline(case_files_dir="data/synthetic_cases", out_dir="eda_out"):
    """
    Run minimal EDA pipeline.
    
    Processes synthetic case files through a deterministic pipeline without
    LLM dependencies. Creates minimal records, validates data quality,
    and generates EDA outputs for analysis and visualization.
    
    Output files created:
    - eda_cases_min.jsonl: Minimal case records in JSONL format
    - eda_counts.json: Frequency counts and coverage statistics
    - validation_report.json: Data quality validation results
    
    Args:
        case_files_dir (str): Directory containing synthetic case JSON files
        out_dir (str): Output directory for EDA results
        
    Examples:
        >>> run_minimal_pipeline("data/synthetic_cases", "eda_out")
        Wrote:
          - eda_out/eda_cases_min.jsonl
          - eda_out/eda_counts.json
          - eda_out/validation_report.json
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    gaz_lut = _GAZ  # Use the unified gazetteer

    # 1) load cases
    cases = []
    for p in sorted(Path(case_files_dir).glob("GRD-*.json")):
        with p.open("r", encoding="utf-8") as f:
            cases.append(json.load(f))

    # 2) build minimal records
    min_records, validations = [], []
    for c in cases:
        r = to_min_record(c, gaz_lut)
        issues = validate_min_record(r)
        min_records.append(r)
        validations.append({"case_id": r["case_id"], "issues": issues})

    # 3) write products
    # 3a) JSONL of minimal records (for mapping, KDE, etc.)
    jsonl_path = outp / "eda_cases_min.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in min_records:
            f.write(json.dumps(r, indent=2, ensure_ascii=False) + "\n")

    # 3b) counts for quick EDA
    counts_path = outp / "eda_counts.json"
    counts = eda_counts(min_records)
    with counts_path.open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    # 3c) validation report (only errors by default)
    bad = [v for v in validations if v["issues"]]
    val_path = outp / "validation_report.json"
    with val_path.open("w", encoding="utf-8") as f:
        json.dump({"total": len(validations), "failing": len(bad), "details": bad}, f, indent=2)

    print(f"\nWrote:\n  - {jsonl_path}\n  - {counts_path}\n  - {val_path}")
    # simple coverage print
    n = len(min_records)
    if n == 0:
        print(f"\nNo cases found in {case_files_dir}")
        return
    
    ok_age = sum(1 for r in min_records if r["age"] is not None)
    ok_gen = sum(1 for r in min_records if r["gender"] in ("M","F"))
    ok_cty = sum(1 for r in min_records if r.get('county'))
    ok_ll  = sum(1 for r in min_records if (r.get('lat') is not None and r.get('lon') is not None))
    print(f"\nCoverage over {n} cases:")
    print(f"  age:     {ok_age/n:.0%}")
    print(f"  gender:  {ok_gen/n:.0%}")
    print(f"  county:  {ok_cty/n:.0%}")
    print(f"  lat/lon: {ok_ll/n:.0%}")

def main():
    """
    Main function to run Guardian analysis pipeline.
    
    Entry point for the Guardian LLM analysis script. Determines processing
    mode based on phase configuration and runs appropriate pipeline:
    - Minimal mode: Deterministic EDA pipeline 
    - Full mode: Complete LLM processing with all models
    
    Handles both stage-by-stage and case-by-case processing with fallback
    mechanisms for error handling and performance optimization.
    
    Examples:
        >>> main()  # Run with current phase configuration
        Guardian Minimal EDA Pipeline
        ==================================================
        Processing cases...
    """
    if PHASE_MINIMAL:
        print(" Guardian Minimal EDA Pipeline ")
        print("=" * 50)
        run_minimal_pipeline(case_files_dir="data/synthetic_cases", out_dir="eda_out")
    else:
        print(" Guardian LLM Analysis Script")
        print("=" * 50)
        
        # Load synthetic cases
        print(" Loading synthetic cases...")
        cases = load_synthetic_cases()
        
        if not cases:
            print(" No cases found! Exiting.")
            return
        
        print(f" Loaded {len(cases)} cases")
        
        # Use stage-by-stage processing for better performance
        total_start_time = time.time()
        
        try:
            all_results = run_llm_analysis_stage_by_stage(cases)
        except Exception as e:
            print(f" Stage-by-stage processing failed: {e}")
            print(" Falling back to case-by-case processing...")
            traceback.print_exc()
            
            # Fallback to case-by-case processing
            all_results = []
            for i, case in enumerate(cases, 1):
                print(f"\n Processing case {i}/{len(cases)}")
                try:
                    result = run_llm_analysis(case)
                    all_results.append(result)
                except Exception as e:
                    print(f" Failed to process case {case.get('case_id', 'Unknown')}: {e}")
                    traceback.print_exc()
                    continue
        
        total_time = time.time() - total_start_time
        
        # Save results
        save_results(all_results)
        
        # --- quick post-processing for EDA ---
        try:
            print("\n Post-processing for EDA (validate, geocode, counts)...")
            # Skip post-processing since postprocess_cases module is not available
            print("  Skipping post-processing (postprocess_cases module not found)")
        except Exception as e:
            print(f" Post-processing step failed: {e}")
        # --- quick post-processing for EDA ---
        
        # Print summary
        print_summary(all_results)
        
        print(f"\n  Total processing time: {total_time:.2f} seconds")
        print(f" Average time per case: {total_time/len(cases):.2f} seconds")
        
        # Final cleanup - ensure all models are released
        print("\n Final model cleanup...")
        release_sum()
        release_ext()
        release_lbl()
        
        print("\n Analysis complete!")

def run_reasoned_sidecars(inp, out_reasoned, do_summary=False, fallback_extractor=False):
    """
    Run reasoned sidecars for LLM enhancement.
    
    Processes minimal case records through LLM models to enhance data quality.
    Can perform summarization and entity extraction to fill missing fields.
    Used for improving data completeness in the minimal pipeline.
    
    Args:
        inp (str): Input file path (JSONL format)
        out_reasoned (str): Output file path for enhanced records
        do_summary (bool): Whether to generate case summaries
        fallback_extractor (bool): Whether to use fallback entity extraction
    """
    from pathlib import Path
    import json
    
    try:
        from guardian_llm.extractor import extract_json as llm_extract
        from guardian_llm.summarizer import summarize as llm_summarize
    except ImportError:
        print("[WARN] LLM modules not available, skipping reasoned sidecars")
        return
    
    Path(out_reasoned).write_text("", encoding="utf-8")
    
    with open(inp, "r", encoding="utf-8") as f_in, open(out_reasoned, "a", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            text = rec.get("raw_text") or rec.get("narrative") or ""
            
            if fallback_extractor:
                missing = [k for k in ("age", "gender", "county", "state", "lat", "lon", "date_reported")
                          if rec.get(k) in (None, "", "Unknown")]
                if missing and text:
                    try:
                        patch = llm_extract(text, fields=missing) or {}
                        for k, v in patch.items():
                            if v not in (None, "", "Unknown"):
                                rec[k] = float(v) if k in ("lat", "lon") else v
                    except Exception as e:
                        print(f"[WARN] Extractor failed for {rec.get('case_id', 'unknown')}: {e}")
            
            if do_summary and text:
                try:
                    rec["summary"] = llm_summarize(structured_case=rec, narrative=text)
                except Exception as e:
                    print(f"[WARN] Summarizer failed for {rec.get('case_id', 'unknown')}: {e}")
            
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"[DONE] Reasoned sidecars written to {out_reasoned}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoned", action="store_true", help="Run reasoned sidecars")
    parser.add_argument("--do-summary", action="store_true", help="Generate summaries")
    parser.add_argument("--fallback-extractor", action="store_true", help="Use fallback extractor")
    args = parser.parse_args()
    
    if args.reasoned:
        run_reasoned_sidecars(
            inp="eda_out/eda_cases_min.jsonl",
            out_reasoned="eda_out/eda_reasoned.jsonl",
            do_summary=args.do_summary,
            fallback_extractor=args.fallback_extractor
        )
        raise SystemExit
    
    main()
