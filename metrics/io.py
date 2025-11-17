"""Input/output utilities for metrics calculation.

Provides functions for reading JSON/JSONL files and computing geographic
distances.
"""
from __future__ import annotations

import json
import pathlib
import typing as T


def read_json_blocks(path: str | pathlib.Path) -> list[dict]:
    """Read JSON objects from newline-delimited or pretty-printed file.
    
    Handles both compact JSONL format (one JSON per line) and pretty-printed
    format with blank line separators.
    
    Args:
        path: Path to JSON/JSONL file.
        
    Returns:
        list[dict]: List of parsed JSON objects. Returns empty list if file
            doesn't exist or contains no valid JSON.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return []
    objs, buf, depth = [], [], 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() and depth == 0:
                continue
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0 and buf:
                block = "".join(buf).strip()
                try:
                    objs.append(json.loads(block))
                except Exception:
                    for maybe in block.splitlines():
                        maybe = maybe.strip()
                        if maybe:
                            try:
                                objs.append(json.loads(maybe))
                            except:
                                pass
                buf = []
    return objs


def coord_of(row: dict) -> tuple[float | None, float | None]:
    """Extract coordinates from row dictionary.
    
    Supports multiple coordinate field patterns and nested location structures.
    
    Args:
        row: Dictionary that may contain coordinate fields.
        
    Returns:
        Tuple of (lat, lon) or (None, None) if coordinates not found.
    """
    for a, b in (
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("center_lat", "center_lon"),
        ("last_seen_lat", "last_seen_lon"),
    ):
        if row.get(a) is not None and row.get(b) is not None:
            return float(row[a]), float(row[b])
    loc = row.get("location") or row.get("geom") or {}
    for a, b in (("lat", "lon"), ("latitude", "longitude")):
        if loc.get(a) is not None and loc.get(b) is not None:
            return float(loc[a]), float(loc[b])
    return None, None


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in miles.
    
    Args:
        lat1: Latitude of first point in degrees.
        lon1: Longitude of first point in degrees.
        lat2: Latitude of second point in degrees.
        lon2: Longitude of second point in degrees.
        
    Returns:
        Distance in miles between the two points.
    """
    from math import radians, sin, cos, asin, sqrt

    R = 3958.8  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * R * asin(sqrt(a))

