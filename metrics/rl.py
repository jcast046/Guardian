"""Reinforcement learning metrics for search zone evaluation.

Evaluates RL-generated search zones against ground truth using hit rates,
distance metrics, and area-based efficiency measures.
"""
from __future__ import annotations

import json
import math
import pathlib
from datetime import datetime
from .config import load_config
from .io import read_json_blocks, haversine_miles

try:
    from shapely.geometry import Point, shape
    from shapely import wkt as _wkt

    SHAPELY = True
except Exception:
    SHAPELY = False
    _wkt = None


def _zone_score(z: dict) -> float:
    """Extract zone priority score from zone dictionary.
    
    Checks multiple possible field names for zone priority/score.
    
    Args:
        z: Zone dictionary.
        
    Returns:
        Zone priority score as float, or 0.0 if not found.
    """
    for k in ("priority", "priority_llm", "score"):
        if isinstance(z.get(k), (int, float)):
            return float(z[k])
    return 0.0


def _flatten_zones(zs: dict | list) -> list:
    """Flatten nested zone structure to flat list.
    
    Handles both dict mapping window IDs to zone lists and direct zone lists.
    
    Args:
        zs: Zone structure (dict or list).
        
    Returns:
        Flat list of zone dictionaries.
    """
    if isinstance(zs, dict):
        tmp = []
        for v in zs.values():
            if isinstance(v, list):
                tmp.extend(v)
        return tmp
    return zs or []


def _contains(z: dict, lat: float, lon: float, buffer_m: float) -> bool:
    """Check if point is contained in zone with buffer tolerance.
    
    Supports multiple zone formats: polygon dict/WKT, circle with radius_m,
    or circle with radius_miles/radius_km.
    
    Args:
        z: Zone dictionary containing geometry or center/radius information.
        lat: Latitude of point to test.
        lon: Longitude of point to test.
        buffer_m: Buffer distance in meters for containment testing.
        
    Returns:
        True if point is within zone (including buffer), False otherwise.
    """
    if SHAPELY:
        pt = Point(lon, lat)
        if isinstance(z.get("polygon"), dict):
            try:
                poly = shape(z["polygon"])
                buffer_deg = buffer_m / 111000.0
                if poly.buffer(buffer_deg).contains(pt) or poly.touches(pt):
                    return True
            except Exception:
                pass
        if z.get("wkt") and _wkt:
            try:
                geom = _wkt.loads(z["wkt"])
                buffer_deg = buffer_m / 111000.0
                if geom.buffer(buffer_deg).contains(pt) or geom.touches(pt):
                    return True
            except Exception:
                pass
    c = z.get("center") or {}
    if {"lat", "lon"}.issubset(c.keys()) and z.get("radius_m") is not None:
        d_miles = haversine_miles(lat, lon, float(c["lat"]), float(c["lon"]))
        d_m = d_miles * 1609.34
        return d_m <= (float(z["radius_m"]) + buffer_m)
    if z.get("center_lat") is not None and z.get("center_lon") is not None:
        clat = float(z["center_lat"])
        clon = float(z["center_lon"])
        d_miles = haversine_miles(lat, lon, clat, clon)
        if z.get("radius_miles") is not None:
            buffer_miles = buffer_m / 1609.34
            return d_miles <= (float(z["radius_miles"]) + buffer_miles)
        if z.get("radius_km") is not None:
            radius_miles = float(z["radius_km"]) * 0.621371
            buffer_miles = buffer_m / 1609.34
            return d_miles <= (radius_miles + buffer_miles)
    return False


def _zone_center(z: dict) -> tuple[float | None, float | None]:
    """Extract zone center coordinates from zone dictionary.
    
    Supports multiple coordinate field patterns.
    
    Args:
        z: Zone dictionary.
        
    Returns:
        Tuple of (lat, lon) or (None, None) if center not found.
    """
    c = z.get("center") or {}
    if {"lat", "lon"}.issubset(c.keys()):
        return float(c["lat"]), float(c["lon"])
    if z.get("center_lat") is not None and z.get("center_lon") is not None:
        return float(z["center_lat"]), float(z["center_lon"])
    return None, None


def calc_rl_metrics(zones_type: str = "baseline", cfg: dict | None = None) -> dict:
    """Calculate reinforcement learning metrics for search zone evaluation.
    
    Evaluates RL-generated zones against ground truth using hit rates at
    different K values, mean reciprocal rank, NDCG, distance metrics, and
    area-based efficiency measures.
    
    Args:
        zones_type: Type of zones to evaluate ("baseline" or "llm").
        cfg: Optional configuration dictionary. If None, loads default config.
        
    Returns:
        dict: Dictionary containing timestamp, stage, zones_type, metrics
            (hit_at_k, MRR, NDCG, distance metrics, ASUH), and warnings list.
    """
    cfg = cfg or load_config()
    out = {
        "timestamp": datetime.now().isoformat(),
        "stage": "rl",
        "zones_type": zones_type,
        "metrics": {},
        "warnings": [],
    }
    buf_m = float(cfg.get("geo", {}).get("hit_buffer_m", 0) or 0)

    # Load ground truth from reinforcement_learning/ground_truth.json
    truth_path = pathlib.Path("reinforcement_learning/ground_truth.json")
    if truth_path.exists():
        with open(truth_path, 'r', encoding='utf-8') as f:
            truth_data = json.load(f)
        truth = {
            case_id: (float(coords["lat"]), float(coords["lon"]))
            for case_id, coords in truth_data.items()
            if "lat" in coords and "lon" in coords
        }
    else:
        truth = {}

    zones_path = (
        cfg["paths"]["zones_baseline"]
        if zones_type == "baseline"
        else cfg["paths"]["zones_llm"]
    )
    zone_rows = read_json_blocks(zones_path)

    zone_ids = {r.get("case_id") for r in zone_rows if r.get("case_id")}
    truth_ids = set(truth.keys())
    overlap = sorted(zone_ids & truth_ids)
    out["warnings"].append(
        f"RL overlap cases: zones={len(zone_ids)}, truth={len(truth_ids)}, overlap={len(overlap)}"
    )

    ks = cfg["rl"]["ks"]
    hit_at_k = {k: 0 for k in ks}
    rr, ndcg5, d_first_miles, d_top1_miles, asuh = [], [], [], [], []
    total = 0

    for row in zone_rows:
        cid = row.get("case_id")
        if cid not in truth:
            continue
        zones = _flatten_zones(row.get("zones"))
        zones = [z for z in zones if isinstance(z, dict)]
        if not zones:
            continue
        zones = sorted(zones, key=_zone_score, reverse=True)
        total += 1
        tlat, tlon = truth[cid]

        gains = []
        first_rank = None
        for i, z in enumerate(zones):
            hit = _contains(z, tlat, tlon, buf_m)
            gains.append(1 if hit else 0)
            if hit and first_rank is None:
                first_rank = i + 1

        for k in ks:
            if any(gains[:k]):
                hit_at_k[k] += 1

        rr.append(1.0 / first_rank if first_rank else 0.0)

        K = 5
        dcg = sum((g / math.log2(i + 2)) for i, g in enumerate(gains[:K]))
        ndcg5.append(dcg)

        c1 = _zone_center(zones[0])
        if c1[0] is not None:
            d_top1_miles.append(
                haversine_miles(tlat, tlon, c1[0], c1[1])
            )
        if first_rank:
            cf = _zone_center(zones[first_rank - 1])
            if cf[0] is not None:
                d_first_miles.append(
                    haversine_miles(tlat, tlon, cf[0], cf[1])
                )

        if first_rank:
            area = 0.0
            for i in range(first_rank):
                if zones[i].get("radius_m") is not None:
                    rkm = float(zones[i]["radius_m"]) / 1000.0
                    area += math.pi * rkm * rkm
                elif zones[i].get("radius_miles") is not None:
                    rkm = float(zones[i]["radius_miles"]) * 1.60934
                    area += math.pi * rkm * rkm
                elif zones[i].get("radius_km") is not None:
                    rkm = float(zones[i]["radius_km"])
                    area += math.pi * rkm * rkm
                elif zones[i].get("area_km2") is not None:
                    area += float(zones[i]["area_km2"])
            asuh.append(area)

    def _med(x: list) -> float | None:
        """Calculate median of numeric list or return None if empty."""
        import statistics as s
        return float(s.median(x)) if x else None

    out["metrics"] = {
        "geo_hit_at_k": {
            f"geo_hit_at_{k}": (hit_at_k[k] / total if total else 0.0)
            for k in ks
        },
        "mrr": (sum(rr) / len(rr)) if rr else 0.0,
        "ndcg_at_5": (sum(ndcg5) / len(ndcg5)) if ndcg5 else 0.0,
        "median_distance_to_first_hit_miles": _med(d_first_miles),
        "median_distance_top1_to_truth_miles": _med(d_top1_miles),
        "asuh_km2": (sum(asuh) / len(asuh)) if asuh else None,
    }
    return out

