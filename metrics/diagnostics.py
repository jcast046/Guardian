"""Diagnostic metrics for pipeline data availability and validation.

Reports data file counts, ID overlaps between datasets, and schema validation
information for debugging and data quality assessment.
"""
from __future__ import annotations

from datetime import datetime
from .config import load_config
from .io import read_json_blocks


def calc_diagnostics(cfg: dict | None = None) -> dict:
    """Calculate diagnostic metrics for pipeline data availability.
    
    Reports counts, ID overlaps between datasets, and schema information
    for debugging and data quality assessment.
    
    Args:
        cfg: Optional configuration dictionary. If None, loads default config.
        
    Returns:
        dict: Dictionary containing timestamp, stage, and diagnostics
            (counts, ID overlaps, field presence information).
    """
    cfg = cfg or load_config()

    eda = read_json_blocks(cfg["paths"]["eda_min"])
    gold = read_json_blocks(cfg["paths"]["gold_cases"])
    truth = read_json_blocks(cfg["paths"]["gold_zones"])
    zones = read_json_blocks(cfg["paths"]["zones_baseline"])

    def keys(rows: list[dict], k: str) -> list:
        """Extract keys from nested dict field.
        
        Args:
            rows: List of row dictionaries.
            k: Key to search for in rows.
            
        Returns:
            List of keys from first matching nested dictionary, or empty list.
        """
        for r in rows:
            if k in r and isinstance(r[k], dict):
                return list(r[k].keys())
        return []

    eda_ids = {r.get("case_id") for r in eda if r.get("case_id")}
    gold_ids = {r.get("case_id") for r in gold if r.get("case_id")}
    truth_ids = {r.get("case_id") for r in truth if r.get("case_id")}
    zone_ids = {r.get("case_id") for r in zones if r.get("case_id")}

    eda_sample = eda[:20] if len(eda) > 20 else eda
    eda_fields = set()
    for r in eda_sample:
        eda_fields.update(r.keys())

    out = {
        "counts": {
            "eda": len(eda),
            "gold": len(gold),
            "truth": len(truth),
            "zones": len(zones),
        },
        "id_overlap": {
            "zones∩truth": len(zone_ids & truth_ids),
            "zones∩eda": len(zone_ids & eda_ids),
            "truth∩eda": len(truth_ids & eda_ids),
            "gold∩eda": len(gold_ids & eda_ids),
        },
        "eda_fields_present": sorted(eda_fields),
        "gold_has_entities": any("entities" in r for r in gold),
        "gold_has_movement_profile": any("movement_profile" in r for r in gold),
        "eda_has_summary": any(
            "summary" in r
            or "summary_5bullets" in r
            or (isinstance(r.get("llm"), dict) and "summary" in r.get("llm", {}))
            for r in eda
        ),
        "zones_schema_hint": (
            list((zones[0].get("zones") or {}).keys())
            if zones
            and isinstance(zones[0].get("zones"), dict)
            else "list"
        ),
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "stage": "diagnostics",
        "diagnostics": out,
    }

