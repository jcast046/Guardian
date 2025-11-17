"""Entity extraction metrics for information extraction evaluation.

Evaluates precision, recall, and F1 scores for person names, vehicle
descriptions, and location strings extracted by LLM models against
ground truth GRD cases.
"""
from __future__ import annotations

import statistics as s
from datetime import datetime
from .config import load_config
from .io import read_json_blocks, haversine_miles


def _norm_entities(container: dict) -> dict:
    """Normalize entity extraction from nested container structure.
    
    Supports multiple container formats: top-level "entities", nested
    "extracted.entities", or nested "llm.entities".
    
    Args:
        container: Dictionary that may contain entities in various formats.
        
    Returns:
        Dictionary containing normalized entities, or empty dict if none found.
    """
    e = container.get("entities") or {}
    if not e and isinstance(container.get("extracted"), dict):
        e = container["extracted"].get("entities") or e
    if not e and isinstance(container.get("llm"), dict):
        e = container["llm"].get("entities") or e
    return e or {}


def _fold_vehicle(v: dict) -> str:
    """Normalize vehicle dictionary to comparable string.
    
    Combines make, model, and color fields into normalized string format.
    Handles make_model field if present, otherwise combines individual fields.
    
    Args:
        v: Vehicle dictionary containing make, model, color, or make_model.
        
    Returns:
        Normalized lowercase vehicle string.
    """
    if v.get("make_model"):
        return str(v["make_model"])
    make = str(v.get("make", "")).strip()
    model = str(v.get("model", "")).strip()
    color = str(v.get("color", "")).strip()
    # Normalize common variations
    result = f"{make} {model} {color}".strip()
    # Remove extra spaces and normalize hyphens
    result = " ".join(result.split())
    result = result.replace("F-150", "F150").replace("F - 150", "F150")
    return result.lower()


def _normalize_state(state: str | None) -> str:
    """Normalize state names to standard abbreviations.
    
    Args:
        state: State name or abbreviation string.
        
    Returns:
        Normalized lowercase state abbreviation.
    """
    if not state:
        return ""
    state_str = str(state).strip()
    # Normalize common variations
    state_map = {
        "virginia": "va",
        "va": "va",
        "Virginia": "va",
        "VA": "va",
    }
    return state_map.get(state_str, state_str.lower())


def _normalize_person_name(name: str | None) -> str:
    """Normalize person name for comparison.
    
    Args:
        name: Person name string.
        
    Returns:
        Normalized lowercase name string, or empty string if None.
    """
    if not name:
        return ""
    return str(name).strip().lower()


def _normalize_location(city: str, county: str, state: str) -> str:
    """Normalize location components into comparable string.
    
    Combines city, county, and state into normalized format.
    
    Args:
        city: City name string.
        county: County name string.
        state: State name or abbreviation string.
        
    Returns:
        Normalized lowercase location string.
    """
    city_str = str(city or "").strip().lower()
    county_str = str(county or "").strip().lower()
    state_str = _normalize_state(state)
    parts = [p for p in [city_str, county_str, state_str] if p]
    return " ".join(parts)


def _extract_persons_from_pred(pred_data: dict) -> list[str]:
    """Extract person names from predicted LLM data.
    
    Extracts main person name and persons of interest from LLM predictions.
    
    Args:
        pred_data: Dictionary containing LLM prediction data.
        
    Returns:
        List of normalized person name strings.
    """
    persons = []
    if not isinstance(pred_data, dict):
        return persons
    try:
        # Main person name
        name = pred_data.get("name")
        if name:
            normalized = _normalize_person_name(name)
            if normalized:
                persons.append(normalized)
        # Persons of interest
        poi_list = pred_data.get("persons_of_interest", [])
        if isinstance(poi_list, list):
            for poi in poi_list:
                if isinstance(poi, dict):
                    poi_name = poi.get("name")
                    if poi_name:
                        normalized = _normalize_person_name(poi_name)
                        if normalized:
                            persons.append(normalized)
    except (AttributeError, TypeError, KeyError):
        pass
    return persons


def _extract_persons_from_gold(gold_case: dict) -> list[str]:
    """Extract person names from ground truth GRD case.
    
    Extracts main person from demographic and persons of interest from
    narrative_osint sections.
    
    Args:
        gold_case: Dictionary containing ground truth case data.
        
    Returns:
        List of normalized person name strings.
    """
    persons = []
    if not isinstance(gold_case, dict):
        return persons
    try:
        # Main person from demographic
        demo = gold_case.get("demographic", {})
        if isinstance(demo, dict):
            name = demo.get("name")
            if name:
                normalized = _normalize_person_name(name)
                if normalized:
                    persons.append(normalized)
        # Persons of interest
        narrative_osint = gold_case.get("narrative_osint", {})
        if isinstance(narrative_osint, dict):
            poi_list = narrative_osint.get("persons_of_interest", [])
            if isinstance(poi_list, list):
                for poi in poi_list:
                    if isinstance(poi, dict):
                        poi_name = poi.get("name")
                        if poi_name:
                            normalized = _normalize_person_name(poi_name)
                            if normalized:
                                persons.append(normalized)
    except (AttributeError, TypeError, KeyError):
        pass
    return persons


def _extract_vehicles_from_pred(pred_data: dict) -> list[str]:
    """Extract vehicles from predicted LLM data.
    
    Extracts vehicle information from persons_of_interest in LLM predictions.
    
    Args:
        pred_data: Dictionary containing LLM prediction data.
        
    Returns:
        List of normalized vehicle strings.
    """
    vehicles = []
    if not isinstance(pred_data, dict):
        return vehicles
    try:
        poi_list = pred_data.get("persons_of_interest", [])
        if isinstance(poi_list, list):
            for poi in poi_list:
                if isinstance(poi, dict):
                    veh = poi.get("vehicle")
                    if veh:
                        if isinstance(veh, dict):
                            normalized = _fold_vehicle(veh)
                            if normalized:
                                vehicles.append(normalized)
                        else:
                            normalized = str(veh).strip().lower()
                            if normalized:
                                vehicles.append(normalized)
    except (AttributeError, TypeError, KeyError):
        pass
    return vehicles


def _extract_vehicles_from_gold(gold_case: dict) -> list[str]:
    """Extract vehicles from ground truth GRD case.
    
    Extracts vehicle information from persons_of_interest in narrative_osint.
    
    Args:
        gold_case: Dictionary containing ground truth case data.
        
    Returns:
        List of normalized vehicle strings.
    """
    vehicles = []
    if not isinstance(gold_case, dict):
        return vehicles
    try:
        narrative_osint = gold_case.get("narrative_osint", {})
        if isinstance(narrative_osint, dict):
            poi_list = narrative_osint.get("persons_of_interest", [])
            if isinstance(poi_list, list):
                for poi in poi_list:
                    if isinstance(poi, dict):
                        veh = poi.get("vehicle")
                        if veh:
                            if isinstance(veh, dict):
                                normalized = _fold_vehicle(veh)
                                if normalized:
                                    vehicles.append(normalized)
                            else:
                                normalized = str(veh).strip().lower()
                                if normalized:
                                    vehicles.append(normalized)
    except (AttributeError, TypeError, KeyError):
        pass
    return vehicles


def _extract_locations_from_pred(pred_data: dict) -> list[str]:
    """Extract locations from predicted LLM data.
    
    Extracts location string from location field in LLM predictions.
    
    Args:
        pred_data: Dictionary containing LLM prediction data.
        
    Returns:
        List of normalized location strings.
    """
    locations = []
    if not isinstance(pred_data, dict):
        return locations
    try:
        loc = pred_data.get("location", {})
        if isinstance(loc, dict):
            city = loc.get("city", "") or ""
            county = loc.get("county", "") or ""
            state = loc.get("state", "") or ""
            loc_str = _normalize_location(city, county, state)
            if loc_str:
                locations.append(loc_str)
    except (AttributeError, TypeError, KeyError):
        pass
    return locations


def _extract_locations_from_gold(gold_case: dict) -> list[str]:
    """Extract locations from ground truth GRD case.
    
    Extracts location string from spatial.last_seen fields.
    
    Args:
        gold_case: Dictionary containing ground truth case data.
        
    Returns:
        List of normalized location strings.
    """
    locations = []
    if not isinstance(gold_case, dict):
        return locations
    try:
        spatial = gold_case.get("spatial", {})
        if isinstance(spatial, dict):
            city = spatial.get("last_seen_city", "") or ""
            county = spatial.get("last_seen_county", "") or ""
            state = spatial.get("last_seen_state", "") or ""
            loc_str = _normalize_location(city, county, state)
            if loc_str:
                locations.append(loc_str)
    except (AttributeError, TypeError, KeyError):
        pass
    return locations


def _calculate_precision_recall_f1(pred_entities: list[str], gold_entities: list[str]) -> dict[str, float | None]:
    """Calculate Precision, Recall, and F1-Score for entity lists.
    
    Args:
        pred_entities: List of predicted entity strings (normalized)
        gold_entities: List of ground truth entity strings (normalized)
        
    Returns:
        Dictionary with 'precision', 'recall', and 'f1' scores, or None if not applicable
    """
    # Convert to sets for comparison
    pred_set = set(pred_entities) if pred_entities else set()
    gold_set = set(gold_entities) if gold_entities else set()
    
    # Remove empty strings
    pred_set = {e for e in pred_set if e}
    gold_set = {e for e in gold_set if e}
    
    # Calculate TP, FP, FN
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    # Calculate Precision
    if tp + fp == 0:
        precision = None  # No predictions made
    else:
        precision = tp / (tp + fp)
    
    # Calculate Recall
    if tp + fn == 0:
        recall = None  # No ground truth entities
    else:
        recall = tp / (tp + fn)
    
    # Calculate F1-Score
    if precision is None or recall is None:
        f1 = None
    elif precision == 0.0 and recall == 0.0:
        f1 = 0.0
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def _f1_sets(pred_list: list[str], gold_list: list[str]) -> float:
    """Calculate F1 score between two string lists using set intersection.
    
    Args:
        pred_list: List of predicted strings.
        gold_list: List of ground truth strings.
        
    Returns:
        F1 score (harmonic mean of precision and recall).
    """
    pset = set(str(x or "").strip().lower() for x in pred_list if x)
    gset = set(str(x or "").strip().lower() for x in gold_list if x)
    tp = len(pset & gset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return f


def calc_extractor_metrics(cfg: dict | None = None) -> dict:
    """Calculate entity extraction metrics for information extraction.
    
    Evaluates LLM-extracted entities (persons, vehicles, locations) against
    ground truth GRD cases. Computes precision, recall, and F1 scores for
    each entity type, plus time/geocode accuracy metrics.
    
    Args:
        cfg: Optional configuration dictionary. If None, loads default config.
        
    Returns:
        dict: Dictionary containing timestamp, stage, metrics (precision/recall/F1
            for persons/vehicles/locations, time MAE, geocode MAE, county accuracy),
            and warnings list.
    """
    cfg = cfg or load_config()
    out = {
        "timestamp": datetime.now().isoformat(),
        "stage": "extractor",
        "metrics": {},
        "warnings": [],
    }

    # Load LLM results (predictions)
    import json
    import pathlib

    llm_results_path = pathlib.Path(cfg["paths"]["llm_results"])
    if not llm_results_path.exists():
        out["warnings"].append(f"LLM results file not found: {llm_results_path}")
        out["metrics"].update(
            {
                "persons": {"precision": None, "recall": None, "f1": None},
                "vehicles": {"precision": None, "recall": None, "f1": None},
                "locations": {"precision": None, "recall": None, "f1": None},
                "macro_f1_persons": None,
                "macro_f1_vehicles": None,
                "macro_f1_locations": None,
                "time_mae_hours": None,
                "geocode_mae_miles": None,
                "county_accuracy": None,
            }
        )
        return out

    # Load LLM results (array of objects)
    with open(llm_results_path, "r", encoding="utf-8") as f:
        llm_results = json.load(f)

    # Load GRD cases for ground truth
    synth_dir = pathlib.Path(cfg["paths"]["synthetic_cases"])
    gold_cases = {}
    if synth_dir.exists():
        for grd_file in synth_dir.glob("GRD-*.json"):
            try:
                with open(grd_file, "r", encoding="utf-8") as f:
                    case = json.load(f)
                    cid = case.get("case_id")
                    if cid:
                        gold_cases[cid] = case
            except Exception as e:
                out["warnings"].append(f"Error loading {grd_file.name}: {e}")

    if not gold_cases:
        out["warnings"].append("No GRD case files found for ground truth")
        out["metrics"].update(
            {
                "persons": {"precision": None, "recall": None, "f1": None},
                "vehicles": {"precision": None, "recall": None, "f1": None},
                "locations": {"precision": None, "recall": None, "f1": None},
                "macro_f1_persons": None,
                "macro_f1_vehicles": None,
                "macro_f1_locations": None,
                "time_mae_hours": None,
                "geocode_mae_miles": None,
                "county_accuracy": None,
            }
        )
        return out

    # Extract predictions from LLM results
    pred = {}
    for result in llm_results:
        cid = result.get("case_id")
        if not cid:
            continue
        llm_entities = result.get("llm_results", {}).get("entities", {})
        if "data" in llm_entities:
            pred[cid] = llm_entities["data"]

    # Collect entities across all cases for Precision/Recall/F1 calculation
    all_pred_persons = []
    all_gold_persons = []
    all_pred_vehicles = []
    all_gold_vehicles = []
    all_pred_locations = []
    all_gold_locations = []

    # Legacy metrics (for backward compatibility)
    f1p, f1v, f1l, t_mae_h, geo_miles, c_acc = [], [], [], [], [], []

    for cid, g_case in gold_cases.items():
        p = pred.get(cid, {})
        if not p:
            continue

        try:
            # Extract ground truth from GRD case
            demo = g_case.get("demographic", {})
            spatial = g_case.get("spatial", {})
            temporal = g_case.get("temporal", {})

            # Extract entities for Precision/Recall/F1 calculation
            pred_persons = _extract_persons_from_pred(p)
            gold_persons = _extract_persons_from_gold(g_case)
            pred_vehicles = _extract_vehicles_from_pred(p)
            gold_vehicles = _extract_vehicles_from_gold(g_case)
            pred_locations = _extract_locations_from_pred(p)
            gold_locations = _extract_locations_from_gold(g_case)

            # Collect entities for aggregate metrics
            all_pred_persons.extend(pred_persons)
            all_gold_persons.extend(gold_persons)
            all_pred_vehicles.extend(pred_vehicles)
            all_gold_vehicles.extend(gold_vehicles)
            all_pred_locations.extend(pred_locations)
            all_gold_locations.extend(gold_locations)

            g_name = demo.get("name") or ""
            p_name = p.get("name") or ""
            if g_name or p_name:
                f1p.append(_f1_sets([p_name], [g_name]))

            g_vehicles = []
            poi_list = g_case.get("narrative_osint", {}).get("persons_of_interest", [])
            for poi in poi_list:
                veh = poi.get("vehicle")
                if veh:
                    if isinstance(veh, dict):
                        g_vehicles.append(_fold_vehicle(veh))
                    else:
                        g_vehicles.append(str(veh))

            p_vehicles = []
            poi_data = p.get("persons_of_interest", [])
            for poi in poi_data if isinstance(poi_data, list) else []:
                veh = poi.get("vehicle")
                if veh:
                    if isinstance(veh, dict):
                        p_vehicles.append(_fold_vehicle(veh))
                    else:
                        p_vehicles.append(str(veh))

            if g_vehicles or p_vehicles:
                f1v.append(_f1_sets(p_vehicles, g_vehicles))

            g_city = spatial.get("last_seen_city") or ""
            g_county = spatial.get("last_seen_county") or ""
            g_state = spatial.get("last_seen_state") or ""
            g_loc_str = f"{g_city} {g_county} {g_state}".strip()

            p_loc = p.get("location", {})
            p_city = p_loc.get("city") or "" if isinstance(p_loc, dict) else ""
            p_county = p_loc.get("county") or "" if isinstance(p_loc, dict) else ""
            p_state = p_loc.get("state") or "" if isinstance(p_loc, dict) else ""
            p_loc_str = f"{p_city} {p_county} {p_state}".strip()

            if g_loc_str or p_loc_str:
                f1l.append(_f1_sets([p_loc_str], [g_loc_str]))

            gt_ts = temporal.get("reported_missing_ts")
            pt_ts = p.get("date_reported")
            if gt_ts and pt_ts:

                def parse_any(x):
                    if isinstance(x, (int, float)):
                        return float(x)
                    return datetime.fromisoformat(
                        str(x).replace("Z", "+00:00")
                    ).timestamp() / 3600.0

                try:
                    t_mae_h.append(abs(parse_any(pt_ts) - parse_any(gt_ts)))
                except:
                    pass

            g_lat = spatial.get("last_seen_lat")
            g_lon = spatial.get("last_seen_lon")
            p_lat = p.get("lat")
            p_lon = p.get("lon")
            if None not in (g_lat, g_lon, p_lat, p_lon):
                geo_miles.append(
                    haversine_miles(
                        float(g_lat), float(g_lon), float(p_lat), float(p_lon)
                    )
                )

            gc = g_county
            pc = p_county
            if gc and pc:
                c_acc.append(1 if str(gc).lower() == str(pc).lower() else 0)
        except Exception as e:
            out["warnings"].append(f"Error processing case {cid}: {e}")
            continue

    # Calculate Precision, Recall, and F1-Score for each entity type
    persons_metrics = _calculate_precision_recall_f1(all_pred_persons, all_gold_persons)
    vehicles_metrics = _calculate_precision_recall_f1(all_pred_vehicles, all_gold_vehicles)
    locations_metrics = _calculate_precision_recall_f1(all_pred_locations, all_gold_locations)

    def mean(x):
        """Calculate mean of numeric list or return None if empty."""
        return float(s.fmean(x)) if x else None
    out["metrics"] = {
        # New Precision/Recall/F1 metrics
        "persons": {
            "precision": persons_metrics["precision"],
            "recall": persons_metrics["recall"],
            "f1": persons_metrics["f1"]
        },
        "vehicles": {
            "precision": vehicles_metrics["precision"],
            "recall": vehicles_metrics["recall"],
            "f1": vehicles_metrics["f1"]
        },
        "locations": {
            "precision": locations_metrics["precision"],
            "recall": locations_metrics["recall"],
            "f1": locations_metrics["f1"]
        },
        # Legacy metrics (for backward compatibility)
        "macro_f1_persons": mean(f1p),
        "macro_f1_vehicles": mean(f1v),
        "macro_f1_locations": mean(f1l),
        "time_mae_hours": mean(t_mae_h),
        "geocode_mae_miles": mean(geo_miles),
        "county_accuracy": mean(c_acc),
    }
    if not any([f1p, f1v, f1l]) and not any([all_pred_persons, all_pred_vehicles,
                                              all_pred_locations]):
        out["warnings"].append("No comparable entities found in preds/gold.")
    return out

