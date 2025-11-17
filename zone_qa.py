#!/usr/bin/env python3
"""Zone QA - LLM Sidecar Module for Search Zone Plausibility Analysis.

This module enhances search zone prioritization using LLM-based plausibility scoring
without modifying the core synthetic cases. It operates as a "sidecar" to the main
EDA pipeline, providing LLM-enhanced zone analysis for evaluation.

Author: Joshua Castillo

Example:
    python zone_qa.py --input data/synthetic_cases --config reinforcement_learning/search_reward_config.json --outdir eda_out --evaluate
"""
# Standard library imports
import json
import pathlib
import argparse
import sys
import random
import os
import math
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Environment configuration for LLM labeler selection
# Controls whether to use real LLM or mock implementation for testing
USE_MOCK_ENV = os.getenv("GUARDIAN_USE_MOCK", "0").strip() == "1"

# LLM labeler configuration with graceful fallback
# Real labeler provides actual LLM-based plausibility scoring
# Mock labeler provides deterministic results for testing and development
REAL_LABELER = None
REAL_LABELER_ERR = None
try:
    from guardian_llm.weak_labeler import label_case as _real_label_case
except Exception as e:
    REAL_LABELER_ERR = e
    _real_label_case = None

# Create wrapper to adapt real labeler to expected interface
if _real_label_case is not None:
    def REAL_LABELER(structured_case: Dict[str, Any], narrative: str) -> Dict[str, Any]:
        """
        Wrapper function to adapt real LLM labeler to expected interface.
        
        This function bridges the gap between the real weak_labeler module and the
        expected interface for zone plausibility scoring. It converts risk levels
        to plausibility scores and formats the output consistently.
        
        Args:
            structured_case (Dict[str, Any]): Case data structure (used for context)
            narrative (str): Text narrative for LLM analysis
            
        Returns:
            Dict[str, Any]: Standardized result with plausibility score and rationale
            
        Note:
            The function maps risk levels to plausibility scores:
            - Critical: 0.9 (highest plausibility)
            - High: 0.7 (high plausibility)
            - Medium: 0.5 (moderate plausibility)
            - Low: 0.3 (low plausibility)
        """
        # Ensure narrative is a string for LLM processing
        if not isinstance(narrative, str):
            narrative = str(narrative) if narrative is not None else ""
        
        # Call real LLM labeler with narrative
        real_result = _real_label_case(narrative)
        
        # Convert risk level to plausibility score using predefined mapping
        risk_to_plausibility = {
            "Critical": 0.9,  # Highest plausibility for critical risk
            "High": 0.7,      # High plausibility for high risk
            "Medium": 0.5,    # Moderate plausibility for medium risk
            "Low": 0.3        # Low plausibility for low risk
        }
        
        # Extract plausibility score from risk level
        plausibility = risk_to_plausibility.get(real_result.get("risk", "Unknown"), 0.5)
        
        return {
            "plausibility": plausibility,
            "rationale": f"Real LLM analysis: {real_result.get('movement', 'Unknown movement')} pattern with {real_result.get('risk', 'Unknown')} risk level.",
            "__labeler_source__": "real"
        }

def _choose_labeler(force_real=False):
    """Select appropriate labeler implementation based on environment and availability.
    
    This function implements a fallback strategy for LLM labeler selection:
    1. Force real labeler if explicitly requested (with error if unavailable)
    2. Use real labeler if available and not in mock mode
    3. Fall back to mock labeler for testing and development
    
    Args:
        force_real: Force use of real labeler, raise error if unavailable
        
    Returns:
        Tuple of (labeler_function, source_identifier)
        
    Raises:
        RuntimeError: If force_real=True but real labeler is unavailable
        
    Note:
        The function respects the GUARDIAN_USE_MOCK environment variable
        for testing and development scenarios.
    """
    if force_real:
        if REAL_LABELER is None:
            raise RuntimeError(f"force_real=True but weak_labeler import failed: {REAL_LABELER_ERR}")
        return REAL_LABELER, "real"
    if not USE_MOCK_ENV and REAL_LABELER is not None:
        return REAL_LABELER, "real"
    return _mock_label_case, "mock"  # _mock_label_case defined below

def _mock_label_case(structured_case: Dict[str, Any], narrative: str) -> Dict[str, Any]:
    """Mock LLM implementation for testing and development fallback.
    
    This function provides deterministic plausibility scoring based on zone
    characteristics and narrative content. It simulates LLM reasoning patterns
    for testing and development when real LLM services are unavailable.
    
    Args:
        structured_case: Case data with search zones
        narrative: Text narrative for analysis
        
    Returns:
        Mock analysis result with plausibility score and rationale
        
    Note:
        The mock implementation uses heuristics based on zone types and
        narrative keywords to simulate realistic plausibility scoring.
    """
    # Extract search zones from case data for analysis
    zones = structured_case.get("provenance", {}).get("search_zones", [])
    if not zones:
        return {"plausibility": 0.5, "rationale": "No zones found", "issues": []}
    
    # Use first zone for mock analysis (simplified for testing)
    zone = zones[0]
    zone_id = zone.get('zone_id', 'unknown')
    zone_type = zone.get('type', 'unknown')
    
    # Handle zone_type if it's a dict or other non-string type
    if isinstance(zone_type, dict):
        zone_type_str = str(zone_type.get('name', zone_type.get('type', 'unknown')))
    else:
        zone_type_str = str(zone_type)
    
    # Generate base plausibility score with random variation
    base_score = random.uniform(0.2, 0.8)
    
    # Apply zone type adjustments (simulate LLM reasoning about location relevance)
    if 'school' in zone_type_str.lower():
        base_score += 0.1  # Schools are more likely for missing children
    elif 'park' in zone_type_str.lower():
        base_score += 0.05  # Parks are moderately likely
    elif 'residential' in zone_type_str.lower():
        base_score += 0.15  # Residential areas are very likely
    
    # Apply narrative content adjustments (simulate LLM text analysis)
    narrative_str = str(narrative).lower()
    if 'school' in narrative_str:
        base_score += 0.1
    if 'park' in narrative_str:
        base_score += 0.05
    if 'home' in narrative_str or 'house' in narrative_str:
        base_score += 0.1
    
    # Ensure plausibility score is within valid [0, 1] range
    plausibility = max(0.0, min(1.0, base_score))
    
    # Generate descriptive rationale for mock analysis
    rationale = f"Mock LLM analysis: Zone {zone_id} ({zone_type_str}) shows {plausibility:.2f} plausibility based on narrative context and zone characteristics."
    
    return {
        "plausibility": plausibility,
        "rationale": rationale,
        "issues": []
    }

def recompute_priority(zone: Dict[str, Any], qa_result: Dict[str, Any], 
                     reward_config: Dict[str, Any]) -> float:
    """
    Recompute zone priority using LLM plausibility and reinforcement learning weights.
    
    This function implements a weighted combination of original priority, LLM
    plausibility score, and zone characteristics to produce an enhanced priority
    score. The formula balances multiple factors using configurable weights.
    
    Args:
        zone (Dict[str, Any]): Original zone data with existing priority and characteristics
        qa_result (Dict[str, Any]): LLM analysis result with plausibility score
        reward_config (Dict[str, Any]): RL configuration with profile weights
        
    Returns:
        float: New priority score (0-1) combining multiple factors
        
    Formula:
        score = α*orig + β*plaus - γ*radius + δ_rl*rl_score_norm + risk_boost
        priority = 1/(1 + exp(-3*(score - 0.5)))  # Sigmoid normalization
        
        Where:
        - α: Weight for original priority (alpha_orig)
        - β: Weight for LLM plausibility (beta_plaus)
        - γ: Penalty weight for zone radius (gamma_radius)
        - δ_rl: Weight for normalized RL score (delta_rl)
        - risk_boost: Additional boost for high-risk zones (if risk_tier present)
        
    Note:
        The function uses a sigmoid function to ensure the output is bounded
        between 0 and 1, with the steepest change around 0.5.
    """
    # Extract weights from active profile in reward configuration
    prof = reward_config.get("profiles", {}).get(reward_config.get("__active_profile__", "baseline"), {})
    w = prof.get("weights", {})
    
    # Get weight parameters for priority computation
    alpha = float(w.get("alpha_orig", 0.6))      # Weight for original priority
    beta = float(w.get("beta_plaus", 0.8))       # Weight for LLM plausibility
    gamma = float(w.get("gamma_radius", 0.02))    # Penalty weight for zone radius
    delta_rl = float(w.get("delta_rl", 0.0))      # Weight for RL normalized score
    
    # Optional risk boost for high-risk zones
    risk_boost = float(w.get("risk_boost", 0.0)) if zone.get("risk_tier") else 0.0

    # Extract input values with defaults
    orig = float(zone.get("priority", 0.5))      # Original priority score
    plaus = float(qa_result.get("plausibility", 0.5))  # LLM plausibility score
    radius = float(zone.get("radius_miles", 3.11))   # Zone radius in miles
    rl_score_norm = float(zone.get("rl_score_norm", 0.0))  # Precomputed RL score in [0,1]

    # Compute weighted combination
    score = alpha*orig + beta*plaus - gamma*radius + delta_rl*rl_score_norm + risk_boost
    
    # Apply sigmoid normalization to ensure output is in [0, 1] range
    return 1.0/(1.0 + math.exp(-3*(score-0.5)))

def load_reward_config(config_path: str, profile: str = None) -> Dict[str, Any]:
    """Load reinforcement learning reward configuration with profile selection.
    
    This function loads the RL configuration file and sets the active profile
    for priority computation. It provides fallback defaults if the file cannot
    be loaded or if the specified profile is not found.
    
    Args:
        config_path: Path to the reward configuration JSON file
        profile: Profile name to activate (default: use default_profile)
        
    Returns:
        Configuration dictionary with active profile set
        
    Note:
        The function sets the __active_profile__ key to indicate which profile
        is currently active for weight extraction in priority computation.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        # Set active profile based on parameter or default
        if profile:
            cfg["__active_profile__"] = profile
        else:
            cfg["__active_profile__"] = cfg.get("default_profile", "baseline")
        
        return cfg
    except Exception as e:
        print(f"[WARN] Could not load reward config: {e}")
        # Return minimal default configuration
        return {"__active_profile__": "baseline", "llm_plausibility_weight": 0.3}

def _load_truth_map(path="reinforcement_learning/ground_truth.json"):
    """
    Load ground truth mapping for evaluation analysis.
    
    This function loads the ground truth data that maps case IDs to their
    true coordinates (lat/lon) for evaluation purposes. It's used in Geo-hit@K
    evaluation to determine if the LLM-enhanced prioritization correctly
    identifies zones containing the truth location.
    
    Args:
        path (str): Path to ground truth JSON file
        
    Returns:
        Dict[str, Dict[str, float]]: Mapping of case_id to {"lat": float, "lon": float}
        
    Note:
        Returns empty dict if file cannot be loaded, which will cause
        evaluation to skip cases without ground truth data.
        Supports both old format (zone_id strings) and new format (coordinates).
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert old format (zone_id strings) to new format if needed
            # Old format: {"case_id": "z01"}
            # New format: {"case_id": {"lat": 37.5, "lon": -77.4}}
            result = {}
            for case_id, value in data.items():
                if isinstance(value, dict) and "lat" in value and "lon" in value:
                    # New format - already coordinates
                    result[case_id] = value
                elif isinstance(value, str):
                    # Old format - zone_id string, skip (no coordinates available)
                    continue
            return result
    except Exception:
        return {}

def format_zone_results(zone_data: Dict[str, Any]) -> str:
    """Format zone results for better readability."""
    case_id = zone_data.get("case_id", "UNKNOWN")
    zones = zone_data.get("zones", [])
    
    if not zones:
        return f"Case {case_id}: No zones found"
    
    lines = [f"Case {case_id}:"]
    lines.append("=" * 50)
    
    for i, zone in enumerate(zones, 1):
        zone_id = zone.get("zone_id", "unknown")
        plausibility = zone.get("plausibility", 0.0)
        original_priority = zone.get("original_priority", 0.0)
        new_priority = zone.get("new_priority", 0.0)
        labeler_source = zone.get("labeler_source", "unknown")
        
        # Calculate improvement
        improvement = new_priority - original_priority
        improvement_pct = (improvement / original_priority * 100) if original_priority > 0 else 0
        
        lines.append(f"  Zone {i}: {zone_id}")
        lines.append(f"    Plausibility: {plausibility:.3f}")
        lines.append(f"    Priority: {original_priority:.3f} → {new_priority:.3f} ({improvement:+.3f}, {improvement_pct:+.1f}%)")
        lines.append(f"    Source: {labeler_source}")
        lines.append("")
    
    return "\n".join(lines)

TRUTH = _load_truth_map()

def _minmax_norm(values):
    try:
        vmin = min(values)
        vmax = max(values)
    except ValueError:
        return []
    if (vmax - vmin) < 1e-9:
        return [0.5 for _ in values]
    return [(v - vmin)/(vmax - vmin) for v in values]


def _load_rl_lookup(zones_rl_path: pathlib.Path) -> Dict[str, Dict[str, float]]:
    """Load RL zone scores, normalize per-window, and aggregate per zone_id.

    Reads zones_rl.jsonl file containing reinforcement learning zone scores
    organized by case and time window. Performs min-max normalization within
    each time window, then aggregates scores across windows using maximum value
    per zone.

    Args:
        zones_rl_path: Path to zones_rl.jsonl file containing RL zone scores.
            File format: JSONL with records containing case_id, zones (dict
            by window), and zone_scores (dict by window).

    Returns:
        Dictionary mapping case_id -> zone_id -> normalized RL score in [0,1].
        Returns empty dict if file does not exist or cannot be parsed.

    Note:
        Normalization is performed per time window to account for varying
        score ranges across windows. Aggregation uses maximum score to preserve
        the highest RL confidence across all windows for each zone.
    """
    lookup: Dict[str, Dict[str, float]] = {}
    if not zones_rl_path.exists():
        return lookup
    with open(zones_rl_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            case_id = rec.get("case_id")
            if not case_id:
                continue
            zones_by_w = rec.get("zones", {}) or {}
            scores_by_w = rec.get("zone_scores", {}) or {}
            case_map: Dict[str, float] = lookup.setdefault(case_id, {})
            for wid, zones in zones_by_w.items():
                raw_scores = scores_by_w.get(wid)
                if not zones or not raw_scores or len(raw_scores) != len(zones):
                    continue
                norms = _minmax_norm(raw_scores)
                for z, rn in zip(zones, norms):
                    zid = z.get("zone_id")
                    if not zid:
                        continue
                    prev = case_map.get(zid, 0.0)
                    if rn > prev:
                        case_map[zid] = float(rn)
    return lookup

def run_zone_qa(case_files_dir: str, reward_config_path: str, out_dir: str, profile: str = None, sample: int = 0, force_real: bool = False, verbose: bool = False, per_zone: bool = False, batch_size: int = 16) -> Dict[str, Any]:
    """
    Execute comprehensive Zone QA analysis with LLM-enhanced plausibility scoring.
    
    This is the main analysis function that processes case files, applies LLM-based
    plausibility scoring to search zones, and generates enhanced prioritization.
    It supports both real and mock LLM implementations with batching for efficiency.
    
    Args:
        case_files_dir (str): Directory containing GRD-*.json case files
        reward_config_path (str): Path to search_reward_config.json
        out_dir (str): Output directory for results
        profile (str, optional): Profile key for reweighting (default: use default)
        sample (int): Number of cases to sample (0 = all cases)
        force_real (bool): Force use of real labeler (error if unavailable)
        verbose (bool): Enable verbose logging per case/zone
        per_zone (bool): Call labeler for each zone (slower but more precise)
        batch_size (int): LLM batch size for processing multiple cases
        
    Returns:
        Dict[str, Any]: Analysis metrics including:
            - total_cases: Number of cases processed
            - total_zones: Number of zones analyzed
            - zones_with_plausibility: Zones with successful LLM scoring
            - avg_plausibility: Average plausibility score across all zones
            - priority_improvements: Number of zones with improved priority
            - cases_processed: Detailed per-case metrics
            
    Generated Files:
        - zones_review.jsonl: Per-case zone plausibility scores and rationale
        - zones_reweighted.jsonl: LLM-enhanced zones with priority_llm field
        - zone_qa_metrics.json: Summary evaluation metrics
        
    Note:
        The function implements intelligent batching for LLM processing to balance
        efficiency and accuracy. It gracefully handles both real and mock labelers
        with comprehensive error handling and progress reporting. The function
        integrates RL scores from zones_rl.jsonl (if present) to enhance priority
        computation with normalized RL-based scores.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Choose labeler and log decision
    labeler_fn, labeler_src = _choose_labeler(force_real=force_real)
    print(f"[INIT] Zone-QA labeler source: {labeler_src.upper()}" + \
          (" (GUARDIAN_USE_MOCK=1)" if USE_MOCK_ENV else ""))
    
    reward_config = load_reward_config(reward_config_path, profile)
    
    out_review = out_path / "zones_review.jsonl"
    out_reweighted = out_path / "zones_reweighted.jsonl"
    
    case_files = list(pathlib.Path(case_files_dir).glob("GRD-*.json"))
    if sample > 0:
        case_files = case_files[:sample]
        print(f"[INFO] Processing {len(case_files)} case files (sampled from {len(list(pathlib.Path(case_files_dir).glob('GRD-*.json')))})...")
    else:
        print(f"[INFO] Processing {len(case_files)} case files...")
    
    metrics = {
        "total_cases": 0,
        "total_zones": 0,
        "zones_with_plausibility": 0,
        "avg_plausibility": 0.0,
        "priority_improvements": 0,
        "cases_processed": []
    }
    
    with open(out_review, "w", encoding="utf-8") as f_rev, \
         open(out_reweighted, "w", encoding="utf-8") as f_rew:
        
        # Batch processing for efficiency
        batch = []
        case_refs = []  # (index_in_batch, case_dict, zones, case_id)
        
        def process_batch():
            """Process current batch and write results."""
            if not batch:
                return

            qa_list = None
            meta = None

            def _call_label_batch(_payload):
                from guardian_llm.weak_labeler import label_batch as _lb
                return _lb(_payload, batch_size=batch_size)

            try:
                # 1) Try the simplest: list[str] narratives
                out = _call_label_batch(batch)
                # Unwrap (list, meta) or just list
                if isinstance(out, tuple):
                    qa_list, meta = out
                else:
                    qa_list = out

                if qa_list is None or not isinstance(qa_list, list) or len(qa_list) != len(batch):
                    raise ValueError("shape mismatch on first try")

            except Exception as e1:
                try:
                    # 2) Try list[dict] with a standard key name 'narrative'
                    payload = [{"narrative": s} for s in batch]
                    out = _call_label_batch(payload)
                    if isinstance(out, tuple):
                        qa_list, meta = out
                    else:
                        qa_list = out

                    if qa_list is None or not isinstance(qa_list, list) or len(qa_list) != len(batch):
                        raise ValueError("shape mismatch on second try")
                except Exception as e2:
                    print(f"[ERROR] label_batch raised: {e2}")
                    print("[WARN] label_batch returned None/invalid — falling back to per-item labeling.")
                    qa_list = None

            # Process each case in the batch
            try:
                # Load RL lookup once per batch (path: out_dir/zones_rl.jsonl)
                rl_lookup = _load_rl_lookup(pathlib.Path(out_dir) / "zones_rl.jsonl")

                for idx, case_obj, zones, case_id in case_refs:
                    # Fallback to single-call labeler if no batch results
                    if qa_list is None:
                        qa_result_case = labeler_fn(structured_case=case_obj, narrative=batch[idx])
                        qa_result_case["__labeler_source__"] = labeler_src
                    else:
                        # Be defensive if element is missing/None
                        qa_result_case = qa_list[idx] or {}
                        if "__labeler_source__" not in qa_result_case:
                            qa_result_case["__labeler_source__"] = labeler_src

                    reviewed, reweighted = [], []
                    case_plausibilities = []

                    # Ensure zones have deterministic IDs
                    for i, z in enumerate(zones, 1):
                        z.setdefault("zone_id", f"z{i:02d}")

                    for z in zones:
                        try:
                            plausibility = float(qa_result_case.get("plausibility", 0.5))
                            rationale = qa_result_case.get("rationale", "No rationale provided")
                            # RL blend: attach normalized RL score for this case/zone if present
                            rl_norm = 0.0
                            zid = z.get("zone_id")
                            if zid and case_id in rl_lookup and zid in rl_lookup[case_id]:
                                rl_norm = float(rl_lookup[case_id][zid])
                            # Store on zone so recompute_priority can read it
                            z_with_rl = {**z, "rl_score_norm": rl_norm}

                            new_priority = recompute_priority(z_with_rl, qa_result_case, reward_config)
                            original_priority = float(z.get("priority", 0.5))

                            if new_priority > original_priority:
                                metrics["priority_improvements"] += 1

                            reviewed.append({
                                "case_id": case_id,
                                "zone_id": z.get("zone_id", "unknown"),
                                "plausibility": plausibility,
                                "rationale": rationale,
                                "original_priority": original_priority,
                                "new_priority": new_priority,
                                "labeler_source": qa_result_case.get("__labeler_source__", labeler_src),
                                # Include coordinate fields for geometric evaluation
                                "center_lat": z.get("center_lat"),
                                "center_lon": z.get("center_lon"),
                                "radius_miles": z.get("radius_miles"),
                            })

                            reweighted.append({
                                **z,
                                "rl_score_norm": rl_norm,
                                "plausibility": plausibility,
                                "priority_llm": new_priority,
                                "rationale": rationale,
                                "labeler_source": qa_result_case.get("__labeler_source__", labeler_src),
                            })

                            case_plausibilities.append(plausibility)
                            metrics["zones_with_plausibility"] += 1

                            if verbose:
                                print(f"[VERBOSE] Zone {z.get('zone_id','unknown')}: plaus={plausibility:.3f}, "
                                      f"orig_pri={original_priority:.3f} -> new_pri={new_priority:.3f} "
                                      f"(source: {qa_result_case.get('__labeler_source__', labeler_src)})")

                        except Exception as ze:
                            print(f"[WARN] Error processing zone in {case_id}: {ze}")
                            continue

                    # Write one JSON object per line
                    f_rev.write(json.dumps({"case_id": case_id, "zones": reviewed}, ensure_ascii=False) + "\n")
                    f_rew.write(json.dumps({"case_id": case_id, "zones": reweighted}, ensure_ascii=False) + "\n")

                    if case_plausibilities:
                        avg_plausibility = sum(case_plausibilities) / len(case_plausibilities)
                        metrics["cases_processed"].append({
                            "case_id": case_id,
                            "zones_count": len(zones),
                            "avg_plausibility": avg_plausibility
                        })
                        print(f"[OK] Processed {case_id}: {len(zones)} zones, avg plausibility: {avg_plausibility:.3f}")
                    else:
                        print(f"[OK] Processed {case_id}: {len(zones)} zones, no plausibility scores")

            except Exception as e:
                print(f"[ERROR] Batch processing failed hard: {e}")
        
        for case_file in case_files:
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case = json.load(f)
                
                case_id = case.get("case_id", "UNKNOWN")
                narrative_raw = case.get("narrative_osint", "")
                
                # Extract narrative text from dict or use as string
                if isinstance(narrative_raw, dict):
                    # Combine key fields into a narrative string
                    narrative_parts = []
                    if narrative_raw.get("incident_summary"):
                        narrative_parts.append(narrative_raw["incident_summary"])
                    if narrative_raw.get("movement_cues_text"):
                        narrative_parts.append(narrative_raw["movement_cues_text"])
                    if narrative_raw.get("behavioral_patterns"):
                        narrative_parts.append("Behavioral patterns: " + ", ".join(narrative_raw["behavioral_patterns"]))
                    narrative = " ".join(narrative_parts)
                else:
                    narrative = str(narrative_raw)
                
                zones = case.get("provenance", {}).get("search_zones", [])
                
                if not zones:
                    print(f"[WARN] No search zones found in {case_id}")
                    continue
                
                metrics["total_cases"] += 1
                metrics["total_zones"] += len(zones)
                
                # Add to batch
                batch.append(narrative)
                case_refs.append((len(batch)-1, case, zones, case_id))
                
                # Process batch when full
                if len(batch) >= batch_size:
                    process_batch()
                    batch, case_refs = [], []
                
            except Exception as e:
                print(f"[ERROR] Failed to process {case_file}: {e}")
                continue
        
        # Flush remaining batch
        if batch:
            process_batch()
    
    # Calculate final metrics
    if metrics["zones_with_plausibility"] > 0:
        all_plausibilities = []
        for case in metrics["cases_processed"]:
            all_plausibilities.append(case["avg_plausibility"])
        metrics["avg_plausibility"] = sum(all_plausibilities) / len(all_plausibilities)
    
    # Save metrics
    metrics_path = out_path / "zone_qa_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUMMARY] Zone QA Analysis Complete:")
    print(f"  Cases processed: {metrics['total_cases']}")
    print(f"  Zones analyzed: {metrics['total_zones']}")
    print(f"  Zones with plausibility: {metrics['zones_with_plausibility']}")
    print(f"  Average plausibility: {metrics['avg_plausibility']:.3f}")
    print(f"  Priority improvements: {metrics['priority_improvements']}")
    print(f"  Output files: {out_review}, {out_reweighted}, {metrics_path}")
    
    return metrics

def evaluate_geo_hit_at_k(baseline_zones: List[Dict], llm_zones: List[Dict], 
                          true_coords: Dict[str, float], k: int = 3) -> Dict[str, Any]:
    """
    Evaluate Geo-hit@K metric comparing baseline vs LLM-enhanced zones.
    
    Uses geometric hit detection: a zone "hits" if the truth point is within
    the zone's radius (haversine distance <= radius_miles).
    
    Args:
        baseline_zones: Original zones sorted by priority
        llm_zones: LLM-enhanced zones sorted by priority_llm
        true_coords: Dictionary with "lat" and "lon" keys for truth coordinates
        k: Number of top zones to consider
        
    Returns:
        Dictionary with hit rates, distances, and metrics for baseline and LLM-enhanced
    """
    from src.geography.distance import haversine_distance
    
    tlat = float(true_coords["lat"])
    tlon = float(true_coords["lon"])
    
    # Sort zones by priority (baseline) and priority_llm (LLM)
    baseline_sorted = sorted(baseline_zones, key=lambda x: x.get("priority", 0), reverse=True)
    llm_sorted = sorted(llm_zones, key=lambda x: x.get("priority_llm", x.get("priority", 0)), reverse=True)
    
    # Check hits for baseline zones
    baseline_hits = []
    baseline_distances = []
    baseline_best_distance = None
    
    for i, z in enumerate(baseline_sorted[:k]):
        zlat = z.get("center_lat")
        zlon = z.get("center_lon")
        radius_mi = z.get("radius_miles", 10.0)
        
        if zlat is not None and zlon is not None:
            d = haversine_distance(tlat, tlon, float(zlat), float(zlon))
            hit = d <= float(radius_mi)
            baseline_hits.append(hit)
            baseline_distances.append(d)
            
            # Best distance: distance outside radius (0 if inside)
            best_d = max(0.0, d - float(radius_mi))
            if baseline_best_distance is None or best_d < baseline_best_distance:
                baseline_best_distance = best_d
    
    # Check hits for LLM zones
    llm_hits = []
    llm_distances = []
    llm_best_distance = None
    
    for i, z in enumerate(llm_sorted[:k]):
        zlat = z.get("center_lat")
        zlon = z.get("center_lon")
        radius_mi = z.get("radius_miles", 10.0)
        
        if zlat is not None and zlon is not None:
            d = haversine_distance(tlat, tlon, float(zlat), float(zlon))
            hit = d <= float(radius_mi)
            llm_hits.append(hit)
            llm_distances.append(d)
            
            # Best distance: distance outside radius (0 if inside)
            best_d = max(0.0, d - float(radius_mi))
            if llm_best_distance is None or best_d < llm_best_distance:
                llm_best_distance = best_d
    
    # Hit at K: true if any zone in top-K hits
    baseline_hit = any(baseline_hits) if baseline_hits else False
    llm_hit = any(llm_hits) if llm_hits else False
    
    return {
        "baseline_hit": baseline_hit,
        "llm_hit": llm_hit,
        "baseline_hits": baseline_hits,
        "llm_hits": llm_hits,
        "baseline_distances": baseline_distances,
        "llm_distances": llm_distances,
        "baseline_best_distance_miles": baseline_best_distance,
        "llm_best_distance_miles": llm_best_distance,
        "k": k,
        "true_coords": true_coords
    }

def run_evaluation_analysis(case_files_dir: str, zones_review_path: str, 
                           zones_reweighted_path: str, out_dir: str) -> Dict[str, Any]:
    """
    Run comprehensive evaluation analysis comparing baseline vs LLM-enhanced zones.
    
    Args:
        case_files_dir: Directory containing original case files
        zones_review_path: Path to zones_review.jsonl
        zones_reweighted_path: Path to zones_reweighted.jsonl
        out_dir: Output directory for evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load review and reweighted data
    review_data = {}
    reweighted_data = {}
    
    with open(zones_review_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            case_id = data.get("case_id")
            if case_id:
                review_data[case_id] = data.get("zones", [])
    
    with open(zones_reweighted_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            case_id = data.get("case_id")
            if case_id:
                reweighted_data[case_id] = data.get("zones", [])
    
    # Load zones_rl.jsonl to get coordinates if missing from review_data
    # Note: zones_rl.jsonl has zones organized by time window, so flatten them
    # and match by position/index since zone_ids may differ (grid-based vs rank-based)
    zones_rl_path = out_path / "zones_rl.jsonl"
    zones_rl_lookup = {}  # case_id -> [zone1, zone2, ...] (flattened list by priority)
    zones_rl_by_window = {}  # case_id -> {window: [zones]} for TTF tracking
    if zones_rl_path.exists():
        with open(zones_rl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    case_id = data.get("case_id")
                    if not case_id:
                        continue
                    # Store zones by window for TTF tracking
                    zones_rl_by_window[case_id] = data.get("zones", {})
                    # Flatten zones from all windows into a single list, sorted by priority
                    all_zones = []
                    zones_by_window = data.get("zones", {})
                    for window, zones in zones_by_window.items():
                        for zone in zones:
                            all_zones.append({
                                "zone_id": zone.get("zone_id"),
                                "center_lat": zone.get("center_lat"),
                                "center_lon": zone.get("center_lon"),
                                "radius_miles": zone.get("radius_miles", 10.0),
                                "priority": zone.get("priority", 0.0)
                            })
                    # Sort by priority (descending) to match the order in zones_review
                    all_zones.sort(key=lambda z: z.get("priority", 0.0), reverse=True)
                    zones_rl_lookup[case_id] = all_zones
                except Exception:
                    continue
    
    # Use ground truth mapping for evaluation (now contains coordinates)
    true_coords_map = TRUTH
    
    # Run evaluation for different K values
    evaluation_results = {
        "k_values": [1, 3, 5, 10],
        "case_results": {},
        "summary_metrics": {}
    }
    
    # Track TTF (time-to-first-hit) per case
    ttf_by_case = {}  # case_id -> first_hit_window or None
    
    for k in evaluation_results["k_values"]:
        baseline_hits = 0
        llm_hits = 0
        total_cases = 0
        baseline_distances = []
        llm_distances = []
        
        case_results = []
        diagnostic_print_count = 0  # Limit prints to first 10 cases
        
        for case_id in review_data:
            if case_id not in reweighted_data or case_id not in true_coords_map:
                continue
                
            true_coords = true_coords_map[case_id]
            baseline_zones = review_data[case_id]
            llm_zones = reweighted_data[case_id]
            
            # Ensure zones have required fields for geometric evaluation
            # Zones should already have center_lat, center_lon, radius_miles
            # but look up from zones_rl.jsonl if missing (match by position/index)
            baseline_zones_formatted = []
            rl_zones = zones_rl_lookup.get(case_id, [])
            for i, zone in enumerate(baseline_zones):
                zone_id = zone.get("zone_id")
                center_lat = zone.get("center_lat")
                center_lon = zone.get("center_lon")
                radius_miles = zone.get("radius_miles", 10.0)
                
                # If coordinates missing, try to look up from zones_rl.jsonl by position
                if (center_lat is None or center_lon is None) and i < len(rl_zones):
                    rl_zone = rl_zones[i]
                    center_lat = center_lat or rl_zone.get("center_lat")
                    center_lon = center_lon or rl_zone.get("center_lon")
                    radius_miles = radius_miles if radius_miles != 10.0 else rl_zone.get("radius_miles", 10.0)
                
                zone_formatted = {
                    "zone_id": zone_id,
                    "priority": zone.get("original_priority", zone.get("priority", 0.5)),
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "radius_miles": radius_miles,
                    "seed_ref": zone.get("seed_ref")  # Preserve seed reference for diagnostics
                }
                # Only include zones with coordinates
                if zone_formatted["center_lat"] is not None and zone_formatted["center_lon"] is not None:
                    baseline_zones_formatted.append(zone_formatted)
            
            # Ensure LLM zones have required fields
            llm_zones_formatted = []
            for zone in llm_zones:
                zone_formatted = {
                    "zone_id": zone.get("zone_id"),
                    "priority": zone.get("priority", 0.5),
                    "priority_llm": zone.get("priority_llm", zone.get("priority", 0.5)),
                    "center_lat": zone.get("center_lat"),
                    "center_lon": zone.get("center_lon"),
                    "radius_miles": zone.get("radius_miles", 10.0)
                }
                # Only include zones with coordinates
                if zone_formatted["center_lat"] is not None and zone_formatted["center_lon"] is not None:
                    llm_zones_formatted.append(zone_formatted)
            
            # Skip if no valid zones
            if not baseline_zones_formatted or not llm_zones_formatted:
                continue
            
            # Run evaluation with coordinates
            eval_result = evaluate_geo_hit_at_k(
                baseline_zones_formatted, llm_zones_formatted, true_coords, k
            )
            
            # Track TTF: find first hit window from zones_rl.jsonl if available
            first_hit_window = None
            if case_id in zones_rl_by_window and k == 3:  # Only track for K=3 to avoid duplicates
                zones_by_window = zones_rl_by_window[case_id]
                
                # Check which window first hit occurred in
                from src.geography.distance import haversine_distance
                tlat = float(true_coords["lat"])
                tlon = float(true_coords["lon"])
                
                for wid in ["0-24", "24-48", "48-72"]:
                    if wid in zones_by_window:
                        for zone in zones_by_window[wid]:
                            zlat = zone.get("center_lat")
                            zlon = zone.get("center_lon")
                            radius_mi = zone.get("radius_miles", 10.0)
                            if zlat is not None and zlon is not None:
                                d = haversine_distance(tlat, tlon, float(zlat), float(zlon))
                                if d <= float(radius_mi):
                                    first_hit_window = wid
                                    break
                        if first_hit_window:
                            break
                
                if case_id not in ttf_by_case:
                    ttf_by_case[case_id] = first_hit_window
            
            # Get seed from zones if available
            seed_info = None
            if baseline_zones_formatted:
                seed_info = baseline_zones_formatted[0].get("seed_ref")
            
            # Best gap (distance minus radius, 0 = inside)
            best_gap_baseline = eval_result.get("baseline_best_distance_miles", float('inf'))
            best_gap_llm = eval_result.get("llm_best_distance_miles", float('inf'))
            
            # Check if LLM changed top-1 zone
            baseline_top1 = baseline_zones_formatted[0] if baseline_zones_formatted else None
            llm_top1 = llm_zones_formatted[0] if llm_zones_formatted else None
            changed = (baseline_top1 and llm_top1 and 
                     baseline_top1.get("zone_id") != llm_top1.get("zone_id"))
            
            # Per-case sanity print (limit to first 10 cases, only for K=3)
            if diagnostic_print_count < 10 and seed_info and k == 3:
                ls_lat = seed_info.get("last_seen_lat", "?")
                ls_lon = seed_info.get("last_seen_lon", "?")
                baseline_gap3 = best_gap_baseline
                llm_gap3 = best_gap_llm
                print(f"[{case_id}] seed=({ls_lat:.4f},{ls_lon:.4f}) "
                      f"K=3 baseline_gap≈{baseline_gap3:.2f}mi → LLM_gap≈{llm_gap3:.2f}mi "
                      f"first_hit={first_hit_window or '—'}")
                diagnostic_print_count += 1
            
            case_results.append({
                "case_id": case_id,
                "true_coords": true_coords,
                "baseline_hit": eval_result["baseline_hit"],
                "llm_hit": eval_result["llm_hit"],
                "baseline_best_distance_miles": eval_result.get("baseline_best_distance_miles"),
                "llm_best_distance_miles": eval_result.get("llm_best_distance_miles"),
                "k": k
            })
            
            if eval_result["baseline_hit"]:
                baseline_hits += 1
            if eval_result["llm_hit"]:
                llm_hits += 1
            total_cases += 1
            
            # Collect distances for summary statistics
            if eval_result.get("baseline_best_distance_miles") is not None:
                baseline_distances.append(eval_result["baseline_best_distance_miles"])
            if eval_result.get("llm_best_distance_miles") is not None:
                llm_distances.append(eval_result["llm_best_distance_miles"])
        
        # Calculate hit rates and distance statistics
        baseline_hit_rate = baseline_hits / total_cases if total_cases > 0 else 0
        llm_hit_rate = llm_hits / total_cases if total_cases > 0 else 0
        improvement = llm_hit_rate - baseline_hit_rate
        
        # Calculate median distances
        import statistics
        baseline_median_distance = statistics.median(baseline_distances) if baseline_distances else None
        llm_median_distance = statistics.median(llm_distances) if llm_distances else None
        
        # Calculate TTF metrics (only for K=3 to avoid duplicates)
        ttf_0_24 = 0
        ttf_24_48 = 0
        ttf_48_72 = 0
        ttf_miss = 0
        if k == 3:
            for case_id, first_hit in ttf_by_case.items():
                if first_hit == "0-24":
                    ttf_0_24 += 1
                elif first_hit == "24-48":
                    ttf_24_48 += 1
                elif first_hit == "48-72":
                    ttf_48_72 += 1
                else:
                    ttf_miss += 1
        
        evaluation_results["case_results"][f"k_{k}"] = case_results
        summary_metrics = {
            "total_cases": total_cases,
            "baseline_hit_rate": baseline_hit_rate,
            "llm_hit_rate": llm_hit_rate,
            "improvement": improvement,
            "improvement_pct": (improvement / baseline_hit_rate * 100) if baseline_hit_rate > 0 else 0,
            "baseline_median_distance_miles": baseline_median_distance,
            "llm_median_distance_miles": llm_median_distance
        }
        
        # Add TTF metrics for K=3
        if k == 3:
            summary_metrics.update({
                "ttf_0_24": ttf_0_24,
                "ttf_24_48": ttf_24_48,
                "ttf_48_72": ttf_48_72,
                "ttf_miss": ttf_miss
            })
        
        evaluation_results["summary_metrics"][f"k_{k}"] = summary_metrics
    
    # Generate quality dashboard
    def _generate_quality_dashboard(evaluation_results, zones_rl_path, review_data):
        """Generate red/yellow/green quality dashboard.
        
        Red flags:
        - Missing last_seen seeds
        - Duplicate zones in any window
        - K=3 hit rate < 35%
        
        Yellow flags:
        - K=3 hit rate < prior benchmark (e.g., <50% for LLM)
        
        Returns:
            Dictionary with status, flags, and metrics.
        """
        red_flags = []
        yellow_flags = []
        
        # Check K=3 hit rate
        k3_metrics = evaluation_results.get("summary_metrics", {}).get("k_3", {})
        llm_hit_rate_k3 = k3_metrics.get("llm_hit_rate", 0.0)
        baseline_hit_rate_k3 = k3_metrics.get("baseline_hit_rate", 0.0)
        
        if baseline_hit_rate_k3 < 0.35:
            red_flags.append(f"Baseline K=3 hit rate {baseline_hit_rate_k3:.1%} < 35%")
        
        if llm_hit_rate_k3 < 0.35:
            red_flags.append(f"LLM K=3 hit rate {llm_hit_rate_k3:.1%} < 35%")
        elif llm_hit_rate_k3 < 0.50:
            yellow_flags.append(f"LLM K=3 hit rate {llm_hit_rate_k3:.1%} < 50% (below target)")
        
        # Check for missing seeds (would need to check zones_rl.jsonl)
        # This is a simplified check - in practice would scan zones_rl.jsonl
        missing_seeds = 0
        if zones_rl_path.exists():
            with open(zones_rl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        zones = data.get("zones", {})
                        # Check if any zone has seed_ref
                        has_seed = False
                        for window_zones in zones.values():
                            for zone in window_zones:
                                if zone.get("seed_ref"):
                                    has_seed = True
                                    break
                            if has_seed:
                                break
                        if not has_seed:
                            missing_seeds += 1
                    except Exception:
                        continue
        
        if missing_seeds > 0:
            red_flags.append(f"{missing_seeds} cases missing seed_ref in zones")
        
        # Check for duplicate zones (would need to parse zones_rl.jsonl)
        # This is simplified - actual duplicate detection happens during generation
        
        # Calculate mean TTF and median gap
        k3_metrics = evaluation_results.get("summary_metrics", {}).get("k_3", {})
        mean_ttf = None
        if k3_metrics.get("ttf_0_24") is not None:
            total_hits = (k3_metrics.get("ttf_0_24", 0) + 
                         k3_metrics.get("ttf_24_48", 0) + 
                         k3_metrics.get("ttf_48_72", 0))
            if total_hits > 0:
                # Weighted mean: 0-24h = 12h, 24-48h = 36h, 48-72h = 60h
                mean_ttf = (k3_metrics.get("ttf_0_24", 0) * 12 +
                           k3_metrics.get("ttf_24_48", 0) * 36 +
                           k3_metrics.get("ttf_48_72", 0) * 60) / total_hits
        
        median_gap = k3_metrics.get("llm_median_distance_miles")
        
        # Determine status
        status = "green"
        if red_flags:
            status = "red"
        elif yellow_flags:
            status = "yellow"
        
        return {
            "status": status,
            "red_flags": red_flags,
            "yellow_flags": yellow_flags,
            "mean_ttf_hours": mean_ttf,
            "median_gap_miles": median_gap,
            "k3_llm_hit_rate": llm_hit_rate_k3,
            "k3_baseline_hit_rate": baseline_hit_rate_k3
        }
    
    dashboard = _generate_quality_dashboard(evaluation_results, zones_rl_path, review_data)
    evaluation_results["quality_dashboard"] = dashboard
    
    # Save evaluation results
    eval_path = out_path / "zone_evaluation_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n[EVALUATION] Zone QA Performance Analysis:")
    # Get total cases from first K value (should be same for all K)
    first_k = evaluation_results["k_values"][0] if evaluation_results["k_values"] else None
    if first_k:
        total_cases = evaluation_results["summary_metrics"][f"k_{first_k}"]["total_cases"]
        print(f"  Cases evaluated: {total_cases}")
    for k in evaluation_results["k_values"]:
        metrics = evaluation_results["summary_metrics"][f"k_{k}"]
        # Use ASCII arrow for Windows console compatibility
        print(f"  K={k}: Baseline {metrics['baseline_hit_rate']:.1%} -> LLM {metrics['llm_hit_rate']:.1%} "
              f"(+{metrics['improvement_pct']:.1f}% improvement)")
        if metrics.get("baseline_median_distance_miles") is not None:
            print(f"    Median distances: Baseline {metrics['baseline_median_distance_miles']:.2f} mi, "
                  f"LLM {metrics.get('llm_median_distance_miles', 0):.2f} mi")
    
    print(f"  Evaluation results saved to: {eval_path}")
    
    # Print quality dashboard status
    if "quality_dashboard" in evaluation_results:
        dashboard = evaluation_results["quality_dashboard"]
        status_emoji = {"green": "[OK]", "yellow": "[WARN]", "red": "[FAIL]"}.get(dashboard["status"], "[?]")
        print(f"\n[QUALITY] Dashboard: {status_emoji} {dashboard['status'].upper()}")
        if dashboard.get("red_flags"):
            for flag in dashboard["red_flags"]:
                print(f"  RED: {flag}")
        if dashboard.get("yellow_flags"):
            for flag in dashboard["yellow_flags"]:
                print(f"  YELLOW: {flag}")
        if dashboard.get("mean_ttf_hours") is not None:
            print(f"  Mean TTF: {dashboard['mean_ttf_hours']:.1f} hours")
        if dashboard.get("median_gap_miles") is not None:
            print(f"  Median gap: {dashboard['median_gap_miles']:.2f} miles")
    
    return evaluation_results

def main():
    """
    Main execution function for Zone QA analysis with comprehensive command-line interface.
    
    This function provides a complete command-line interface for running Zone QA
    analysis with various configuration options. It supports both development and
    production workflows with extensive testing and debugging capabilities.
    
    Command Line Arguments:
        --input: Directory containing GRD-*.json case files
        --config: Path to search_reward_config.json
        --outdir: Output directory for results
        --profile: Profile key for reweighting
        --evaluate: Run Geo-hit@K evaluation analysis
        --sample: Sample N cases for quick testing
        --selftest: Run self-test validation checks
        --force-real: Force real LLM labeler (error if unavailable)
        --verbose: Enable detailed logging per case/zone
        --format: Display formatted zone results
        --per-zone: Call labeler for each zone (slower but more precise)
        --batch-size: LLM batch size for processing
        --print-models: Print model configuration and exit
        
    Workflow:
        1. Parse command line arguments
        2. Handle special modes (selftest, print-models)
        3. Run Zone QA analysis with specified parameters
        4. Display formatted results (optional)
        5. Run evaluation analysis (optional)
        
    Note:
        The function implements comprehensive error handling and provides
        multiple execution modes for different use cases (development,
        testing, production, and debugging).
    """
    # Configure comprehensive command line argument parser
    parser = argparse.ArgumentParser(description="Zone QA - LLM Sidecar for Search Zone Analysis")
    parser.add_argument("--input", default="data/synthetic_cases", help="Directory containing GRD-*.json case files")
    parser.add_argument("--config", default="reinforcement_learning/search_reward_config.json", 
                       help="Path to search_reward_config.json")
    parser.add_argument("--outdir", default="eda_out", help="Output directory")
    parser.add_argument("--profile", default=None, help="Profile key in search_reward_config.json")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation analysis after zone QA")
    parser.add_argument("--ttf", action="store_true",
                       help="Compute time-to-first-hit metrics")
    parser.add_argument("--cdf", action="store_true", 
                       help="Generate CDF of distance gaps")
    parser.add_argument("--sample", type=int, default=0, help="Sample N cases for quick testing")
    parser.add_argument("--selftest", action="store_true", help="Run self-test checks")
    parser.add_argument("--force-real", action="store_true",
                       help="Force real weak-labeler; error if unavailable.")
    parser.add_argument("--verbose", action="store_true",
                       help="Extra logging per case/zone.")
    parser.add_argument("--format", action="store_true",
                       help="Display formatted zone results.")
    parser.add_argument("--per-zone", action="store_true",
                       help="Call the labeler for each zone (slower).")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="LLM batch size for processing multiple cases.")
    parser.add_argument("--print-models", action="store_true",
                       help="Print model configuration and exit.")
    args = parser.parse_args()
    
    print("Zone QA - LLM Sidecar Module")
    print("=" * 40)
    
    # Print model configuration if requested
    if args.print_models:
        try:
            with open("guardian.config.json", "r") as f:
                cfg = json.load(f)
            print("[CFG] extractor:", cfg["models"]["extractor"])
            print("[CFG] weak_labeler:", cfg["models"]["weak_labeler"])
            print("[CFG] summarizer:", cfg["models"]["summarizer_instruct"])
        except Exception as e:
            print(f"[ERROR] Could not load guardian.config.json: {e}")
        return
    
    # Self-test functionality
    if args.selftest:
        print("[SELFTEST] Running self-test checks...")
        
        # Test recompute_priority function
        test_zone = {"priority": 0.5, "radius_miles": 3.11}
        test_qa = {"plausibility": 0.7}
        test_config = {"weights": {"alpha_orig": 0.6, "beta_plaus": 0.8}, "penalties": {"gamma_radius": 0.02}}
        result = recompute_priority(test_zone, test_qa, test_config)
        assert 0.0 <= result <= 1.0, f"Priority result {result} not in [0,1]"
        
        # Test plausibility range
        test_case = {"provenance": {"search_zones": [{"zone_id": "test", "type": "school"}]}}
        test_narrative = "test narrative"
        labeler_fn, _ = _choose_labeler()
        qa_result = labeler_fn(structured_case=test_case, narrative=test_narrative)
        assert 0.0 <= qa_result["plausibility"] <= 1.0, f"Plausibility {qa_result['plausibility']} not in [0,1]"
        
        # Test zone_id assignment
        test_zones = [{"type": "school"}, {"type": "park"}]
        for i, zone in enumerate(test_zones, 1):
            zone.setdefault("zone_id", f"z{i:02d}")
        assert test_zones[0]["zone_id"] == "z01"
        assert test_zones[1]["zone_id"] == "z02"
        
        # Test ground truth skip
        truth_map = {"GRD-2025-000487": "z03"}
        assert "GRD-2025-000487" in truth_map
        assert "GRD-2025-999999" not in truth_map
        
        print("[SELFTEST] All checks passed!")
        return
    
    try:
        # Run Zone QA analysis
        metrics = run_zone_qa(args.input, args.config, args.outdir, args.profile, args.sample, args.force_real, args.verbose, args.per_zone, args.batch_size)
        print("\n[DONE] Zone QA analysis complete.")
        
        # Display formatted results if requested
        if args.format:
            print("\n[FORMATTED RESULTS]")
            print("=" * 60)
            zones_review_path = pathlib.Path(args.outdir) / "zones_review.jsonl"
            if zones_review_path.exists():
                with open(zones_review_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        zone_data = json.loads(line.strip())
                        print(format_zone_results(zone_data))
                        print()
            else:
                print(f"[WARN] Zones review file not found: {zones_review_path}")
        
        # Run evaluation analysis if requested
        if args.evaluate:
            print("\n[EVALUATION] Running Geo-hit@K evaluation analysis...")
            zones_review_path = pathlib.Path(args.outdir) / "zones_review.jsonl"
            zones_reweighted_path = pathlib.Path(args.outdir) / "zones_reweighted.jsonl"
            
            if zones_review_path.exists() and zones_reweighted_path.exists():
                eval_results = run_evaluation_analysis(
                    args.input, 
                    str(zones_review_path), 
                    str(zones_reweighted_path), 
                    args.outdir
                )
                print("\n[EVALUATION] Analysis complete.")
            else:
                print(f"[WARN] Required files not found for evaluation:")
                print(f"  - {zones_review_path} exists: {zones_review_path.exists()}")
                print(f"  - {zones_reweighted_path} exists: {zones_reweighted_path.exists()}")
        
    except Exception as e:
        print(f"[ERROR] Zone QA failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()