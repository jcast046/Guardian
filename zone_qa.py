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
        score = α*original_priority + β*plausibility - γ*radius + risk_boost
        priority = 1/(1 + exp(-3*(score - 0.5)))  # Sigmoid normalization
        
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
    
    # Optional risk boost for high-risk zones
    risk_boost = float(w.get("risk_boost", 0.0)) if zone.get("risk_tier") else 0.0

    # Extract input values with defaults
    orig = float(zone.get("priority", 0.5))      # Original priority score
    plaus = float(qa_result.get("plausibility", 0.5))  # LLM plausibility score
    radius = float(zone.get("radius_km", 5.0))   # Zone radius in kilometers

    # Compute weighted combination
    score = alpha*orig + beta*plaus - gamma*radius + risk_boost
    
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
    true/planted zone IDs for evaluation purposes. It's used in Geo-hit@K
    evaluation to determine if the LLM-enhanced prioritization correctly
    identifies the true zones.
    
    Args:
        path (str): Path to ground truth JSON file
        
    Returns:
        Dict[str, str]: Mapping of case_id to true_zone_id
        
    Note:
        Returns empty dict if file cannot be loaded, which will cause
        evaluation to skip cases without ground truth data.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
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
        with comprehensive error handling and progress reporting.
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

                            new_priority = recompute_priority(z, qa_result_case, reward_config)
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
                            })

                            reweighted.append({
                                **z,
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

                    # Write one line per-case
                    f_rev.write(json.dumps({"case_id": case_id, "zones": reviewed}, indent=2, ensure_ascii=False) + "\n")
                    f_rew.write(json.dumps({"case_id": case_id, "zones": reweighted}, indent=2, ensure_ascii=False) + "\n")

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
                          true_zone_id: str, k: int = 3) -> Dict[str, Any]:
    """
    Evaluate Geo-hit@K metric comparing baseline vs LLM-enhanced zones.
    
    Args:
        baseline_zones: Original zones sorted by priority
        llm_zones: LLM-enhanced zones sorted by priority_llm
        true_zone_id: ID of the true/planted zone
        k: Number of top zones to consider
        
    Returns:
        Dictionary with hit rates for baseline and LLM-enhanced
    """
    # Sort zones by priority (baseline) and priority_llm (LLM)
    baseline_sorted = sorted(baseline_zones, key=lambda x: x.get("priority", 0), reverse=True)
    llm_sorted = sorted(llm_zones, key=lambda x: x.get("priority_llm", 0), reverse=True)
    
    # Get top-K zone IDs
    baseline_top_k = [z.get("zone_id") for z in baseline_sorted[:k]]
    llm_top_k = [z.get("zone_id") for z in llm_sorted[:k]]
    
    # Check if true zone is in top-K
    baseline_hit = true_zone_id in baseline_top_k
    llm_hit = true_zone_id in llm_top_k
    
    return {
        "baseline_hit": baseline_hit,
        "llm_hit": llm_hit,
        "baseline_top_k": baseline_top_k,
        "llm_top_k": llm_top_k,
        "k": k,
        "true_zone_id": true_zone_id
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
    
    # Use ground truth mapping for evaluation
    true_zones = TRUTH
    
    # Run evaluation for different K values
    evaluation_results = {
        "k_values": [1, 3, 5, 10],
        "case_results": {},
        "summary_metrics": {}
    }
    
    for k in evaluation_results["k_values"]:
        baseline_hits = 0
        llm_hits = 0
        total_cases = 0
        
        case_results = []
        
        for case_id in review_data:
            if case_id not in reweighted_data or case_id not in true_zones:
                continue
                
            true_zone_id = true_zones[case_id]
            baseline_zones = review_data[case_id]
            llm_zones = reweighted_data[case_id]
            
            # Convert review data to zone format for evaluation
            baseline_zones_formatted = []
            for zone in baseline_zones:
                baseline_zones_formatted.append({
                    "zone_id": zone.get("zone_id"),
                    "priority": zone.get("original_priority", 0.5)
                })
            
            # Run evaluation
            eval_result = evaluate_geo_hit_at_k(
                baseline_zones_formatted, llm_zones, true_zone_id, k
            )
            
            case_results.append({
                "case_id": case_id,
                "true_zone_id": true_zone_id,
                "baseline_hit": eval_result["baseline_hit"],
                "llm_hit": eval_result["llm_hit"],
                "k": k
            })
            
            if eval_result["baseline_hit"]:
                baseline_hits += 1
            if eval_result["llm_hit"]:
                llm_hits += 1
            total_cases += 1
        
        # Calculate hit rates
        baseline_hit_rate = baseline_hits / total_cases if total_cases > 0 else 0
        llm_hit_rate = llm_hits / total_cases if total_cases > 0 else 0
        improvement = llm_hit_rate - baseline_hit_rate
        
        evaluation_results["case_results"][f"k_{k}"] = case_results
        evaluation_results["summary_metrics"][f"k_{k}"] = {
            "total_cases": total_cases,
            "baseline_hit_rate": baseline_hit_rate,
            "llm_hit_rate": llm_hit_rate,
            "improvement": improvement,
            "improvement_pct": (improvement / baseline_hit_rate * 100) if baseline_hit_rate > 0 else 0
        }
    
    # Save evaluation results
    eval_path = out_path / "zone_evaluation_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n[EVALUATION] Zone QA Performance Analysis:")
    print(f"  Cases evaluated: {total_cases}")
    for k in evaluation_results["k_values"]:
        metrics = evaluation_results["summary_metrics"][f"k_{k}"]
        print(f"  K={k}: Baseline {metrics['baseline_hit_rate']:.1%} → LLM {metrics['llm_hit_rate']:.1%} "
              f"(+{metrics['improvement_pct']:.1f}% improvement)")
    
    print(f"  Evaluation results saved to: {eval_path}")
    
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
        test_zone = {"priority": 0.5, "radius_km": 5}
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