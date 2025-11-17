"""
Behavioral Prior Sampling Module

Provides weighted sampling functions for behavioral priors from lexicon files.
These priors map directly to movement model and RL reward features.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

BASE = Path(".")

# Load lexicon files
def _load(p: Path) -> Dict[str, Any]:
    """Load JSON file with encoding."""
    return json.load(open(p, "r", encoding="utf-8"))

L_MOT = _load(BASE / "data/lexicons/offender_motives.json")
L_LT = _load(BASE / "data/lexicons/lures_transport.json")
L_MOV = _load(BASE / "data/lexicons/movement_profiles.json")
L_CON = _load(BASE / "data/lexicons/concealment_sites.json")
L_TIM = _load(BASE / "data/lexicons/time_patterns.json")


def _weighted_choice(weights: Dict[str, float]) -> str:
    """
    Weighted random choice from a dictionary of weights.
    
    Args:
        weights: Dictionary mapping item IDs to their weights
        
    Returns:
        Selected item ID
    """
    items = list(weights.items())
    total = sum(v for _, v in items)
    if total == 0:
        # Fallback to uniform if all weights are zero
        return random.choice(list(weights.keys())) if weights else None
    
    r = random.random() * total
    cumulative = 0.0
    
    for key, value in items:
        cumulative += value
        if r <= cumulative:
            return key
    
    # Fallback to last item (shouldn't happen due to floating point)
    return items[-1][0]


def sample_motive(region: Optional[str] = None, age_band: Optional[str] = None) -> str:
    """
    Sample an offender motive with optional regional and age biases.
    
    Args:
        region: Optional region name (e.g., "NoVA", "Rural") for regional bias
        age_band: Optional age band (e.g., "6-12", "13-17") for age bias
        
    Returns:
        Selected motive ID
    """
    # Build base priors
    priors = {m["id"]: m["prior"] for m in L_MOT["motives"]}
    
    # Apply regional constraints if specified
    if region and "constraints" in L_MOT and "by_region" in L_MOT["constraints"]:
        region_constraints = L_MOT["constraints"]["by_region"].get(region, {})
        for motive_id, multiplier in region_constraints.items():
            if motive_id in priors:
                priors[motive_id] *= multiplier
    
    # Apply age bias if specified
    if age_band:
        for motive in L_MOT["motives"]:
            if "age_bias" in motive and age_band in motive["age_bias"]:
                if motive["id"] in priors:
                    priors[motive["id"]] *= motive["age_bias"][age_band]
    
    # Filter by age_lock constraints
    for motive in L_MOT["motives"]:
        if "age_lock" in motive:
            # If age_lock exists and age_band doesn't match, set weight to 0
            age_lock = motive["age_lock"]
            if age_band:
                # Simple check: if age_lock is "<=1" and age_band is not "<=1", exclude
                if age_lock == "<=1" and age_band != "<=1":
                    priors[motive["id"]] = 0.0
                elif age_lock != "<=1" and age_band == "<=1":
                    # For non-infant motives, exclude if age_band is infant
                    priors[motive["id"]] = 0.0
    
    # Remove zero-weight items
    priors = {k: v for k, v in priors.items() if v > 0}
    
    if not priors:
        # Fallback to uniform if all filtered out
        priors = {m["id"]: 1.0 for m in L_MOT["motives"]}
    
    return _weighted_choice(priors)


def sample_lure(motive: Optional[str] = None) -> str:
    """
    Sample a lure method, optionally respecting motive couplings.
    
    Args:
        motive: Optional motive ID to apply coupling biases
        
    Returns:
        Selected lure ID
    """
    # Build base priors
    priors = {l["id"]: l["prior"] for l in L_LT["lures"]}
    
    # Apply motive couplings if specified
    if motive and "couplings" in L_LT:
        for coupling in L_LT["couplings"]:
            if coupling.get("motive") == motive and "lures" in coupling:
                for lure_id, multiplier in coupling["lures"].items():
                    if lure_id in priors:
                        priors[lure_id] *= multiplier
    
    return _weighted_choice(priors)


def sample_transport(motive: Optional[str] = None, lure: Optional[str] = None) -> str:
    """
    Sample a transport method, optionally respecting motive couplings.
    
    Args:
        motive: Optional motive ID to apply coupling biases
        lure: Optional lure ID (unused for now, but kept for future use)
        
    Returns:
        Selected transport ID
    """
    # Build base priors
    priors = {t["id"]: t["prior"] for t in L_LT["transport"]}
    
    # Apply motive couplings if specified
    if motive and "couplings" in L_LT:
        for coupling in L_LT["couplings"]:
            if coupling.get("motive") == motive and "transport" in coupling:
                for transport_id, multiplier in coupling["transport"].items():
                    if transport_id in priors:
                        priors[transport_id] *= multiplier
    
    # Check if lure requires vehicle
    if lure:
        lure_data = next((l for l in L_LT["lures"] if l["id"] == lure), None)
        if lure_data and lure_data.get("requires") and "vehicle" in lure_data["requires"]:
            # Boost vehicle transport options
            for transport_id in ["vehicle_local", "vehicle_highway"]:
                if transport_id in priors:
                    priors[transport_id] *= 1.5
    
    return _weighted_choice(priors)


def sample_movement_profile(motive: Optional[str] = None) -> str:
    """
    Sample a movement profile, optionally respecting motive couplings.
    
    Args:
        motive: Optional motive ID to apply coupling biases
        
    Returns:
        Selected movement profile ID
    """
    # Build base priors
    priors = {p["id"]: p["prior"] for p in L_MOV["profiles"]}
    
    # Apply motive couplings if specified
    if motive and "motive_couplings" in L_MOV:
        couplings = L_MOV["motive_couplings"].get(motive, {})
        for profile_id, multiplier in couplings.items():
            if profile_id in priors:
                priors[profile_id] *= multiplier
    
    return _weighted_choice(priors)


def sample_concealment_site(motive: Optional[str] = None) -> str:
    """
    Sample a concealment site type, optionally respecting motive couplings.
    
    Args:
        motive: Optional motive ID to apply coupling biases
        
    Returns:
        Selected concealment site ID
    """
    # Build base priors
    priors = {s["id"]: s["prior"] for s in L_CON["site_types"]}
    
    # Apply motive couplings if specified
    if motive and "motive_couplings" in L_CON:
        couplings = L_CON["motive_couplings"].get(motive, {})
        for site_id, multiplier in couplings.items():
            if site_id in priors:
                priors[site_id] *= multiplier
    
    return _weighted_choice(priors)


def sample_time_window(motive: Optional[str] = None, lure: Optional[str] = None) -> str:
    """
    Sample a time window pattern, optionally respecting motive and lure biases.
    
    Args:
        motive: Optional motive ID to apply bias
        lure: Optional lure ID to apply bias
        
    Returns:
        Selected time window ID
    """
    # Build base priors (uniform for now, or could use equal weights)
    priors = {w["id"]: 1.0 for w in L_TIM["windows"]}
    
    # Apply motive bias if specified
    if motive and "motive_bias" in L_TIM:
        motive_bias = L_TIM["motive_bias"].get(motive, {})
        for window_id, multiplier in motive_bias.items():
            if window_id in priors:
                priors[window_id] *= multiplier
    
    # Apply lure bias if specified
    if lure and "lure_bias" in L_TIM:
        lure_bias = L_TIM["lure_bias"].get(lure, {})
        for window_id, multiplier in lure_bias.items():
            if window_id in priors:
                priors[window_id] *= multiplier
    
    return _weighted_choice(priors)


def get_movement_params(profile_id: str) -> Dict[str, Any]:
    """
    Get movement parameters for a given profile ID.
    
    Args:
        profile_id: Movement profile ID
        
    Returns:
        Dictionary with movement parameters:
        - comfort_radius_miles: tuple (min, max) or float
        - beta_cost: float
        - beta_seclusion: float
        - steps: tuple (min, max) or int
        - seclusion_bias: list of site types
        - familiarity_prob: float
    """
    profile = next((p for p in L_MOV["profiles"] if p["id"] == profile_id), None)
    if not profile:
        raise ValueError(f"Unknown movement profile: {profile_id}")
    
    return {
        "comfort_radius_miles": profile.get("comfort_radius_miles"),
        "beta_cost": profile.get("beta_cost"),
        "beta_seclusion": profile.get("beta_seclusion"),
        "steps": profile.get("steps"),
        "seclusion_bias": profile.get("seclusion_bias", []),
        "familiarity_prob": profile.get("familiarity_prob")
    }


def get_time_window_hours(window_id: str) -> Tuple[int, int]:
    """
    Get the hour range for a time window.
    
    Args:
        window_id: Time window ID
        
    Returns:
        Tuple of (start_hour, end_hour)
    """
    window = next((w for w in L_TIM["windows"] if w["id"] == window_id), None)
    if not window:
        raise ValueError(f"Unknown time window: {window_id}")
    
    return tuple(window["hours"])


def get_concealment_radius_hint(site_id: str) -> Optional[Tuple[float, float]]:
    """
    Get radius hint for a concealment site type.
    
    Args:
        site_id: Concealment site ID
        
    Returns:
        Tuple of (min_radius_mi, max_radius_mi) or None if not specified
    """
    site = next((s for s in L_CON["site_types"] if s["id"] == site_id), None)
    if not site:
        return None
    
    radius_hint = site.get("radius_hint_mi")
    if radius_hint:
        return tuple(radius_hint)
    return None
