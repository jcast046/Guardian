"""Configuration management for metrics calculation.

Provides default configuration paths and loading utilities for metrics
calculation across different pipeline stages.
"""
from __future__ import annotations

import json
import pathlib

_DEFAULT = {
    "paths": {
        "eda_min": "eda_out/eda_cases_min.jsonl",
        "gold_cases": "gold/cases_gold.jsonl",
        "gold_zones": "gold/zone_truth.jsonl",
        "synthetic_cases": "data/synthetic_cases/",
        "llm_results": "gold/llm_analysis_results.json",
        "real_cases": "data/real_cases/guardian_output.jsonl",
        "zones_baseline": "eda_out/zones_rl.jsonl",
        "zones_llm": "eda_out/zones_reweighted.jsonl",
        "va_boundary": "data/geo/va_boundary.geojson",
        "real_min": "data/real_cases/guardian_output.jsonl"
    },
    "ops": {
        "llm_timings": "gold/llm_analysis_results.json",
        "validation_report": "eda_out/validation_report.json",
        "expect_outputs": [
            "eda_out/distribution_summary.png",
            "eda_out/age_hist.png",
            "eda_out/gender_bar.png",
            "eda_out/maps_dark/kde_all.png",
            "eda_out/zones_review.jsonl"
        ]
    },
    "rl": {"ks": [1, 3, 5, 10]},
    "geo": {"hit_buffer_m": 0}
}


def load_config(path: str | None = "metrics/metrics_config.json") -> dict:
    """Load metrics configuration from JSON file.
    
    Loads user configuration file and merges with default configuration.
    User values override defaults for matching keys.
    
    Args:
        path: Path to configuration JSON file. If None or file doesn't exist,
            returns default configuration.
            
    Returns:
        dict: Merged configuration dictionary.
    """
    p = pathlib.Path(path) if path else None
    if p and p.exists():
        with p.open("r", encoding="utf-8") as f:
            user = json.load(f)
        merged = _DEFAULT.copy()
        for k, v in user.items():
            if isinstance(v, dict) and k in merged:
                merged[k].update(v)
            else:
                merged[k] = v
        return merged
    return _DEFAULT

