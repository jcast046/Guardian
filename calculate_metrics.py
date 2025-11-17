#!/usr/bin/env python3
"""Comprehensive metrics calculation system for Guardian pipeline.

Calculates system health and accuracy metrics for the Guardian system.
Supports multiple stages: ops, perf, rl, extractor, weak, summarizer, e2e,
diagnostics, and all.

Examples:
    python calculate_metrics.py ops
    python calculate_metrics.py perf
    python calculate_metrics.py all --zones baseline --output metrics.json
"""

import argparse
import json
import sys
from datetime import datetime

from metrics.config import load_config
from metrics.ops import calc_ops_metrics
from metrics.rl import calc_rl_metrics
from metrics.summarizer import calc_summarizer_metrics
from metrics.extractor import calc_extractor_metrics
from metrics.weak import calc_weak_labeler_metrics
from metrics.perf import calc_perf_metrics
from metrics.diagnostics import calc_diagnostics


def main():
    """Calculate metrics for specified pipeline stages.
    
    Parses command-line arguments, loads configuration, calculates metrics
    for requested stages, and outputs results to stdout or file.
    
    Returns:
        int: Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Guardian Pipeline Metrics Calculator"
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=[
            "ops",
            "perf",
            "rl",
            "extractor",
            "weak",
            "summarizer",
            "e2e",
            "diagnostics",
            "all",
        ],
        default="all",
        help="Metrics stage to calculate (default: all)",
    )
    parser.add_argument(
        "--zones",
        choices=["baseline", "llm"],
        default="baseline",
        help="Zone type for RL metrics (default: baseline)",
    )
    parser.add_argument(
        "--config",
        default="metrics/metrics_config.json",
        help="Path to metrics config file (default: metrics/metrics_config.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: print to stdout)",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    results = {}

    if args.stage == "ops" or args.stage == "all":
        results["ops"] = calc_ops_metrics(cfg)

    if args.stage == "rl" or args.stage == "all":
        results["rl"] = calc_rl_metrics(args.zones, cfg)

    if args.stage == "extractor" or args.stage == "all":
        results["extractor"] = calc_extractor_metrics(cfg)

    if args.stage == "weak" or args.stage == "all":
        results["weak"] = calc_weak_labeler_metrics(cfg)

    if args.stage == "summarizer" or args.stage == "all":
        results["summarizer"] = calc_summarizer_metrics(cfg)

    if args.stage == "perf" or args.stage == "all":
        results["perf"] = calc_perf_metrics(cfg)

    if args.stage == "diagnostics":
        results["diagnostics"] = calc_diagnostics(cfg)
        print(json.dumps(results["diagnostics"], indent=2))
        return

    if args.stage == "e2e":
        rl_metrics = calc_rl_metrics(args.zones, cfg)
        rl_m = rl_metrics["metrics"]
        hit3 = rl_m.get("geo_hit_at_k", {}).get("geo_hit_at_3")
        median_asuh_km2 = rl_m.get("asuh_km2")
        median_dist_to_first_hit_miles = rl_m.get("median_distance_to_first_hit_miles")
        median_dist_to_first_hit_m = (
            median_dist_to_first_hit_miles * 1609.34
            if median_dist_to_first_hit_miles is not None
            else None
        )
        results["e2e"] = {
            "timestamp": datetime.now().isoformat(),
            "stage": "e2e",
            "metrics": {
                "mission_success_top3": hit3,
                "effort_median_asuh_km2": median_asuh_km2,
                "speed_median_dist_to_first_hit_m": median_dist_to_first_hit_m,
            },
            "warnings": rl_metrics["warnings"],
        }

    # Output results
    output_json = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Metrics saved to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    sys.exit(main())
