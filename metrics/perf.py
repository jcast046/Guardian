"""Performance metrics aggregator for pipeline evaluation.

Combines metrics from multiple stages to compute aggregate performance
indicators including ROUGE scores and zone plausibility.
"""
from __future__ import annotations

from datetime import datetime
from .config import load_config
from .io import read_json_blocks
from .summarizer import calc_summarizer_metrics
from .rl import calc_rl_metrics


def calc_perf_metrics(cfg: dict | None = None) -> dict:
    """Calculate aggregate performance metrics across pipeline stages.
    
    Combines metrics from summarizer and RL stages to compute mean ROUGE
    and average zone plausibility scores.
    
    Args:
        cfg: Optional configuration dictionary. If None, loads default config.
        
    Returns:
        dict: Dictionary containing timestamp, stage, metrics (mean ROUGE,
            KL divergence placeholders, average zone plausibility), and
            warnings list.
    """
    cfg = cfg or load_config()
    out = {
        "timestamp": datetime.now().isoformat(),
        "stage": "perf",
        "metrics": {},
        "warnings": [],
    }

    # Get metrics from other stages
    summarizer_metrics = calc_summarizer_metrics(cfg)
    rl_metrics = calc_rl_metrics("llm", cfg)

    if (
        summarizer_metrics["metrics"].get("rouge_1")
        and summarizer_metrics["metrics"].get("rouge_2")
    ):
        out["metrics"]["mean_rouge"] = (
            summarizer_metrics["metrics"]["rouge_1"]
            + summarizer_metrics["metrics"]["rouge_2"]
        ) / 2
    else:
        out["metrics"]["mean_rouge"] = None

    out["metrics"]["kl_divergence_age"] = None
    out["metrics"]["kl_divergence_county"] = None
    out["warnings"].append("KL divergence requires real-world statistics")

    import pathlib

    zr_path = "eda_out/zones_review.jsonl"
    zr = read_json_blocks(zr_path)
    if zr:
        vals = []
        for r in zr:
            zones = r.get("zones", [])
            if isinstance(zones, list):
                for z in zones:
                    s = z.get("plausibility")
                    if isinstance(s, (int, float)):
                        vals.append(float(s))
            elif isinstance(zones, dict):
                for window_zones in zones.values():
                    if isinstance(window_zones, list):
                        for z in window_zones:
                            s = z.get("plausibility")
                            if isinstance(s, (int, float)):
                                vals.append(float(s))
        out["metrics"]["avg_zone_plausibility"] = (
            (sum(vals) / len(vals)) if vals else None
        )
    else:
        out["metrics"]["avg_zone_plausibility"] = None
    return out

