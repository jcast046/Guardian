#!/usr/bin/env python3
"""Predictive consistency evaluation script.

Evaluates consistency of forecasts across time horizons and parameter variations.
Computes Jaccard overlap, Spearman rank correlation, and KL divergence between
forecast distributions at different horizons.

Author: Joshua Castillo

Example:
    python -m metrics.evaluation.predictive_consistency --cases data/synthetic_cases --outdir eda_out
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_here = Path(__file__).resolve()
_proj_root = _here.parents[2]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from reinforcement_learning.forecast_api import forecast_timeline


def compute_topk_jaccard(p1: np.ndarray, p2: np.ndarray, k: int = 100) -> float:
    """Compute Jaccard overlap of top-K grid cell indices.
    
    Args:
        p1: Probability distribution array (shape: (N,)).
        p2: Probability distribution array (shape: (N,)).
        k: Number of top cells to consider.
        
    Returns:
        Jaccard overlap coefficient (float in [0, 1]).
    """
    # Get top-K indices for each distribution
    topk1 = np.argsort(p1)[-k:]
    topk2 = np.argsort(p2)[-k:]
    
    # Compute Jaccard: |intersection| / |union|
    intersection = len(np.intersect1d(topk1, topk2))
    union = len(np.union1d(topk1, topk2))
    
    return intersection / union if union > 0 else 0.0


def compute_spearman_rank_correlation(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman rank correlation between probability distributions.
    
    Args:
        p1: Probability distribution array (shape: (N,)).
        p2: Probability distribution array (shape: (N,)).
        
    Returns:
        Tuple of (correlation_coefficient, p_value).
    """
    correlation, p_value = spearmanr(p1, p2)
    
    # Handle NaN values (can occur if all values are identical)
    if np.isnan(correlation):
        correlation = 0.0
    if np.isnan(p_value):
        p_value = 1.0
    
    return float(correlation), float(p_value)


def compute_kl_divergence(p1: np.ndarray, p2: np.ndarray, epsilon: float = 1e-9) -> float:
    """Compute KL divergence between two probability distributions.
    
    Args:
        p1: Probability distribution array (shape: (N,)) - reference distribution.
        p2: Probability distribution array (shape: (N,)) - approximate distribution.
        epsilon: Small value to prevent log(0).
        
    Returns:
        KL divergence value (float, >= 0).
    """
    # Ensure distributions are normalized
    p1_norm = p1 / (p1.sum() + epsilon)
    p2_norm = p2 / (p2.sum() + epsilon)
    
    # Avoid log(0) by adding epsilon
    p2_norm = np.clip(p2_norm, epsilon, None)
    
    # Compute KL divergence: sum(p1 * log(p1 / p2))
    kl = np.sum(p1_norm * np.log((p1_norm + epsilon) / p2_norm))
    
    # KL divergence should be non-negative (handles numerical issues)
    return max(0.0, float(kl))


def evaluate_case_consistency(
    case: Dict[str, Any],
    horizons: List[int] = [24, 48, 72],
    top_k: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """Evaluate consistency metrics for a single case across horizons.
    
    Args:
        case: Case dictionary.
        horizons: List of time horizons in hours.
        top_k: Number of top cells for Jaccard computation.
        **kwargs: Additional arguments passed to forecast_timeline().
        
    Returns:
        Dictionary with consistency metrics per horizon pair.
    """
    try:
        # Get forecasts for all horizons
        forecasts = forecast_timeline(case, horizons=tuple(horizons), **kwargs)
    except Exception as e:
        # Handle cases that fail (missing coordinates, etc.)
        return {"error": str(e)}
    
    results = {}
    
    # Evaluate each horizon pair
    for i in range(len(horizons) - 1):
        h1 = horizons[i]
        h2 = horizons[i + 1]
        pair_key = f"{h1}-{h2}"
        
        p1 = forecasts[h1]
        p2 = forecasts[h2]
        
        # Compute metrics
        jaccard = compute_topk_jaccard(p1, p2, k=top_k)
        spearman_corr, spearman_p = compute_spearman_rank_correlation(p1, p2)
        kl = compute_kl_divergence(p1, p2)
        
        results[pair_key] = {
            "jaccard_overlap": jaccard,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
            "kl_divergence": kl,
        }
    
    return results


def evaluate_consistency_under_parameter_nudge(
    case: Dict[str, Any],
    base_params: Dict[str, Any],
    nudge_params: Dict[str, Any],
    horizons: List[int] = [24, 48, 72],
    top_k: int = 100
) -> Dict[str, Any]:
    """Evaluate consistency when parameters are slightly changed.
    
    Args:
        case: Case dictionary.
        base_params: Base parameter dictionary.
        nudge_params: Nudged parameter dictionary (same keys as base_params).
        horizons: List of time horizons in hours.
        top_k: Number of top cells for Jaccard computation.
        
    Returns:
        Dictionary with comparison metrics.
    """
    # Run forecasts with base and nudged parameters
    try:
        forecasts_base = forecast_timeline(case, horizons=tuple(horizons), **base_params)
        forecasts_nudge = forecast_timeline(case, horizons=tuple(horizons), **nudge_params)
    except Exception as e:
        return {"error": str(e)}
    
    results = {
        "base_params": base_params,
        "nudge_params": nudge_params,
        "horizon_comparisons": {},
    }
    
    # Compare forecasts at each horizon
    for horizon in horizons:
        p_base = forecasts_base[horizon]
        p_nudge = forecasts_nudge[horizon]
        
        jaccard = compute_topk_jaccard(p_base, p_nudge, k=top_k)
        spearman_corr, _ = compute_spearman_rank_correlation(p_base, p_nudge)
        
        results["horizon_comparisons"][str(horizon)] = {
            "jaccard_overlap": jaccard,
            "spearman_correlation": spearman_corr,
        }
    
    # Aggregate across horizons (mean)
    all_jaccards = [v["jaccard_overlap"] for v in results["horizon_comparisons"].values()]
    all_spearmans = [v["spearman_correlation"] for v in results["horizon_comparisons"].values()]
    
    results["aggregate"] = {
        "mean_jaccard_overlap": np.mean(all_jaccards),
        "std_jaccard_overlap": np.std(all_jaccards),
        "mean_spearman_correlation": np.mean(all_spearmans),
        "std_spearman_correlation": np.std(all_spearmans),
    }
    
    return results


def evaluate_all_cases(
    case_files_dir: str,
    horizons: List[int] = [24, 48, 72],
    top_k: int = 100,
    nudge_test: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Evaluate consistency metrics across all cases.
    
    Args:
        case_files_dir: Path to directory containing case JSON files.
        horizons: List of time horizons in hours.
        top_k: Number of top cells for Jaccard computation.
        nudge_test: Whether to run parameter nudge sensitivity tests.
        **kwargs: Additional arguments passed to forecast_timeline().
        
    Returns:
        Dictionary with aggregated consistency metrics.
    """
    case_dir = Path(case_files_dir)
    if not case_dir.exists():
        raise ValueError(f"Case directory not found: {case_files_dir}")
    
    json_files = list(case_dir.glob("*.json"))
    print(f"[INFO] Evaluating {len(json_files)} cases...")
    
    # Collect metrics per horizon pair
    horizon_pairs = [f"{horizons[i]}-{horizons[i+1]}" for i in range(len(horizons) - 1)]
    
    jaccard_values = {pair: [] for pair in horizon_pairs}
    spearman_values = {pair: [] for pair in horizon_pairs}
    kl_values = {pair: [] for pair in horizon_pairs}
    
    cases_processed = 0
    cases_failed = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                case = json.load(f)
            
            case_results = evaluate_case_consistency(
                case, horizons=horizons, top_k=top_k, **kwargs
            )
            
            if "error" in case_results:
                cases_failed += 1
                continue
            
            # Aggregate metrics
            for pair_key in horizon_pairs:
                if pair_key in case_results:
                    jaccard_values[pair_key].append(case_results[pair_key]["jaccard_overlap"])
                    spearman_values[pair_key].append(case_results[pair_key]["spearman_correlation"])
                    kl_values[pair_key].append(case_results[pair_key]["kl_divergence"])
            
            cases_processed += 1
            
            if cases_processed % 10 == 0:
                print(f"  Processed {cases_processed}/{len(json_files)} cases...")
        
        except Exception as e:
            cases_failed += 1
            continue
    
    print(f"[INFO] Processed {cases_processed} cases, {cases_failed} failed")
    
    # Aggregate results
    results = {
        "n_cases": cases_processed,
        "horizon_pairs": {},
    }
    
    for pair_key in horizon_pairs:
        results["horizon_pairs"][pair_key] = {
            "jaccard_overlap": {
                "mean": np.mean(jaccard_values[pair_key]) if jaccard_values[pair_key] else None,
                "std": np.std(jaccard_values[pair_key]) if jaccard_values[pair_key] else None,
                "values": jaccard_values[pair_key],
            },
            "spearman_correlation": {
                "mean": np.mean(spearman_values[pair_key]) if spearman_values[pair_key] else None,
                "std": np.std(spearman_values[pair_key]) if spearman_values[pair_key] else None,
                "values": spearman_values[pair_key],
            },
            "kl_divergence": {
                "mean": np.mean(kl_values[pair_key]) if kl_values[pair_key] else None,
                "std": np.std(kl_values[pair_key]) if kl_values[pair_key] else None,
                "values": kl_values[pair_key],
            },
        }
    
    # Parameter sensitivity testing (optional)
    if nudge_test:
        print("[INFO] Running parameter sensitivity tests...")
        
        # Test with a sample case (first valid case)
        sample_case = None
        for json_file in json_files[:10]:  # Try first 10 cases
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    sample_case = json.load(f)
                
                # Quick validation - check for coordinates
                spatial = sample_case.get("spatial", {})
                if spatial.get("last_seen_lat") and spatial.get("last_seen_lon"):
                    break
            except:
                continue
        
        if sample_case:
            base_params = {
                "alpha_prior": kwargs.get("alpha_prior", 0.5),
                "steps_per_24h": kwargs.get("steps_per_24h", 3),
                "profile": kwargs.get("profile", "default"),
            }
            
            sensitivity_results = {}
            
            # Test alpha_prior nudge
            nudge_alpha = base_params.copy()
            nudge_alpha["alpha_prior"] = base_params["alpha_prior"] + 0.05
            result_alpha = evaluate_consistency_under_parameter_nudge(
                sample_case, base_params, nudge_alpha, horizons=horizons, top_k=top_k
            )
            if "error" not in result_alpha:
                sensitivity_results["alpha_prior"] = result_alpha["aggregate"]
            
            # Test steps_per_24h nudge
            nudge_steps = base_params.copy()
            nudge_steps["steps_per_24h"] = base_params["steps_per_24h"] + 1
            result_steps = evaluate_consistency_under_parameter_nudge(
                sample_case, base_params, nudge_steps, horizons=horizons, top_k=top_k
            )
            if "error" not in result_steps:
                sensitivity_results["steps_per_24h"] = result_steps["aggregate"]
            
            results["parameter_sensitivity"] = sensitivity_results
    
    return results


def plot_consistency_results(results: Dict[str, Any], outdir: str) -> None:
    """Generate plots for consistency metrics.
    
    Args:
        results: Results dictionary from evaluate_all_cases().
        outdir: Output directory for plots.
    """
    out_path = Path(outdir)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    horizon_pairs = list(results.get("horizon_pairs", {}).keys())
    
    if not horizon_pairs:
        print("[WARN] No horizon pairs found in results, skipping plots")
        return
    
    # Plot 1: Jaccard overlap vs horizon pair
    fig, ax = plt.subplots(figsize=(10, 6))
    
    jaccard_means = []
    jaccard_stds = []
    
    for pair_key in horizon_pairs:
        pair_data = results["horizon_pairs"][pair_key]["jaccard_overlap"]
        if pair_data["mean"] is not None:
            jaccard_means.append(pair_data["mean"])
            jaccard_stds.append(pair_data["std"] or 0.0)
        else:
            jaccard_means.append(0.0)
            jaccard_stds.append(0.0)
    
    x_pos = np.arange(len(horizon_pairs))
    ax.bar(x_pos, jaccard_means, yerr=jaccard_stds, capsize=5, alpha=0.7)
    ax.set_xlabel("Horizon Pair")
    ax.set_ylabel("Mean Jaccard Overlap")
    ax.set_title("Predictive Consistency: Jaccard Overlap by Horizon Pair")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(horizon_pairs)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "predictive_consistency_jaccard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved Jaccard plot: {plots_dir / 'predictive_consistency_jaccard.png'}")
    
    # Plot 2: Spearman correlation vs horizon pair
    fig, ax = plt.subplots(figsize=(10, 6))
    
    spearman_means = []
    spearman_stds = []
    
    for pair_key in horizon_pairs:
        pair_data = results["horizon_pairs"][pair_key]["spearman_correlation"]
        if pair_data["mean"] is not None:
            spearman_means.append(pair_data["mean"])
            spearman_stds.append(pair_data["std"] or 0.0)
        else:
            spearman_means.append(0.0)
            spearman_stds.append(0.0)
    
    x_pos = np.arange(len(horizon_pairs))
    ax.bar(x_pos, spearman_means, yerr=spearman_stds, capsize=5, alpha=0.7)
    ax.set_xlabel("Horizon Pair")
    ax.set_ylabel("Mean Spearman Rank Correlation")
    ax.set_title("Predictive Consistency: Spearman Correlation by Horizon Pair")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(horizon_pairs)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "predictive_consistency_spearman.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved Spearman plot: {plots_dir / 'predictive_consistency_spearman.png'}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate predictive consistency across horizons and parameters"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="data/synthetic_cases",
        help="Path to directory containing case JSON files (default: data/synthetic_cases)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="eda_out",
        help="Output directory for results (default: eda_out)"
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[24, 48, 72],
        help="Time horizons in hours (default: 24 48 72)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top cells for Jaccard computation (default: 100)"
    )
    parser.add_argument(
        "--no-nudge-test",
        action="store_true",
        help="Skip parameter sensitivity testing"
    )
    parser.add_argument(
        "--alpha-prior",
        type=float,
        default=0.5,
        help="Mixing weight for KDE prior (default: 0.5)"
    )
    parser.add_argument(
        "--steps-per-24h",
        type=int,
        default=3,
        help="Number of Markov steps per 24 hours (default: 3)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Survival profile type (default: default)"
    )
    
    args = parser.parse_args()
    
    # Prepare forecast parameters
    forecast_kwargs = {
        "alpha_prior": args.alpha_prior,
        "steps_per_24h": args.steps_per_24h,
        "profile": args.profile,
    }
    
    # Evaluate all cases
    results = evaluate_all_cases(
        args.cases,
        horizons=args.horizons,
        top_k=args.top_k,
        nudge_test=not args.no_nudge_test,
        **forecast_kwargs
    )
    
    # Add metadata
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_cases": results["n_cases"],
        "horizons": args.horizons,
        "top_k": args.top_k,
        "forecast_parameters": forecast_kwargs,
        "horizon_pairs": results["horizon_pairs"],
    }
    
    if "parameter_sensitivity" in results:
        output["parameter_sensitivity"] = results["parameter_sensitivity"]
    
    # Save JSON results
    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "predictive_consistency.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    output_serializable = convert_to_json_serializable(output)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Saved results to {json_path}")
    
    # Generate plots
    plot_consistency_results(results, args.outdir)
    
    print("\n[SUMMARY] Predictive Consistency Evaluation Complete")
    print(f"  Cases processed: {results['n_cases']}")
    print(f"  Horizons: {args.horizons}")
    print(f"  Results saved to: {json_path}")


if __name__ == "__main__":
    main()

