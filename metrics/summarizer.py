"""Summarizer metrics for case summary generation.

Evaluates summary quality using ROUGE scores and bullet point count accuracy.
"""
from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime
from .config import load_config
from .io import read_json_blocks


def _tok(s: str) -> list[str]:
    """Tokenize string into words.
    
    Args:
        s: Input string.
        
    Returns:
        List of lowercase word tokens.
    """
    return re.findall(r"\w+", (s or "").lower())


def _bigrams(toks: list[str]) -> list[tuple[str, str]]:
    """Generate bigrams from token list.
    
    Args:
        toks: List of word tokens.
        
    Returns:
        List of bigram tuples (word pairs).
    """
    return list(zip(toks, toks[1:])) if len(toks) > 1 else []


def _fscore(ref: list[str], hyp: list[str]) -> float:
    """Calculate F1 score between reference and hypothesis token lists.
    
    Args:
        ref: Reference token list.
        hyp: Hypothesis token list.
        
    Returns:
        F1 score (harmonic mean of precision and recall).
    """
    from collections import Counter

    cr, ch = Counter(ref), Counter(hyp)
    inter = sum((cr & ch).values())
    p = inter / (sum(ch.values()) or 1)
    r = inter / (sum(cr.values()) or 1)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def calc_summarizer_metrics(cfg: dict | None = None) -> dict:
    """Calculate summarizer metrics for case summary generation.
    
    Evaluates LLM-generated summaries against ground truth GRD case narratives
    using ROUGE-1, ROUGE-2, and bullet point count accuracy.
    
    Args:
        cfg: Optional configuration dictionary. If None, loads default config.
        
    Returns:
        dict: Dictionary containing timestamp, stage, metrics (bullet count
            accuracy, ROUGE-1, ROUGE-2), and warnings list.
    """
    cfg = cfg or load_config()
    out = {
        "timestamp": datetime.now().isoformat(),
        "stage": "summarizer",
        "metrics": {},
        "warnings": [],
    }

    # Load LLM results (predictions)
    llm_results_path = pathlib.Path(cfg["paths"]["llm_results"])
    if not llm_results_path.exists():
        out["warnings"].append(f"LLM results file not found: {llm_results_path}")
        out["metrics"].update(
            {
                "bullet_count_accuracy": None,
                "rouge_1": None,
                "rouge_2": None,
            }
        )
        return out

    # Load LLM results
    with open(llm_results_path, "r", encoding="utf-8") as f:
        llm_results = json.load(f)

    # Load GRD cases for reference summaries (ground truth)
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

    # Extract summaries from LLM results
    preds = {}
    for result in llm_results:
        cid = result.get("case_id")
        if not cid:
            continue
        llm_summary = result.get("llm_results", {}).get("summary", {})
        if "text" in llm_summary:
            preds[cid] = {"summary": llm_summary["text"]}

    bullet_counts = []
    rouge1, rouge2 = [], []

    summaries_with_bullets = 0
    for cid, row in preds.items():
        s = row.get("summary") or ""
        if s:
            summaries_with_bullets += 1
            b = sum(
                1
                for ln in s.splitlines()
                if ln.strip().startswith(("-", "•"))
            )
            bullet_counts.append(b)

            g_case = gold_cases.get(cid, {})
            narrative_osint = g_case.get("narrative_osint", {})
            ref = (
                narrative_osint.get("amber_alert")
                or narrative_osint.get("narrative")
                or g_case.get("narrative")
                or ""
            )

            if ref:
                ref_t, hyp_t = _tok(ref), _tok(s)
                rouge1.append(_fscore(ref_t, hyp_t))
                rouge2.append(_fscore(_bigrams(ref_t), _bigrams(hyp_t)))

    out["metrics"]["bullet_count_accuracy"] = (
        (sum(1 for b in bullet_counts if b == 5) / summaries_with_bullets)
        if summaries_with_bullets > 0
        else None
    )
    out["metrics"]["rouge_1"] = (sum(rouge1) / len(rouge1)) if rouge1 else None
    out["metrics"]["rouge_2"] = (sum(rouge2) / len(rouge2)) if rouge2 else None
    if not bullet_counts:
        out["warnings"].append("No summaries found in llm_results")
    return out

