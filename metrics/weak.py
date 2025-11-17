"""Weak labeler metrics for movement profile classification.

Evaluates classification accuracy, confusion matrix, F1 scores, and
calibration metrics for movement profile prediction tasks.
"""
from __future__ import annotations

import statistics as s
from datetime import datetime
from .config import load_config
from .io import read_json_blocks

CLS_MAP = {
    "stationary": "stationary",
    "walking": "walking",
    "driving": "driving",
    "public transit": "public transit",
    "transit": "public transit",
    "bus": "public transit",
    "metro": "public transit",
    "unknown": "unknown",
}
CLASSES = ["stationary", "walking", "driving", "public transit", "unknown"]


def _norm_cls(x: str) -> str:
    """Normalize movement class string to standard format.
    
    Args:
        x: Raw movement class string.
        
    Returns:
        Normalized class string (one of: stationary, walking, driving,
        public transit, unknown).
    """
    if not x:
        return "unknown"
    x = x.strip().lower().replace("_", " ")
    return CLS_MAP.get(x, "unknown")


def calc_weak_labeler_metrics(cfg: dict | None = None) -> dict:
    """Calculate weak labeler metrics for movement profile classification.
    
    Evaluates LLM predictions against ground truth GRD cases for movement
    profile classification accuracy, confusion matrix, F1 scores, Brier score,
    and AUC.
    
    Args:
        cfg: Optional configuration dictionary. If None, loads default config.
        
    Returns:
        dict: Dictionary containing timestamp, stage, metrics (classification
            accuracy, confusion matrix, macro F1, Brier score, AUC), and
            warnings list.
    """
    cfg = cfg or load_config()
    out = {
        "timestamp": datetime.now().isoformat(),
        "stage": "weak",
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
                "classification_accuracy": None,
                "confusion_matrix": None,
                "macro_f1": None,
                "brier_score": None,
                "auc": None,
                "n_pairs_auc": None,
            }
        )
        return out

    # Load LLM results
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
                "classification_accuracy": None,
                "confusion_matrix": None,
                "macro_f1": None,
                "brier_score": None,
                "auc": None,
                "n_pairs_auc": None,
            }
        )
        return out

    # Extract predictions from LLM results
    pred = {}
    for result in llm_results:
        cid = result.get("case_id")
        if not cid:
            continue
        llm_labels = result.get("llm_results", {}).get("labels", {})
        if "data" in llm_labels:
            pred[cid] = llm_labels["data"]

    cm = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
    total = correct = 0
    probs, labels = [], []

    for cid, g_case in gold_cases.items():
        p = pred.get(cid, {})
        if not p:
            continue

        # Get ground truth movement_profile from GRD case
        gt_raw = (g_case.get("movement_profile") or "").strip()
        pr_raw = (p.get("movement") or "").strip()
        gt = _norm_cls(gt_raw)
        pr = _norm_cls(pr_raw)
        if gt in CLASSES and pr in CLASSES:
            cm[gt][pr] += 1
            total += 1
            if gt == pr:
                correct += 1

        rs = p.get("risk_score")
        if isinstance(rs, (int, float)):
            if rs > 1.0:
                rs = rs / 100.0
            rs = max(0.0, min(1.0, rs))
        else:
            risk_str = (p.get("risk") or "").strip().lower()
            rs = 1.0 if risk_str == "critical" else 0.0
        probs.append(rs)
        gt_risk = (g_case.get("risk") or "").strip().lower()
        labels.append(1 if gt_risk == "critical" else 0)

    def macro_f1(cm: dict) -> float | None:
        """Calculate macro-averaged F1 score from confusion matrix.
        
        Args:
            cm: Confusion matrix dictionary mapping true classes to predicted
                class counts.
                
        Returns:
            Macro-averaged F1 score or None if no valid classes.
        """
        f1s = []
        for c in CLASSES:
            tp = cm[c][c]
            fp = sum(cm[r][c] for r in CLASSES if r != c)
            fn = sum(cm[c][r] for r in CLASSES if r != c)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            f1s.append(f)
        return float(s.fmean(f1s)) if f1s else None

    brier = (
        float(s.fmean([(p - y) ** 2 for p, y in zip(probs, labels)]))
        if probs
        else None
    )
    auc = None
    n_pairs = None
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if pos and neg:
        wins = ties = 0
        for pp in pos:
            for nn in neg:
                if pp > nn:
                    wins += 1
                elif pp == nn:
                    ties += 1
        n_pairs = len(pos) * len(neg)
        auc = (wins + 0.5 * ties) / n_pairs

    out["metrics"] = {
        "classification_accuracy": (correct / total) if total else None,
        "confusion_matrix": cm,
        "macro_f1": macro_f1(cm),
        "brier_score": brier,
        "auc": auc,
        "n_pairs_auc": n_pairs,
    }
    if total == 0:
        out["warnings"].append("No comparable movement labels in preds/gold.")
    return out

