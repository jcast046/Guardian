"""Operations metrics for pipeline health monitoring.

Evaluates schema validation rates, case generation rates, coordinate
coverage, error rates, and LLM processing times.
"""
from __future__ import annotations

import json
import pathlib
import statistics as stats
from datetime import datetime
from .config import load_config
from .io import read_json_blocks, coord_of

try:
    from shapely.geometry import shape, Point

    SHAPELY = True
except Exception:
    SHAPELY = False


def _load_va_boundary(path: str) -> list:
    """Load Virginia boundary polygons from GeoJSON file.
    
    Args:
        path: Path to GeoJSON file containing Virginia boundary.
        
    Returns:
        List of Shapely geometry objects, or empty list if file not found
        or Shapely unavailable.
    """
    p = pathlib.Path(path)
    if not (SHAPELY and p.exists()):
        return []
    import json

    with p.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features") or []
    return [shape(feat["geometry"]) for feat in feats if feat.get("geometry")]


def calc_ops_metrics(cfg: dict | None = None) -> dict:
    """Calculate operations metrics for pipeline health monitoring.
    
    Evaluates schema validation rates, case generation throughput, coordinate
    coverage, geographic validation (inside VA), pipeline error rates, and
    LLM processing performance.
    
    Args:
        cfg: Optional configuration dictionary. If None, loads default config.
        
    Returns:
        dict: Dictionary containing timestamp, stage, metrics (validation rates,
            generation rates, coordinate coverage, error rates, LLM timings,
            expected outputs), and warnings list.
    """
    cfg = cfg or load_config()
    out = {
        "timestamp": datetime.now().isoformat(),
        "stage": "ops",
        "metrics": {},
        "warnings": [],
    }

    val = pathlib.Path(cfg["ops"]["validation_report"])
    if val.exists():
        try:
            v = json.loads(val.read_text(encoding="utf-8"))
            total = v.get("total_files") or v.get("total")
            valid = v.get("valid_files")
            failing = v.get("failing")
            if total and valid is not None:
                out["metrics"]["schema_validation_rate"] = valid / total
            elif total and failing is not None:
                out["metrics"]["schema_validation_rate"] = (total - failing) / total
            else:
                out["metrics"]["schema_validation_rate"] = None
        except Exception as e:
            out["warnings"].append(f"validation_report parse error: {e}")
            out["metrics"]["schema_validation_rate"] = None
    else:
        out["warnings"].append("validation_report.json not found")
        out["metrics"]["schema_validation_rate"] = None

    synth = pathlib.Path(cfg["paths"]["synthetic_cases"])
    rate = None
    if synth.exists():
        files = list(synth.rglob("GRD-*.json"))
        if len(files) > 1:
            mt = sorted(f.stat().st_mtime for f in files)
            span = mt[-1] - mt[0]
            rate = (len(files) / (span / 60.0)) if span > 0 else None
    else:
        out["warnings"].append("synthetic_cases dir not found")
    out["metrics"]["case_generation_rate_per_min"] = rate

    eda_rows = read_json_blocks(cfg["paths"]["eda_min"])
    gold_rows = (
        read_json_blocks(cfg["paths"]["gold_cases"]) if not eda_rows else []
    )
    rows = eda_rows or gold_rows
    total = len(rows)
    have = 0
    inside = 0
    polys = (
        _load_va_boundary(cfg["paths"]["va_boundary"]) if SHAPELY else []
    )
    for r in rows:
        lat, lon = coord_of(r)
        if lat is not None and lon is not None:
            have += 1
            if polys:
                pt = Point(lon, lat)
                if any(poly.contains(pt) or poly.touches(pt) for poly in polys):
                    inside += 1
    out["metrics"]["geo_valid_coords_rate"] = (have / total) if total else None
    out["metrics"]["geo_inside_va_rate"] = (
        (inside / have) if (have and polys) else None
    )

    errs = sum(1 for r in rows if ("error" in r or "llm_error" in r))
    out["metrics"]["pipeline_error_rate"] = (errs / total) if total else None

    tpath = pathlib.Path(cfg["ops"]["llm_timings"])
    if tpath.exists():
        try:
            t = json.loads(tpath.read_text(encoding="utf-8"))
            if isinstance(t, list) and t:
                times = []
                for result in t:
                    llm_res = result.get("llm_results", {})
                    case_time = 0.0
                    if "summary" in llm_res and "processing_time" in llm_res["summary"]:
                        case_time += llm_res["summary"]["processing_time"]
                    if "entities" in llm_res and "processing_time" in llm_res["entities"]:
                        case_time += llm_res["entities"]["processing_time"]
                    if "labels" in llm_res and "processing_time" in llm_res["labels"]:
                        case_time += llm_res["labels"]["processing_time"]
                    if case_time > 0:
                        times.append(case_time)
                if times:
                    out["metrics"]["llm_avg_processing_sec"] = sum(times) / len(times)
                else:
                    out["metrics"]["llm_avg_processing_sec"] = None
            elif isinstance(t, dict):
                vals = [v for v in t.values() if isinstance(v, (int, float))]
                out["metrics"]["llm_avg_processing_sec"] = (
                    (sum(vals) / len(vals)) if vals else None
                )
        except Exception as e:
            out["warnings"].append(f"llm_analysis_results.json parse error: {e}")
            out["metrics"]["llm_avg_processing_sec"] = None
    else:
        out["warnings"].append(f"llm_analysis_results.json not found: {tpath}")
        out["metrics"]["llm_avg_processing_sec"] = None

    expected = cfg["ops"]["expect_outputs"]
    present = [p for p in expected if pathlib.Path(p).exists()]
    missing = [p for p in expected if p not in present]
    out["metrics"]["expected_outputs_present"] = {
        "total": len(expected),
        "present": len(present),
        "missing": len(missing),
        "present_files": present,
        "missing_files": missing,
    }
    return out

