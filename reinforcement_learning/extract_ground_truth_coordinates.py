#!/usr/bin/env python3
"""Extract coordinates from case files to generate coordinate-based ground truth.

Reads case files and extracts coordinates (recovery or last_seen) to create
a coordinate-based ground_truth.json file for evaluation.
"""

import json
import pathlib
import sys
from pathlib import Path

# Add project root to path
_here = Path(__file__).resolve()
_proj_root = _here.parents[1]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))


def extract_coordinates_from_case(case_file: Path) -> dict | None:
    """Extract coordinates from a case file.
    
    Args:
        case_file: Path to case JSON file
        
    Returns:
        Dictionary with "lat" and "lon" keys, or None if coordinates not found
    """
    try:
        with open(case_file, 'r', encoding='utf-8') as f:
            case = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load case file {case_file}: {e}")
        return None
    
    case_id = case.get("case_id")
    if not case_id:
        print(f"[WARN] Case file {case_file} missing case_id")
        return None
    
    # Priority 1: Recovery coordinates
    outcome = case.get("outcome", {})
    recovery_lat = outcome.get("recovery_lat")
    recovery_lon = outcome.get("recovery_lon")
    
    if recovery_lat is not None and recovery_lon is not None:
        try:
            return {
                "lat": float(recovery_lat),
                "lon": float(recovery_lon),
                "source": "recovery"
            }
        except (ValueError, TypeError):
            pass
    
    # Priority 2: Last seen coordinates
    spatial = case.get("spatial", {})
    last_seen_lat = spatial.get("last_seen_lat")
    last_seen_lon = spatial.get("last_seen_lon")
    
    if last_seen_lat is not None and last_seen_lon is not None:
        try:
            return {
                "lat": float(last_seen_lat),
                "lon": float(last_seen_lon),
                "source": "last_seen"
            }
        except (ValueError, TypeError):
            pass
    
    print(f"[WARN] Case {case_id} has no valid coordinates (recovery or last_seen)")
    return None


def generate_ground_truth_json(cases_dir: str, output_path: str, case_ids: list[str] | None = None) -> dict:
    """Generate coordinate-based ground truth JSON from case files.
    
    Args:
        cases_dir: Directory containing case JSON files
        output_path: Path to output ground_truth.json file
        case_ids: Optional list of case IDs to extract (if None, extracts all cases)
        
    Returns:
        Dictionary mapping case_id to coordinate dict
    """
    cases_path = Path(cases_dir)
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")
    
    # Find all case files
    case_files = list(cases_path.glob("GRD-*.json"))
    if not case_files:
        raise FileNotFoundError(f"No case files found in {cases_dir}")
    
    ground_truth = {}
    skipped = 0
    
    for case_file in sorted(case_files):
        case_id = case_file.stem
        if case_ids and case_id not in case_ids:
            continue
        
        coords = extract_coordinates_from_case(case_file)
        if coords:
            ground_truth[case_id] = {
                "lat": coords["lat"],
                "lon": coords["lon"]
            }
        else:
            skipped += 1
    
    output_file = Path(output_path)
    if not output_file.is_absolute():
        if "/" not in output_path and "\\" not in output_path:
            # Just a filename - put in same directory as script
            output_file = _here.parent / output_path
        else:
            # Has directory components - resolve relative to project root
            output_file = _proj_root / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Extracted coordinates for {len(ground_truth)} cases")
    print(f"[INFO] Skipped {skipped} cases without coordinates")
    print(f"[INFO] Output written to {output_path}")
    
    return ground_truth


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract coordinates from case files for ground truth")
    parser.add_argument("--cases-dir", type=str, default="data/synthetic_cases",
                       help="Directory containing case JSON files")
    parser.add_argument("--output", type=str, default="ground_truth.json",
                       help="Output path for ground_truth.json (relative to script directory or absolute)")
    parser.add_argument("--case-ids", type=str, nargs="*",
                       help="Optional list of case IDs to extract (if not provided, extracts all)")
    
    args = parser.parse_args()
    
    generate_ground_truth_json(args.cases_dir, args.output, args.case_ids)

