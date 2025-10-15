"""
Entity extraction using Qwen2.5-3B-Instruct for JSON extraction.

This module provides structured entity extraction capabilities using the Qwen2.5-3B-Instruct
model. It extracts persons, vehicles, locations, timeline events, and evidence from case
narratives and returns them in a structured JSON format.

The extractor is designed to work with the Guardian case management system and provides
high-quality entity extraction for investigative workflows.

Author: Joshua Castillo

Classes:
    None

Functions:
    extract_entities(narrative: str) -> dict: Extract all entities from case narrative
    extract_persons(narrative: str) -> list: Extract only person entities
    extract_vehicles(narrative: str) -> list: Extract only vehicle entities
    extract_locations(narrative: str) -> list: Extract only location entities
    extract_json(text: str, hints: dict | None = None) -> dict: Deterministic scaffold with LLM fill
    batch_extract_json(narratives: list[str], batch_size: int = 2, hints_list: list[dict] | None = None) -> list[dict]: Batch processing
    scaffold_from_narrative(text: str) -> dict: Lightweight deterministic scaffold from structured lines
    merge_keep_existing(base: dict, add: dict) -> dict: Merge dicts preferring existing non-null values
    backfill(entity, narrative): Fast, deterministic backfills for missing fields
    minimal_entities_from_case(case: dict) -> dict: Extract minimal entities for EDA

Example:
    >>> from guardian_llm import extract_entities, extract_persons
    >>> entities = extract_entities("John Smith was seen driving a red Honda...")
    >>> persons = extract_persons("John Smith was seen driving a red Honda...")
"""
import re, json, math, datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .prompts import EXTRACTION_PROMPT

# --- minimal helpers for EDA ---
def _abbr_state(s: str | None) -> str | None:
    """
    Normalize state abbreviations to standard format.
    
    Args:
        s (str | None): State string to normalize
        
    Returns:
        str | None: Normalized state abbreviation or None if input is None/empty
    """
    if not s: return None
    s = s.strip()
    return {"Virginia":"VA","va":"VA","Va":"VA"}.get(s, s)

def _norm_gender_short(g):
    """
    Normalize gender to single letter format.
    
    Args:
        g: Gender string to normalize
        
    Returns:
        str | None: "F" for female, "M" for male, None if unclear
    """
    g = (g or "").strip().lower()
    if g.startswith("f"): return "F"
    if g.startswith("m"): return "M"
    return None

def minimal_entities_from_case(case: dict) -> dict:
    """
    Extract minimal entity information from case data for EDA analysis.
    
    Args:
        case (dict): Case data dictionary with demographic, spatial, and temporal keys
        
    Returns:
        dict: Minimal entity data with age, gender, location, coordinates, and date_reported
    """
    d = (case.get("demographic") or {})
    s = (case.get("spatial") or {})
    t = (case.get("temporal") or {})
    return {
        "age": d.get("age_years"),
        "gender": _norm_gender_short(d.get("gender")),
        "location": {
            "city":   s.get("last_seen_city") or s.get("last_seen_location"),
            "county": s.get("last_seen_county"),
            "state":  _abbr_state(s.get("last_seen_state") or "VA") or "VA",
        },
        "lat": s.get("last_seen_lat"),
        "lon": s.get("last_seen_lon"),
        "date_reported": t.get("reported_missing_ts"),
    }
# --- minimal helpers for EDA ---

# -------- fast patterns tuned to the synthetic format --------
RE_COORDS     = re.compile(r'Coordinates:\s*([+-]?\d{1,2}\.\d+)\s*,\s*([+-]?\d{1,3}\.\d+)', re.I)
RE_LAST_SEEN  = re.compile(r'Last Seen:\s*([^,]+),\s*([^,]+),\s*([A-Za-z]{2,})', re.I)
RE_AGE        = re.compile(r'\b(\d{1,2})-year-old\b', re.I)
RE_GENDER     = re.compile(r'\b(male|female)\b', re.I)
RE_NAME       = re.compile(r'Missing Person:\s*([^,]+)', re.I)
RE_REPORTED   = re.compile(r'(Reported Missing|Date Reported):\s*([0-9]{4}-[0-9]{2}-[0-9]{2}T[^ \n]+)', re.I)
RE_MOVES      = re.compile(r'\b(I-\d{1,3}|US-\d{1,3}|VA-\d{1,3})\b', re.I)

def scaffold_from_narrative(text: str) -> dict:
    """
    Create lightweight deterministic scaffold from structured narrative lines.
    
    This function uses regex patterns to extract basic information from well-structured
    case narratives, providing a foundation for LLM-based extraction.
    
    Args:
        text (str): Case narrative text to parse
        
    Returns:
        dict: Scaffold data with extracted fields and risk factors
    """
    data = {
        "name": None,
        "age": None,
        "gender": None,
        "location": {"city": None, "county": None, "state": "VA"},
        "lat": None,
        "lon": None,
        "date_reported": None
    }

    # Missing Person: Child_6168, 6-year-old female (deterministic parse)
    m = re.search(r"Missing Person:\s*([A-Za-z0-9_]+),\s*(\d{1,2})-year-old\s+(male|female)", text, re.I)
    if m:
        data["name"] = m.group(1).strip()
        data["age"] = int(m.group(2))
        data["gender"] = "F" if m.group(3).lower() == "female" else "M"

    # Last Seen: City, County, State (accepts both Virginia and VA)
    if m := RE_LAST_SEEN.search(text):
        city, county, st = [x.strip() for x in m.groups()]
        data["location"]["city"]   = None if city.lower()=="unknown" else city
        data["location"]["county"] = None if county.lower()=="unknown" else county
        data["location"]["state"]  = "VA" if st.lower()=="virginia" else st

    # Coordinates: 37.08, -76.49
    if m := RE_COORDS.search(text):
        data["lat"], data["lon"] = float(m.group(1)), float(m.group(2))

    # Date reported (ISO) from narrative
    m = re.search(r"Reported Missing:\s*([0-9T:\.\+\-Z]+)", text)
    if m:
        data["date_reported"] = m.group(1)

    # Time-to-report minutes -> risk factor
    late_min = None
    m = re.search(r"Time to Report:\s*(\d+)\s*minutes", text)
    if m:
        late_min = int(m.group(1))

    # highways / routes
    data["movement_cues"] = sorted(set(RE_MOVES.findall(text)))

    # Expand risk_factors beyond weapons/abduction
    risks = []
    # existing signals
    if re.search(r"\bweapon|gun|knife\b", text, re.I):
        risks.append("weapon_mentioned")
    if re.search(r"\babduction|forced\b", text, re.I):
        risks.append("abduction_cue")
    # new signals commonly present in synthetic cases
    if late_min is not None and late_min >= 120:
        risks.append("late_report_2h_plus")
    elif late_min is not None and late_min >= 60:
        risks.append("late_report_1h_plus")
    if re.search(r"(entered|into)\s+vehicle|offered?\s+a\s+ride", text, re.I):
        risks.append("vehicle_lure_or_transport")
    if re.search(r"\b(parent|adult|companion)\b.*(following|exiting with|offering a ride)", text, re.I):
        risks.append("non_family_adult_involved")
    # highway/corridor signal if multiple interstates
    roads = re.findall(r"\bI-\d{1,3}\b", text)
    if len(set(roads)) >= 2:
        risks.append("highway_corridor")
    data["risk_factors"] = list(dict.fromkeys(risks))
    return data

def merge_keep_existing(base: dict, add: dict) -> dict:
    """
    Merge two dictionaries preferring existing non-null values in base.
    
    This function performs a deep merge where existing non-null values in the base
    dictionary are preserved, and only missing or null values are filled from the add dictionary.
    
    Args:
        base (dict): Base dictionary with existing values
        add (dict): Dictionary with additional values to merge
        
    Returns:
        dict: Merged dictionary with preserved existing values
    """
    out = dict(base)
    for k, v in (add or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_keep_existing(out[k], v)
        elif out.get(k) in (None, [], {}):
            out[k] = v
    return out

def backfill(entity, narrative):
    """
    Fast, deterministic backfills for missing entity fields.
    
    This function uses regex patterns to extract missing information from the narrative
    and fill in null or empty fields in the entity dictionary.
    
    Args:
        entity (dict): Entity dictionary to backfill
        narrative (str): Case narrative text to extract from
        
    Returns:
        dict: Updated entity dictionary with backfilled fields
    """
    # date_reported from "Reported Missing: <ISO>"
    m = re.search(r"Reported Missing:\s*([0-9\-:T\.\+Z]+)", narrative)
    if entity.get("date_reported") in (None, "") and m:
        entity["date_reported"] = m.group(1)

    # city/county/state from "Last Seen:" line
    m = re.search(r"Last Seen:\s*([^,\n]+)(?:,\s*([^,\n]+))?,\s*([A-Z]{2})", narrative)
    if m:
        city, maybe_county, state = m.group(1), m.group(2), m.group(3)
        loc = entity.setdefault("location", {})
        loc.setdefault("city", city if city and loc.get("city") in (None, "") else loc.get("city"))
        if maybe_county and loc.get("county") in (None, ""):
            loc["county"] = re.sub(r"\s*County$", "", maybe_county)
        loc.setdefault("state", state or loc.get("state"))

    # risk_factors from keywords
    rf = set(entity.get("risk_factors") or [])
    txt = narrative.lower()
    if any(k in txt for k in ["entered vehicle", "offered a ride", "lured"]): rf.add("Luring/vehicle involvement")
    if any(k in txt for k in ["i-95", "i-64", "us-58", "i-81"]): rf.add("Intercity/Interstate movement")
    if re.search(r"elapsed|time to report:\s*(\d+)\s*minutes", txt):
        match = re.search(r"(\d+)\s*minutes", txt)
        if match and int(match.group(1)) > 120:
            rf.add("Delayed reporting")
    entity["risk_factors"] = sorted(rf)

    return entity

def _generate_json_strict(prompt: str, max_new_tokens: int = 192) -> str:
    """
    Internal generate wrapper for JSON extraction with optimized settings.
    
    This function handles the actual model inference for JSON extraction tasks,
    using optimized generation parameters for consistent JSON output.
    
    Args:
        prompt (str): Input prompt for the model
        max_new_tokens (int): Maximum number of new tokens to generate
        
    Returns:
        str: Generated text from the model
    """
    _ensure_loaded()
    
    enc = _tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    if torch.cuda.is_available():
        ids, mask = ids.to(_mdl.device), mask.to(_mdl.device)

    # Clamp generation 
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=_tok.eos_token_id
    )

    with torch.inference_mode():
        out = _mdl.generate(
            input_ids=ids, attention_mask=mask,
            **gen_kwargs,
            eos_token_id=_tok.eos_token_id,
            use_cache=True
        )

    # decode new tokens only
    gen = out[0, ids.shape[-1]:]
    return _tok.decode(gen, skip_special_tokens=True).strip()

# Enable TF32 for RTX 40xx speedup and advanced optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load configuration from guardian.config.json
try:
    with open("guardian.config.json", "r") as f:
        config = json.load(f)
    DIR_EXTRACTOR = config["models"]["extractor"]
    USE_LLAMA = config.get("use_llama_as_extractor", False)
except:
    DIR_EXTRACTOR = r"C:\Users\N0Cir\CS698\Guardian\models\Qwen2.5-3B-Instruct"
    USE_LLAMA = False

# Global model cache variables
_tok = _mdl = None

# Regex patterns for robust extraction
_ROAD_RX = re.compile(r'\b(?:I|US|VA)-\d{1,3}\b')
_AGE_RX = re.compile(r'\b(\d{1,2})-year-old\b', re.I)
_COORD_RX = re.compile(r'Coordinates:\s*([+-]?\d{1,2}\.\d+)\s*,\s*([+-]?\d{1,3}\.\d+)')
_LASTSEEN_RX = re.compile(r'Last Seen:\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([A-Za-z ]+)\b')
_NAME_RX = re.compile(r'Missing Person:\s*([^,]+)')
_GENDER_RX = re.compile(r'\b( male| female)\b', re.I)

def _sex_to_letter(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().lower()
    return 'M' if 'male' in s else 'F' if 'female' in s else None

def _safe(obj):
    # convert python None/True/False to strict JSON-friendly
    return json.loads(json.dumps(obj))

def _json_repair(s: str) -> Optional[dict]:
    """Try hard to pull the first JSON object from text and parse it."""
    # Grab the first {...} block
    start = s.find('{')
    if start == -1: return None
    # simple stack to find matching closing brace
    depth, end = 0, None
    for i, ch in enumerate(s[start:], start=start):
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None: return None
    frag = s[start:end]
    # trivial fixes: stray trailing commas before },] and smart quotes
    frag = re.sub(r',\s*([}\]])', r'\1', frag)
    frag = frag.replace('"','"').replace('"','"').replace("'","'")
    try:
        return json.loads(frag)
    except Exception:
        return None

def _dedupe(seq):
    """Remove duplicates while preserving order"""
    seen = set()
    out = []
    for x in seq:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _seed_from_narrative(narr: str) -> Dict[str, Any]:
    """Robust prefill from narrative using regex patterns."""
    name = (m.group(1).strip() if (m := _NAME_RX.search(narr)) else None)
    age = int(m.group(1)) if (m := _AGE_RX.search(narr)) else None
    gender = _sex_to_letter((m.group(1) if (m := _GENDER_RX.search(narr)) else None))
    lat = lon = None
    if (m := _COORD_RX.search(narr)):
        lat = float(m.group(1)); lon = float(m.group(2))
    city = county = state = None
    if (m := _LASTSEEN_RX.search(narr)):
        city = m.group(1).strip() if m.group(1).strip().lower() != 'unknown' else None
        county = m.group(2).strip() if m.group(2).strip().lower() != 'unknown' else None
        st = m.group(3).strip()
        state = 'VA' if 'virginia' in st.lower() else st  # keep VA abbrev

    roads = list(dict.fromkeys(_ROAD_RX.findall(narr)))  # uniq and keep order

    # light heuristics for risk factors
    rf = []
    if age is not None and age <= 8: rf.append("Very young child")
    if re.search(r'\boffer(ing)? a ride\b', narr, re.I): rf.append("Luring/offer of ride")
    if re.search(r'\b(abduct|force|weapon|gun|knife|threat)\b', narr, re.I): rf.append("Violence/weapon indicators")
    if 'unknown adult' in narr.lower() or 'companion' in narr.lower(): rf.append("Unknown adult involved")
    if roads: rf.append("Vehicle movement")

    return {
      "name": name, "aliases": [],
      "date_reported": None,
      "location": {"city": city, "county": county, "state": state or "VA"},
      "lat": lat, "lon": lon,
      "age": age, "gender": gender, "race": None,
      "height_cm": None, "weight_kg": None,
      "movement_cues": roads,
      "risk_factors": rf
    }

def _merge(a: dict, b: dict) -> dict:
    """Deep-merge only where b has non-null values."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v not in (None, "", [], {}):
            out[k] = v
    return out

LOCKED_PROMPT = """You are an extraction assistant.
You will receive:
1) A short case text
2) A JSON object with some fields already filled (LOCKED). These must not be changed.
3) Some fields set to null. Fill only the null fields, using evidence from the text.

Rules:
- DO NOT change any non-null value in the JSON.
- Output ONLY a single JSON object. No commentary.
- If a value is unknown, keep it as null (do not invent).
- "name" MUST be the missing person (not vehicles/places).
- "gender" should be "M" or "F".
- Coordinates should be numeric floats if present.

Text:
{narrative}

LOCKED+PARTIAL JSON:
{seed}
"""

def _json_repair(s: str) -> Dict[str, Any]:
    """Strip to the outermost braces and try strict JSON"""
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    try:
        return json.loads(s)
    except Exception:
        # very light repair: remove trailing commas
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(s)

def _trim_to_tokens(text: str, tok, max_input_tokens: int = 800) -> str:
    """Trim text to fit within token limit for faster processing"""
    ids = tok(text, return_tensors="pt", add_special_tokens=False, truncation=True,
              max_length=max_input_tokens)["input_ids"][0]
    return tok.decode(ids, skip_special_tokens=True)

def unload_model(model, tokenizer):
    """Explicitly unload model and clear GPU memory"""
    del model, tokenizer
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def release():
    """Release model and clear GPU memory"""
    global _mdl, _tok
    try:
        del _mdl, _tok
    except Exception:
        pass
    _mdl = _tok = None
    import torch, gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _assert_model_dir(path_str):
    """
    Validate that the model directory exists and contains config.json.
    
    Args:
        path_str (str): Path to the model directory
        
    Raises:
        FileNotFoundError: If model directory is missing or lacks config.json
    """
    p = Path(path_str)
    if not p.exists() or not (p / "config.json").exists():
        raise FileNotFoundError(
            f"Model dir '{p}' is missing or lacks config.json. "
            "Point to the exact directory that contains the model files."
        )

def _ensure_loaded():
    """
    Ensure the model and tokenizer are loaded and cached globally.
    
    This function implements lazy loading - the model is only loaded when
    first needed, and then cached for subsequent calls to avoid reloading.
    Uses GPU optimization with 4-bit quantization for memory efficiency.
    
    Raises:
        FileNotFoundError: If model directory is not found
        RuntimeError: If model loading fails
    """
    global _tok, _mdl
    if _mdl is not None: 
        return
    
    # Validate model directory exists
    _assert_model_dir(DIR_EXTRACTOR)
    
    print(f"[INIT] Using model dir: {Path(DIR_EXTRACTOR).resolve()}")
    
    _tok = AutoTokenizer.from_pretrained(DIR_EXTRACTOR, use_fast=True, local_files_only=True)
    print(f"[CHK]  Tokenizer path: {_tok.name_or_path}")
    if _tok.pad_token_id is None and _tok.eos_token_id is not None:
        _tok.pad_token = _tok.eos_token

    # 4-bit NF4 quantization for Qwen2-7B (fast GPU extraction) - optimized for RTX 4060
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        _mdl = AutoModelForCausalLM.from_pretrained(
            DIR_EXTRACTOR,
            device_map="cuda:0",          # force onto GPU
            quantization_config=bnb_config,
            attn_implementation="eager",  # Use eager attention for Windows compatibility
            torch_dtype=torch.float16,
            use_cache=True,  # Enable KV cache for faster inference
            local_files_only=True
        )
        
        # Disable FlashAttention 2 on Windows
        if hasattr(_mdl.config, "use_flash_attention_2"):
            _mdl.config.use_flash_attention_2 = False
    else:
        # Fallback to CPU with offloading
        max_mem = {"cpu": "28GiB"}
        Path("offload").mkdir(exist_ok=True)
        _mdl = AutoModelForCausalLM.from_pretrained(
            DIR_EXTRACTOR,
            device_map="auto",
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            max_memory=max_mem,
            offload_state_dict=True,
            offload_folder="offload",
            attn_implementation="eager",  # Use eager attention for Windows compatibility
            local_files_only=True
        )
        
        # Disable FlashAttention 2 on Windows
        if hasattr(_mdl.config, "use_flash_attention_2"):
            _mdl.config.use_flash_attention_2 = False
    
    print(f"[CHK]  Model name_or_path: {getattr(_mdl.config, '_name_or_path', 'unknown')}")
    print(f"[CHK]  4-bit loaded: {getattr(_mdl, 'is_loaded_in_4bit', False)}")
    print(f"[CHK]  Device map keys: {list(getattr(_mdl, 'hf_device_map', {}).keys())[:5]}...")
    _mdl.eval()  # Set to evaluation mode

def extract_entities(narrative: str) -> dict:
    """
    Extract entities from case narrative using Qwen2.5-3B-Instruct.
    
    This function processes a case narrative and extracts structured entities
    including persons, vehicles, locations, timeline events, and evidence.
    The output is returned as a dictionary with standardized JSON structure.
    
    Args:
        narrative (str): The case narrative text to extract entities from
        
    Returns:
        dict: Dictionary containing extracted entities with the following structure:
            {
                "persons": [{"name": str, "description": str, "role": str}],
                "vehicles": [{"make": str, "model": str, "color": str, "license": str}],
                "locations": [{"address": str, "landmark": str, "coordinates": str}],
                "timeline": [{"date": str, "time": str, "event": str}],
                "evidence": [{"type": str, "description": str, "location": str}]
            }
            
    Example:
        >>> narrative = "John Smith was seen driving a red Honda Civic..."
        >>> entities = extract_entities(narrative)
        >>> print(entities["persons"][0]["name"])  # "John Smith"
        
    Note:
        If JSON parsing fails, returns an empty structure with empty lists.
        The function uses low temperature (0.1) for consistent JSON output.
    """
    _ensure_loaded()
    
    # Trim input to ~800 tokens for faster processing
    trimmed_narrative = _trim_to_tokens(narrative, _tok, max_input_tokens=800)
    
    # Use chat template if available
    has_chat = getattr(_tok, "chat_template", None) not in (None, "")
    
    # Much simpler and more direct prompt
    improved_prompt = f"""Extract entities from this missing person case:

{trimmed_narrative}

Return JSON with:
- persons: names, ages, roles
- vehicles: make, model, color, license
- locations: addresses, cities, coordinates  
- timeline: dates, times, events
- evidence: descriptions, types

JSON:"""
    
    if has_chat:
        messages = [{"role": "user", "content": improved_prompt}]
        prompt = _tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = _tok(prompt, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
    else:
        enc = _tok(improved_prompt, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
    
    if torch.cuda.is_available(): 
        ids, mask = ids.to(_mdl.device), mask.to(_mdl.device)
    
    # Generate with optimized settings for better JSON extraction
    with torch.inference_mode():
        out = _mdl.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=96,   # Short JSON output
            do_sample=False,     # Greedy for faster, deterministic JSON
            pad_token_id=_tok.eos_token_id,
            eos_token_id=_tok.eos_token_id
        )
    
    # Decode response
    response = _tok.decode(out[0], skip_special_tokens=True)
    
    # Extract JSON from response
    if has_chat:
        response = response.split("assistant")[-1].strip()
    else:
        response = response.split("JSON:")[-1].strip()
    
    # Try to parse JSON with better error handling
    try:
        # Find JSON block
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response[start:end]
            result = json.loads(json_str)
            
            # Ensure all required keys exist
            required_keys = ["persons", "vehicles", "locations", "timeline", "evidence"]
            for key in required_keys:
                if key not in result:
                    result[key] = []
                elif not isinstance(result[key], list):
                    result[key] = []
            
            return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing failed: {e}")
        print(f"Response was: {response[:200]}...")
    
    # Fallback: return empty structure
    return {
        "persons": [],
        "vehicles": [],
        "locations": [],
        "timeline": [],
        "evidence": []
    }

def extract_persons(narrative: str) -> list:
    """
    Extract only person entities from case narrative.
    
    This is a convenience function that extracts only person entities from
    a case narrative, filtering out other entity types.
    
    Args:
        narrative (str): The case narrative text to extract persons from
        
    Returns:
        list: List of person dictionaries with structure:
            [{"name": str, "description": str, "role": str}, ...]
            
    Example:
        >>> narrative = "John Smith, a 35-year-old suspect, was seen..."
        >>> persons = extract_persons(narrative)
        >>> print(persons[0]["name"])  # "John Smith"
    """
    entities = extract_entities(narrative)
    return entities.get("persons", [])

def extract_vehicles(narrative: str) -> list:
    """
    Extract only vehicle entities from case narrative.
    
    This is a convenience function that extracts only vehicle entities from
    a case narrative, filtering out other entity types.
    
    Args:
        narrative (str): The case narrative text to extract vehicles from
        
    Returns:
        list: List of vehicle dictionaries with structure:
            [{"make": str, "model": str, "color": str, "license": str}, ...]
            
    Example:
        >>> narrative = "A red Honda Civic with license plate ABC123..."
        >>> vehicles = extract_vehicles(narrative)
        >>> print(vehicles[0]["make"])  # "Honda"
    """
    entities = extract_entities(narrative)
    return entities.get("vehicles", [])

def extract_locations(narrative: str) -> list:
    """
    Extract only location entities from case narrative.
    
    This is a convenience function that extracts only location entities from
    a case narrative, filtering out other entity types.
    
    Args:
        narrative (str): The case narrative text to extract locations from
        
    Returns:
        list: List of location dictionaries with structure:
            [{"address": str, "landmark": str, "coordinates": str}, ...]
            
    Example:
        >>> narrative = "The incident occurred at 123 Main Street..."
        >>> locations = extract_locations(narrative)
        >>> print(locations[0]["address"])  # "123 Main Street"
    """
    entities = extract_entities(narrative)
    return entities.get("locations", [])

# Schema keys for structured JSON extraction
SCHEMA_KEYS = [
    "name", "aliases", "date_reported",
    "location", "lat", "lon", "age", "gender", "race",
    "height_cm", "weight_kg", "movement_cues", "risk_factors"
]

SYSTEM = (
    "You are an information extractor for missing-child cases. "
    "Return ONE JSON object and nothing else. Use null when a value is not present.\n\n"
    "Field definitions:\n"
    "- name: the missing child's personal name only (e.g., 'Alex Kim'); never a place.\n"
    "- aliases: list of nicknames/aliases for the child.\n"
    "- date_reported: ISO date/time the case was reported missing.\n"
    "- location: object with city, county, state (2-letter code like 'VA').\n"
    "- lat/lon: decimal degrees if the last-seen location gives coordinates; else null.\n"
    "- age: numeric age in years.\n"
    "- gender: 'M' or 'F' if explicit (boy/girl → child gender); else null.\n"
    "- race: demographic if explicitly stated; else null.\n"
    "- height_cm/weight_kg: numbers if explicit; else null.\n"
    "- movement_cues: list of road/transit strings (e.g., 'I-95', 'US-58').\n"
    "- risk_factors: list of concise flags if explicit (e.g., 'weapon_mentioned').\n"
)

FEWSHOT = """Text: "Jane Roe, 11, last seen 08:30 near I-95 in Henrico County, VA."
JSON:
{"name":"Jane Roe","aliases":[],"date_reported":null,
 "location":{"city":null,"county":"Henrico","state":"VA"},
 "lat":null,"lon":null,"age":11,"gender":null,"race":null,
 "height_cm":null,"weight_kg":null,
 "movement_cues":["I-95"],"risk_factors":[]}
"""

def _robust_json_slice(s: str) -> str:
    """Return the last top-level {...} block in s"""
    depth = 0
    start = end = -1
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
    return s[start:end] if start != -1 and end != -1 else "{}"

def _regex_backfill(text: str, obj: dict) -> dict:
    """Only fill fields that are null/empty using regex patterns"""
    def need(k): 
        v = obj.get(k, None)
        return v is None or (isinstance(v, (list,dict)) and not v)

    # name + age
    if need("name") or need("age"):
        m = re.search(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b(?:,|\s)(?:age\s*)?(\d{1,2})?", text)
        if m:
            if need("name"): 
                obj["name"] = m.group(1)
            if need("age") and m.group(2): 
                obj["age"] = int(m.group(2))

    # county + state
    loc = obj.get("location") or {}
    if not isinstance(loc, dict): 
        loc = {}
    if loc.get("county") in (None, ""):
        m = re.search(r"\b([A-Z][a-z]+)\s+County\b", text)
        if m: 
            loc["county"] = m.group(1)
    if loc.get("state") in (None, ""):
        if re.search(r"\bVA\b|\bVirginia\b", text, re.I):
            loc["state"] = "VA"
    if "city" not in loc: 
        loc["city"] = None
    obj["location"] = loc

    # movement cues (highways/routes)
    if need("movement_cues"):
        cues = re.findall(r"\b(I-\d{1,3}|US-\d+|VA-\d+|Route\s*\d+)\b", text)
        obj["movement_cues"] = list(dict.fromkeys(cues))  # unique, keep order

    # gender
    if need("gender"):
        if re.search(r"\b(girl|female|she|her)\b", text, re.I): 
            obj["gender"] = "F"
        elif re.search(r"\b(boy|male|he|him)\b", text, re.I):   
            obj["gender"] = "M"

    # city (from 'Last Seen: City, County, State' or 'Last Seen: City, State')
    if obj.get("location", {}).get("city") in (None, ""):
        m = re.search(r"Last Seen:\s*([A-Z][A-Za-z .'-]+),\s*(?:[A-Z][A-Za-z .'-]+,\s*)?(Virginia|VA)\b", text)
        if m:
            loc = obj.get("location") or {}
            loc["city"] = m.group(1)
            loc["state"] = "VA"
            obj["location"] = loc

    # lat/lon
    if need("lat") or need("lon"):
        m = re.search(r"Coordinates:\s*([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)", text)
        if m:
            obj["lat"] = float(m.group(1)); obj["lon"] = float(m.group(2))

    # vehicles (very light)
    if need("movement_cues"):
        cues = re.findall(r"\b(I-\d{1,3}|US-\d+|VA-\d+|Route\s*\d+)\b", text)
        obj["movement_cues"] = list(dict.fromkeys(cues))

    # risk_factors stays [] unless obvious keyword present
    if need("risk_factors"):
        risks = []
        if re.search(r"\bweapon|gun|knife\b", text, re.I): 
            risks.append("weapon_mentioned")
        if re.search(r"\babduction|forced\b", text, re.I): 
            risks.append("abduction_cue")
        obj["risk_factors"] = risks

    return obj

def extract_json(text: str, hints: dict | None = None) -> dict:
    """
    Extract structured JSON from case narrative using hybrid approach.
    
    This function combines deterministic regex-based extraction with LLM-based
    filling of missing fields. It first creates a scaffold using regex patterns,
    then uses the LLM to fill in missing information.
    
    Args:
        text (str): Case narrative text to extract from
        hints (dict | None): Optional hints dictionary with spatial, temporal, and narrative_osint data
        
    Returns:
        dict: Structured entity data with all available fields populated
    """
    _ensure_loaded()
    
    # 1) Deterministic scaffold first
    seed = scaffold_from_narrative(text)
    
    # 2) Generate LLM response for missing fields only
    prompt = f"""Extract ONLY the following fields from the case narrative as strict JSON:
{{
  "name": null,
  "age": null,
  "gender": null,
  "location": {{
    "city": null,
    "county": null,
    "state": "VA"
  }},
  "lat": null,
  "lon": null,
  "date_reported": null
}}

Rules:
- Return ONLY valid JSON, no comments or prose.
- Use uppercase "M" or "F" for gender.
- Use null for unknowns.
- Keep state as "VA" when the narrative indicates Virginia (VA or Virginia).
- Do not invent information not supported by the narrative.

Case Narrative:
{text}

JSON:"""

    gen = _generate_json_strict(prompt, max_new_tokens=96)  
    # 3) Parse LLM JSON
    m = re.search(r'\{.*\}', gen, flags=re.S)
    if not m:  # try to repair quotes/trailing commas
        gen = gen.replace("'", '"')
        gen = re.sub(r",\s*([}\]])", r"\1", gen)
        m = re.search(r'\{.*\}', gen, flags=re.S)
    clean = m.group(0) if m else "{}"

    try:
        llm_data = json.loads(clean)
    except Exception:
        # last-resort repairs
        clean = re.sub(r'[\x00-\x1f]', '', clean)
        clean = re.sub(r",\s*([}\]])", r"\1", clean)
        try:
            llm_data = json.loads(clean)
        except Exception:
            llm_data = {}

    # 4) Merge LLM output into scaffold (never overwrite good values)
    result = merge_keep_existing(seed, llm_data)

    # 5) Use hints to fill precise fields (no LLM guesswork)
    if hints:
        loc = hints.get("spatial", {})
        if not result["location"]["city"] and loc.get("last_seen_city"):   result["location"]["city"] = loc["last_seen_city"]
        if not result["location"]["county"] and loc.get("last_seen_county"): result["location"]["county"] = loc["last_seen_county"]
        if not result["location"]["state"] and loc.get("last_seen_state"): result["location"]["state"] = loc["last_seen_state"]
        if result["lat"] is None and loc.get("last_seen_lat") is not None: result["lat"] = loc["last_seen_lat"]
        if result["lon"] is None and loc.get("last_seen_lon") is not None: result["lon"] = loc["last_seen_lon"]
        tmp = hints.get("temporal", {})
        if result["date_reported"] is None and tmp.get("reported_missing_ts"): result["date_reported"] = tmp["reported_missing_ts"]
        if not result["movement_cues"]:
            cues = hints.get("narrative_osint", {}).get("movement_cues_text", "")
            result["movement_cues"] = re.findall(r"\b(I-\d{1,3}|US-\d{1,3}|VA-\d{1,3})\b", cues)

    # 6) Deterministic backfill for missing fields
    result = backfill(result, text)

    # 7) Final guardrails: normalize state and county
    if result["location"]["county"]:
        result["location"]["county"] = re.sub(r'\s+County$', '', result["location"]["county"], flags=re.I)
    if result["location"]["state"] and len(result["location"]["state"]) > 2:
        if result["location"]["state"].lower().startswith("virgin"): 
            result["location"]["state"] = "VA"
    if result.get("gender") and result["gender"] not in ("M","F"):
        result["gender"] = "F" if result["gender"].lower().startswith("f") else "M"

    return result

def batch_extract_json(narratives: list[str], batch_size: int = 2, hints_list: list[dict] | None = None) -> list[dict]:
    """
    Batch process multiple narratives for JSON extraction.
    
    This function processes multiple case narratives in sequence, applying
    the same extraction logic to each. It supports optional hints for each narrative.
    
    Args:
        narratives (list[str]): List of case narrative texts to process
        batch_size (int): Number of narratives to process (currently unused, processes sequentially)
        hints_list (list[dict] | None): Optional list of hints dictionaries, one per narrative
        
    Returns:
        list[dict]: List of extracted entity dictionaries, one per narrative
    """
    out = []
    for i, text in enumerate(narratives):
        hints = hints_list[i] if hints_list and i < len(hints_list) else None
        out.append(extract_json(text, hints=hints))
    return out
