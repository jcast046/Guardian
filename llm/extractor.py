"""
Entity extraction using Qwen2-7B-Instruct for JSON extraction.

This module provides structured entity extraction capabilities using the Qwen2-7B-Instruct
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

Example:
    >>> from llm import extract_entities, extract_persons
    >>> entities = extract_entities("John Smith was seen driving a red Honda...")
    >>> persons = extract_persons("John Smith was seen driving a red Honda...")
"""
import json, re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .prompts import EXTRACTION_PROMPT

# Enable TF32 for RTX 40xx speedup
torch.set_float32_matmul_precision("high")

# Load configuration from guardian.config.json
CFG = json.load(open("guardian.config.json", "r"))
DIR_EXTRACTOR = CFG["models"]["extractor"]
USE_LLAMA = CFG.get("use_llama_as_extractor", False)

# Global model cache variables
_tok = _mdl = None

def unload_model(model, tokenizer):
    """Explicitly unload model and clear GPU memory"""
    del model, tokenizer
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _ensure_loaded():
    """
    Ensure the model and tokenizer are loaded and cached globally.
    
    This function implements lazy loading - the model is only loaded when
    first needed, and then cached for subsequent calls to avoid reloading.
    Uses GPU optimization with offloading for memory safety.
    """
    global _tok, _mdl
    if _mdl is not None: 
        return
    
    _tok = AutoTokenizer.from_pretrained(DIR_EXTRACTOR, use_fast=True)
    if _tok.pad_token_id is None and _tok.eos_token_id is not None:
        _tok.pad_token = _tok.eos_token

    # Use 8-bit quantization for Qwen (fits entirely on GPU)
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        _mdl = AutoModelForCausalLM.from_pretrained(
            DIR_EXTRACTOR,
            device_map="cuda:0",
            quantization_config=bnb_config,
            attn_implementation="sdpa",
        )
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
            attn_implementation="sdpa",
        )
    _mdl.eval()  # Set to evaluation mode

def extract_entities(narrative: str) -> dict:
    """
    Extract entities from case narrative using Qwen2-7B-Instruct.
    
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
    
    # Pre-trim narrative to last 400-600 tokens (keep most recent info)
    if len(narrative) > 2000:  # Rough estimate: trim if very long
        words = narrative.split()
        narrative = " ".join(words[-400:])  # Keep last 400 words
    
    # Use chat template if available
    has_chat = getattr(_tok, "chat_template", None) not in (None, "")
    
    if has_chat:
        messages = [{"role": "user", "content": EXTRACTION_PROMPT.format(narrative=narrative)}]
        prompt = _tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = _tok(prompt, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
    else:
        prompt = EXTRACTION_PROMPT.format(narrative=narrative)
        enc = _tok(prompt, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
    
    if torch.cuda.is_available(): 
        ids, mask = ids.to(_mdl.device), mask.to(_mdl.device)
    
    # Generate with optimized settings and early stopping for JSON
    with torch.inference_mode():
        out = _mdl.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=48, 
            do_sample=False,
            pad_token_id=_tok.eos_token_id,
            eos_token_id=_tok.eos_token_id
        )
    
    # Decode response
    response = _tok.decode(out[0], skip_special_tokens=True)
    
    # Extract JSON from response
    if has_chat:
        response = response.split("assistant")[-1].strip()
    else:
        response = response.split("Return only valid JSON")[-1].strip()
    
    # Try to parse JSON
    try:
        # Find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response[start:end]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    
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
    "You are a precise information extractor. Output ONLY valid JSON as a single object. "
    "Use null if a value is not present in the text. Do not invent values. "
    "Coordinates must be decimal degrees (floats) if present."
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

    # risk_factors stays [] unless obvious keyword present
    if need("risk_factors"):
        risks = []
        if re.search(r"\bweapon|gun|knife\b", text, re.I): 
            risks.append("weapon_mentioned")
        if re.search(r"\babduction|forced\b", text, re.I): 
            risks.append("abduction_cue")
        obj["risk_factors"] = risks

    return obj

def extract_json(text: str) -> dict:
    """
    Extract structured JSON from case narrative using Qwen2-7B-Instruct.
    
    This function uses a robust JSON extraction approach with regex backfill
    to ensure accurate field extraction from case narratives.
    
    Args:
        text (str): The case narrative text to extract from
        
    Returns:
        dict: Structured JSON with all required schema keys
        
    Example:
        >>> text = "John Doe, 12, last seen near I-95 in Henrico County, VA at 10:00 AM."
        >>> result = extract_json(text)
        >>> print(result["name"])  # "John Doe"
    """
    _ensure_loaded()
    
    # Pre-trim narrative to last 400-600 tokens (keep most recent info)
    if len(text) > 2000:  # Rough estimate: trim if very long
        words = text.split()
        text = " ".join(words[-400:])  # Keep last 400 words
    
    instruction = (
        "Extract the following fields as ONE JSON object with these exact keys:\n"
        f"{SCHEMA_KEYS}\n\n"
        "JSON rules:\n"
        "- Output JSON only (no prose)\n"
        "- Use null when the value is not present in the text\n"
        "- location must be an object: {\"city\":null|\"...\",\"county\":null|\"...\",\"state\":\"VA\"|null}\n"
        "- lat/lon are numbers if present; otherwise null\n\n"
        f"{FEWSHOT}\n"
        f'Text: "{text}"\nJSON:'
    )
    
    prompt = _tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": instruction}],
        add_generation_prompt=True, 
        tokenize=False
    )
    
    enc = _tok(prompt, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    
    if torch.cuda.is_available(): 
        ids, mask = ids.to(_mdl.device), mask.to(_mdl.device)
    
    with torch.inference_mode():
        out = _mdl.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=128,  # give room; we'll trim
            do_sample=False
        )
    
    gen_only = out[0, ids.shape[-1]:]
    s = _tok.decode(gen_only, skip_special_tokens=True)
    payload = _robust_json_slice(s)
    
    import json
    try:
        obj = json.loads(payload)
    except Exception:
        obj = {}
    
    # ensure keys exist
    for k in SCHEMA_KEYS:
        if k not in obj:
            obj[k] = None if k not in ("movement_cues", "risk_factors", "location") else ([] if k != "location" else {"city": None, "county": None, "state": None})
    if not isinstance(obj.get("location"), dict):
        obj["location"] = {"city": None, "county": None, "state": None}
    
    # regex backfill: fill obvious fields if still null
    obj = _regex_backfill(text, obj)
    return obj
