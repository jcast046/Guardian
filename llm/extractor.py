"""
Entity extraction using Qwen2-7B-Instruct for JSON extraction.

This module provides structured entity extraction capabilities using the Qwen2-7B-Instruct
model. It extracts persons, vehicles, locations, timeline events, and evidence from case
narratives and returns them in a structured JSON format.

The extractor is designed to work with the Guardian case management system and provides
high-quality entity extraction for investigative workflows.

Author: Guardian AI System


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
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .prompts import EXTRACTION_PROMPT

# Load configuration from guardian.config.json
CFG = json.load(open("guardian.config.json", "r"))
DIR_EXTRACTOR = CFG["models"]["extractor"]
USE_LLAMA = CFG.get("use_llama_as_extractor", False)

# Global variables for model caching
# These are loaded once and reused across function calls for efficiency

def _load_extractor():
    """
    Load the Qwen2-7B-Instruct model and tokenizer for entity extraction.
    
    Returns:
        tuple: (tokenizer, model) - The loaded tokenizer and model objects
        
    Note:
        This function is called internally by _ensure_loaded() and caches the
        model in global variables for efficiency.
    """
    tok = AutoTokenizer.from_pretrained(DIR_EXTRACTOR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        DIR_EXTRACTOR,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    return tok, model

# Global model cache variables
_tok = _mdl = None

def _ensure_loaded():
    """
    Ensure the model and tokenizer are loaded and cached globally.
    
    This function implements lazy loading - the model is only loaded when
    first needed, and then cached for subsequent calls to avoid reloading.
    """
    global _tok, _mdl
    if _mdl is not None: 
        return
    _tok, _mdl = _load_extractor()

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
    
    # Use chat template if available
    has_chat = getattr(_tok, "chat_template", None) not in (None, "")
    
    if has_chat:
        messages = [{"role": "user", "content": EXTRACTION_PROMPT.format(narrative=narrative)}]
        ids = _tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    else:
        prompt = EXTRACTION_PROMPT.format(narrative=narrative)
        ids = _tok(prompt, return_tensors="pt").input_ids
    
    if torch.cuda.is_available(): 
        ids = ids.to(_mdl.device)
    
    # Generate with low temperature for consistent JSON output
    out = _mdl.generate(
        ids, 
        max_new_tokens=1024, 
        temperature=0.1, 
        do_sample=True,
        pad_token_id=_tok.eos_token_id
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
