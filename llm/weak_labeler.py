"""
Weak labeling using Mistral-7B for movement classification and risk assessment.

This module provides automated labeling capabilities for case narratives using the
Mistral-7B-Instruct-v0.2 model. It classifies movement patterns and assesses risk
levels to support investigative workflows and case prioritization.

The weak labeler is designed to provide consistent, automated labeling that can be
used for case categorization, risk assessment, and workflow optimization.

Author: Guardian AI System


Functions:
    classify_movement(narrative: str) -> str: Classify movement pattern in case
    assess_risk(narrative: str) -> str: Assess risk level of case
    label_case(narrative: str) -> dict: Combined movement and risk labeling
    batch_label_cases(narratives: list) -> list: Batch process multiple cases

Example:
    >>> from llm import classify_movement, assess_risk, label_case
    >>> movement = classify_movement("Suspect traveled from NYC to LA...")
    >>> risk = assess_risk("Armed robbery with weapon...")
    >>> labels = label_case("Case narrative...")
"""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .prompts import MOVEMENT_CLASSIFICATION_PROMPT, RISK_ASSESSMENT_PROMPT

# Load configuration from guardian.config.json
CFG = json.load(open("guardian.config.json", "r"))
DIR_WEAK_LABELER = CFG["models"]["weak_labeler"]

def _load_weak_labeler():
    """
    Load the Mistral-7B-Instruct-v0.2 model and tokenizer for weak labeling.
    
    Returns:
        tuple: (tokenizer, model) - The loaded tokenizer and model objects
        
    Note:
        This function is called internally by _ensure_loaded() and caches the
        model in global variables for efficiency.
    """
    tok = AutoTokenizer.from_pretrained(DIR_WEAK_LABELER, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        DIR_WEAK_LABELER,
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
    _tok, _mdl = _load_weak_labeler()

def classify_movement(narrative: str) -> str:
    """
    Classify movement pattern in case narrative.
    
    This function analyzes a case narrative and classifies the movement pattern
    of persons or vehicles involved. The classification helps with case categorization
    and investigative workflow optimization.
    
    Args:
        narrative (str): The case narrative text to classify
        
    Returns:
        str: Movement classification from the following categories:
            - "Stationary": Person/vehicle remained in one location
            - "Local": Short-distance movement within neighborhood/city
            - "Regional": Medium-distance movement within state/region
            - "Interstate": Long-distance movement across state lines
            - "International": Movement across country borders
            - "Unknown": Insufficient information to determine
            
    Example:
        >>> narrative = "Suspect traveled from New York to Los Angeles..."
        >>> movement = classify_movement(narrative)
        >>> print(movement)  # "Interstate"
        
    Note:
        If no clear classification is found, returns "Unknown".
        Uses temperature 0.3 for consistent classification output.
    """
    _ensure_loaded()
    
    prompt = MOVEMENT_CLASSIFICATION_PROMPT.format(narrative=narrative)
    ids = _tok(prompt, return_tensors="pt").input_ids
    
    if torch.cuda.is_available(): 
        ids = ids.to(_mdl.device)
    
    out = _mdl.generate(
        ids, 
        max_new_tokens=50, 
        temperature=0.3, 
        do_sample=True,
        pad_token_id=_tok.eos_token_id
    )
    
    response = _tok.decode(out[0], skip_special_tokens=True)
    response = response.split("Movement Classification:")[-1].strip()
    
    # Extract classification from response
    for category in ["Stationary", "Local", "Regional", "Interstate", "International", "Unknown"]:
        if category.lower() in response.lower():
            return category
    
    return "Unknown"

def assess_risk(narrative: str) -> str:
    """
    Assess risk level of case narrative.
    
    This function analyzes a case narrative and assesses the risk level based on
    various factors including violence indicators, weapon involvement, suspect behavior,
    victim vulnerability, and evidence quality.
    
    Args:
        narrative (str): The case narrative text to assess
        
    Returns:
        str: Risk level from the following categories:
            - "Low": Minimal threat indicators
            - "Medium": Some concerning factors present
            - "High": Multiple threat indicators
            - "Critical": Immediate danger indicators
            
    Example:
        >>> narrative = "Armed robbery with weapon, suspect threatened victim..."
        >>> risk = assess_risk(narrative)
        >>> print(risk)  # "High"
        
    Note:
        If no clear risk level is found, returns "Medium" as default.
        Uses temperature 0.3 for consistent risk assessment.
    """
    _ensure_loaded()
    
    prompt = RISK_ASSESSMENT_PROMPT.format(narrative=narrative)
    ids = _tok(prompt, return_tensors="pt").input_ids
    
    if torch.cuda.is_available(): 
        ids = ids.to(_mdl.device)
    
    out = _mdl.generate(
        ids, 
        max_new_tokens=50, 
        temperature=0.3, 
        do_sample=True,
        pad_token_id=_tok.eos_token_id
    )
    
    response = _tok.decode(out[0], skip_special_tokens=True)
    response = response.split("Risk Assessment:")[-1].strip()
    
    # Extract risk level from response
    for level in ["Low", "Medium", "High", "Critical"]:
        if level.lower() in response.lower():
            return level
    
    return "Medium"

def label_case(narrative: str) -> dict:
    """
    Label case with movement and risk classifications.
    
    This is a convenience function that combines movement classification and
    risk assessment into a single call, returning both labels in a dictionary.
    
    Args:
        narrative (str): The case narrative text to label
        
    Returns:
        dict: Dictionary containing both movement and risk labels:
            {
                "movement": str,  # Movement classification
                "risk": str       # Risk level assessment
            }
            
    Example:
        >>> narrative = "Armed suspect traveled from Norfolk to Alexandria..."
        >>> labels = label_case(narrative)
        >>> print(labels)
        # {"movement": "Interstate", "risk": "High"}
    """
    return {
        "movement": classify_movement(narrative),
        "risk": assess_risk(narrative)
    }

def batch_label_cases(narratives: list) -> list:
    """
    Label multiple cases in batch.
    
    This function processes multiple case narratives and returns labels for each.
    Useful for batch processing large numbers of cases efficiently.
    
    Args:
        narratives (list): List of case narrative strings to label
        
    Returns:
        list: List of dictionaries, each containing movement and risk labels:
            [
                {"movement": str, "risk": str},
                {"movement": str, "risk": str},
                ...
            ]
            
    Example:
        >>> narratives = ["Case 1 narrative...", "Case 2 narrative..."]
        >>> labels = batch_label_cases(narratives)
        >>> print(labels[0]["movement"])  # "Local"
        
    Note:
        This function processes cases sequentially. For very large batches,
        consider implementing parallel processing for better performance.
    """
    return [label_case(narrative) for narrative in narratives]
