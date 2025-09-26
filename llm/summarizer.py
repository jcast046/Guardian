"""
Summarizer that auto-switches (Instruct first, Base fallback)

This module provides intelligent case summarization using Llama-3 8B models with
automatic fallback between instruct-tuned and base model variants. The summarizer
is designed to generate investigator-focused summaries from case narratives.

The module implements a smart fallback system:
1. First attempts to load the instruct-tuned model (Llama-3.1-8B-Instruct)
2. If that fails, falls back to the base model (Llama-3.1-8B)
3. Uses appropriate prompting for each model type

Author: Guardian AI System


Functions:
    summarize(narrative: str) -> str: Generate investigator summary from case narrative

Example:
    >>> from llm import summarize
    >>> summary = summarize("John Smith was seen driving a red Honda...")
    >>> print(summary)  # "• Suspect: John Smith\n• Vehicle: Red Honda..."
"""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load configuration from guardian.config.json
CFG = json.load(open("guardian.config.json", "r"))
DIR_INSTRUCT = CFG["models"]["summarizer_instruct"]
DIR_BASE     = CFG["models"]["summarizer_base"]
MODE         = CFG.get("use_summarizer", "instruct")  # "instruct" or "base"

def _load(dir_):
    """
    Load a model and tokenizer from the specified directory.
    
    Args:
        dir_ (str): Path to the model directory
        
    Returns:
        tuple: (tokenizer, model) - The loaded tokenizer and model objects
        
    Note:
        This function handles device placement and data type optimization
        based on available hardware (CUDA vs CPU).
    """
    tok = AutoTokenizer.from_pretrained(dir_, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        dir_,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    return tok, model

# Global model cache variables
_tok = _mdl = None

def _ensure_loaded():
    """
    Ensure the model and tokenizer are loaded with intelligent fallback.
    
    This function implements the core fallback logic:
    1. If model is already loaded, return immediately
    2. If MODE is "instruct", try to load the instruct model first
    3. If instruct model fails or MODE is "base", fall back to base model
    4. Cache the loaded model globally for efficiency
    """
    global _tok, _mdl
    if _mdl is not None: 
        return
    if MODE == "instruct":
        try:
            _tok, _mdl = _load(DIR_INSTRUCT)
            return
        except Exception:
            pass
    _tok, _mdl = _load(DIR_BASE)

def summarize(narrative: str) -> str:
    """
    Generate an investigator-focused summary from a case narrative.
    
    This function processes a case narrative and generates a concise summary
    optimized for investigators. The summary includes key facts, avoids speculation,
    and is formatted as bullet points for easy reading.
    
    Args:
        narrative (str): The case narrative text to summarize
        
    Returns:
        str: A formatted summary with key points for investigators
        
    Example:
        >>> narrative = "John Smith was seen driving a red Honda Civic..."
        >>> summary = summarize(narrative)
        >>> print(summary)
        # "• Suspect: John Smith\n• Vehicle: Red Honda Civic\n• Location: ..."
        
    Note:
        The function automatically detects whether to use chat templates
        (for instruct models) or instruction/response format (for base models).
        Uses temperature 0.3 for consistent, factual output.
    """
    _ensure_loaded()
    
    # Check if model supports chat templates (instruct models)
    has_chat = getattr(_tok, "chat_template", None) not in (None, "")
    
    # Create the summarization prompt
    user = ("Summarize the case for investigators in 5 concise bullet points. "
            "Be factual, avoid speculation.\n\nText:\n"+narrative)

    if has_chat:  # Instruct path - use chat template
        ids = _tok.apply_chat_template(
            [{"role":"user","content":user}],
            add_generation_prompt=True, return_tensors="pt"
        )
    else:         # Base path - use instruction/response format
        ids = _tok("### Instruction:\n"+user+"\n### Response:\n",
                   return_tensors="pt").input_ids

    # Move to GPU if available
    if torch.cuda.is_available(): 
        ids = ids.to(_mdl.device)
    
    # Generate summary with conservative settings for factual output
    out = _mdl.generate(ids, max_new_tokens=256, temperature=0.3, do_sample=False)
    txt = _tok.decode(out[0], skip_special_tokens=True)
    
    # Extract just the response part
    return txt.split("### Response:")[-1].strip()
