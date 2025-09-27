<<<<<<< HEAD
"""
Guardian LLM Summarizer Module

This module provides case summarization functionality using the Llama-3.1-8B-Instruct model.
It generates concise, factual summaries for investigators in a structured bullet-point format.

Author: Joshua Castillo
"""

import json
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# Enable TF32 for RTX 40xx speedup
torch.set_float32_matmul_precision("high")

# Load configuration
CFG = json.load(open("guardian.config.json", "r"))
DIR_INSTRUCT = CFG["models"]["summarizer_instruct"]

# Global model state (lazy loading)
_tok: Optional[AutoTokenizer] = None
_mdl: Optional[AutoModelForCausalLM] = None

def unload_model(model, tokenizer):
    """Explicitly unload model and clear GPU memory"""
    del model, tokenizer
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _load() -> None:
    """
    Load the tokenizer and model with optimized settings.
    
    This function implements lazy loading to avoid loading the model until it's needed.
    It uses device_map="auto" for GPU optimization and offloading to avoid memory issues.
    
    Raises:
        FileNotFoundError: If the model directory doesn't exist
        RuntimeError: If model loading fails
    """
    global _tok, _mdl
    if _mdl is not None:
        return
    
    try:
        _tok = AutoTokenizer.from_pretrained(DIR_INSTRUCT, use_fast=True)
        
        # Let HF Accelerate split weights across GPU+CPU to avoid pagefile errors
        max_mem = (
            {0: "7.8GiB", "cpu": "18GiB"} if torch.cuda.is_available()
            else {"cpu": "30GiB"}
        )
        Path("offload").mkdir(exist_ok=True)

        _mdl = AutoModelForCausalLM.from_pretrained(
            DIR_INSTRUCT,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            max_memory=max_mem,
            offload_state_dict=True,
            offload_folder="offload",
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        _mdl.eval()  # Set to evaluation mode
    except Exception as e:
        raise RuntimeError(f"Failed to load summarizer model from {DIR_INSTRUCT}: {e}")


def summarize(narrative: str) -> str:
    """
    Generate a concise case summary for investigators.
    
    This function takes a case narrative and generates a structured summary in 5 bullet points.
    The summary is designed for investigators and focuses on factual information without speculation.
    
    Args:
        narrative (str): The case narrative text to summarize
        
    Returns:
        str: A formatted summary with 5 concise bullet points
        
    Raises:
        ValueError: If narrative is empty or None
        RuntimeError: If model generation fails
        
    Example:
        >>> narrative = "Missing child last seen at school..."
        >>> summary = summarize(narrative)
        >>> print(summary)
        • Child: 8-year-old female, last seen at Lincoln Elementary
        • Last known location: School playground at 3:15 PM
        • Clothing: Blue jeans, red sweater, white sneakers
        • Witness: Teacher reported child left with unknown adult
        • Timeline: Missing since 3:15 PM, reported at 4:30 PM
    """
    if not narrative or not narrative.strip():
        raise ValueError("Narrative cannot be empty or None")
    
    _load()
    
    # Pre-trim narrative to last 400-600 tokens (keep most recent info)
    if len(narrative) > 2000:  # Rough estimate: trim if very long
        words = narrative.split()
        narrative = " ".join(words[-400:])  # Keep last 400 words
    
    # Construct the prompt for investigators
    user = ("Summarize the case for investigators in 5 concise bullet points. "
            "Be factual; no speculation.\n\nText:\n" + narrative)
    
    try:
        # Build chat prompt then tokenize to get attention_mask
        prompt = _tok.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True, 
            tokenize=False
        )
        enc = _tok(prompt, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        
        # Move to GPU if available
        if torch.cuda.is_available():
            ids, mask = ids.to(_mdl.device), mask.to(_mdl.device)
        
        # Generate summary with optimized settings
        with torch.inference_mode():
            out = _mdl.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=96,        # give room for 5 bullets; 64 may truncate
                do_sample=False
            )
        
        # Decode only what the model just generated (no prompt echo)
        gen_only = out[0, ids.shape[-1]:]
        text = _tok.decode(gen_only, skip_special_tokens=True)
        
        # Optional: stop after the 5th bullet (handles "•" or "-" starts)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullets = []
        for ln in lines:
            if ln.startswith("•") or ln.startswith("-"):
                bullets.append(ln.lstrip("•- ").strip())
                if len(bullets) == 5:
                    break
        if bullets:
            return "• " + "\n• ".join(bullets)
        return text.strip()
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate summary: {e}")
=======
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
>>>>>>> 78fde9e6dbb2933c5cff903bda29caec00f2c6a3
