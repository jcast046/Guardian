"""
Guardian LLM Summarizer Module

This module provides case summarization functionality using the Llama-3.2-3B-Instruct model.
It generates concise, factual summaries for investigators in a structured bullet-point format.

Author: Joshua Castillo

Functions:
    summarize(sum_text: str) -> str: Generate concise case summary
    batch_summarize(texts: list, batch_size: int = 8) -> list: Batch process multiple summaries
    release(): Release model and clear GPU memory
    unload_model(model, tokenizer): Explicitly unload model and clear GPU memory

Example:
    >>> from guardian_llm import summarize
    >>> summary = summarize("Case narrative text...")
    >>> print(summary)
"""

import torch
import os
import json
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Enable TF32 for RTX 40xx speedup and advanced optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load configuration from guardian.config.json
try:
    with open("guardian.config.json", "r") as f:
        config = json.load(f)
    MODEL_ID = config["models"]["summarizer_instruct"]
except:
    MODEL_ID = os.getenv("GUARDIAN_SUMM_MODEL", "./models/Llama3_2-3B-Instruct")

# Global model state (lazy loading)
_tok: Optional[AutoTokenizer] = None
_mdl: Optional[AutoModelForCausalLM] = None

def _trim_to_tokens(text: str, tok, max_input_tokens: int = 800) -> str:
    """
    Trim text to fit within token limit for faster processing.
    
    Args:
        text (str): Input text to trim
        tok: Tokenizer object
        max_input_tokens (int): Maximum number of input tokens
        
    Returns:
        str: Trimmed text that fits within token limit
    """
    ids = tok(text, return_tensors="pt", add_special_tokens=False, truncation=True,
              max_length=max_input_tokens)["input_ids"][0]
    return tok.decode(ids, skip_special_tokens=True)

def unload_model(model, tokenizer):
    """
    Explicitly unload model and clear GPU memory.
    
    Args:
        model: Model object to unload
        tokenizer: Tokenizer object to unload
    """
    del model, tokenizer
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def release():
    """
    Release model and clear GPU memory.
    
    This function releases the globally cached model and tokenizer,
    clearing GPU memory for other processes.
    """
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

def _ensure_loaded():
    """
    Load model once, cache globally for speed.
    
    This function implements lazy loading - the model is only loaded when
    first needed, and then cached for subsequent calls to avoid reloading.
    Uses 4-bit quantization for memory efficiency.
    
    Raises:
        FileNotFoundError: If model directory is not found
        RuntimeError: If model loading fails
    """
    global _tok, _mdl
    if _mdl is not None:
        return

    # Log which model path is being used
    print(f"[INIT] Using model dir: {Path(MODEL_ID).resolve()}")

    _tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    print(f"[CHK]  Tokenizer path: {_tok.name_or_path}")
    if _tok.pad_token_id is None and _tok.eos_token_id is not None:
        _tok.pad_token = _tok.eos_token  # stops the "Setting pad_token_id..." spam

    # 4-bit quantization for 8GB VRAM with bf16 compute
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    try:
        _mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
            use_cache=True,               # KV cache = big speedup
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[WARN] Quantized model failed, falling back to CPU: {e}")
        _mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            torch_dtype=torch.float32,
            use_cache=True
        )
    
    print(f"[CHK]  Model name_or_path: {getattr(_mdl.config, '_name_or_path', 'unknown')}")
    print(f"[CHK]  4-bit loaded: {getattr(_mdl, 'is_loaded_in_4bit', False)}")
    print(f"[CHK]  Device map keys: {list(getattr(_mdl, 'hf_device_map', {}).keys())[:5]}...")
    
    # Set pad token for efficient batching
    _mdl.generation_config.pad_token_id = _tok.eos_token_id
    _mdl.eval()


def summarize(sum_text: str) -> str:
    """
    Generate concise case summary optimized for speed.
    
    This function creates a bullet-point summary of a case narrative,
    optimized for investigator use. It uses a compact prompt and
    generation settings for fast processing.
    
    Args:
        sum_text (str): Case narrative text to summarize
        
    Returns:
        str: Bullet-point summary (up to 5 points)
        
    Raises:
        RuntimeError: If model is not loaded or generation fails
    """
    _ensure_loaded()

    # keep input short and stable
    sum_text = sum_text[:800]

    # (Llama 3.x is chatty; use a tiny, stable prompt that yields bullets)
    prompt = (
        "Summarize the case in exactly 5 short bullet points.\n"
        "Use compact language. No extra commentary.\n\n"
        f"{sum_text}\n\n"
        "Bullets:\n"
    )

    enc = _tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if _mdl.device.type == "cuda":
        enc = {k: v.to(_mdl.device) for k, v in enc.items()}

    # Clamp generation to keep it snappy
    gen_kwargs = dict(
        max_new_tokens=96,
        do_sample=False,
        pad_token_id=_tok.eos_token_id
    )

    with torch.inference_mode():
        out = _mdl.generate(
            **enc,
            **gen_kwargs,
            eos_token_id=_tok.eos_token_id,
            use_cache=True
        )

    gen = out[0, enc["input_ids"].shape[-1]:]
    text = _tok.decode(gen, skip_special_tokens=True)

    # Hard early stop – keep first 5 bullet lines if model adds extras
    bullets = [ln for ln in text.splitlines() if ln.strip().startswith(("•","-","*"))][:5]
    return "\n".join(bullets) if bullets else text


def batch_summarize(texts: list, batch_size: int = 8) -> list:
    """
    Batch process multiple summaries for efficiency.
    
    This function processes multiple case narratives in sequence,
    generating summaries for each. It uses optimized generation
    settings for speed while maintaining quality.
    
    Args:
        texts (list): List of case narrative texts to summarize
        batch_size (int): Number of texts to process (currently processes sequentially)
        
    Returns:
        list: List of bullet-point summaries, one per input text
        
    Raises:
        RuntimeError: If model is not loaded or generation fails
    """
    _ensure_loaded()
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Process batch
        batch_results = []
        for text in batch:
            # Trim input for speed
            text = _trim_to_tokens(text, _tok, max_input_tokens=600)
            
            prompt = (
                "Summarize the case in exactly 5 short bullet points.\n"
                "Use compact language. No extra commentary.\n\n"
                f"Case: {text}\n\nSummary:"
            )
            
            enc = _tok(prompt, return_tensors="pt", truncation=True, max_length=800)
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            
            # Optimized generation settings
            gen_kwargs = {
                "max_new_tokens": 64,  # Reduced for speed
                "do_sample": False,     # Greedy decoding
                "num_beams": 1,        # No beam search
                "temperature": 0.0,     # Deterministic
                "pad_token_id": _tok.eos_token_id
            }
            
            with torch.no_grad():
                out = _mdl.generate(
                    **enc,
                    **gen_kwargs,
                    eos_token_id=_tok.eos_token_id,
                    use_cache=True
                )
            
            gen = out[0, enc["input_ids"].shape[-1]:]
            text = _tok.decode(gen, skip_special_tokens=True)
            
            # Hard early stop – keep first 5 bullet lines
            bullets = [ln for ln in text.splitlines() if ln.strip().startswith(("•","-","*"))][:5]
            batch_results.append("\n".join(bullets) if bullets else text)
        
        results.extend(batch_results)
    
    return results
