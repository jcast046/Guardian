"""Weak labeling module for movement classification and risk assessment.

This module provides automated labeling capabilities for case narratives using the
Qwen2.5-3B-Instruct model. It classifies movement patterns and assesses risk
levels to support investigative workflows and case prioritization.

The weak labeler combines LLM-based classification with rule-based overlays to
provide consistent, automated labeling for case categorization, risk assessment,
and workflow optimization.

Author: Joshua Castillo

Example:
    >>> from guardian_llm.weak_labeler import classify_movement, label_case
    >>> movement = classify_movement("Suspect traveled from NYC to LA...")
    >>> labels = label_case("Case narrative...")
    >>> print(labels['movement'], labels['risk'])
"""
import json, re, torch, os
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, pipeline
try:
    from .prompts import MOVEMENT_CLASSIFICATION_PROMPT, RISK_ASSESSMENT_PROMPT
except ImportError:
    # Fallback for direct execution
    MOVEMENT_CLASSIFICATION_PROMPT = "Classify the movement pattern in this case narrative."
    RISK_ASSESSMENT_PROMPT = "Assess the risk level of this case narrative."

# Risk keywords for rule-based scoring
_RISK_KEYWORDS = {
    "Critical": [r'\babduct(ed|ion)\b', r'\bkidnap(ped|ping)?\b', r'\bweapon\b', r'\bgun\b', r'\bknife\b', r'\bbrandish', r'\bstrangle', r'\bimmediate danger', r'\bthreat(en(ed|ing)?)?\b', r'\bcoerc(ion|ed|ive)\b'],
    "High":     [r'\blure|\bgroom(ing|ed)\b|\bcoax|\bentice', r'\bthreat', r'\bassault', r'\bviolence', r'\bcoercion', r'\bforce(d)? (into|into a|into an) vehicle', r'\btraffick(ing|er|ed)\b', r'\bnon-?custodial\b', r'\brestraining order\b', r'\b(interstate|out of state|cross(-|\s)state)\b']
}

# Add/confirm HIGH-impact signals:
HIGH_SIGNS = [
    r"\babduct(ed|ion)\b", r"\bkidnap(ped|ping)?\b",
    r"\bweapon\b", r"\bgun\b", r"\bknife\b",
    r"\bthreat(en(ed|ing)?)?\b", r"\bcoerc(ion|ed|ive)\b",
    r"\bgroom(ing|ed)\b", r"\btraffick(ing|er|ed)\b",
    r"\bnon-?custodial\b", r"\brestraining order\b",
    r"\b(interstate|out of state|cross(-|\s)state)\b",
]

INTERSTATE = re.compile(r'\bI-\d{1,3}\b', re.I)

# =============================================================================
# BATCH API FUNCTIONS
# =============================================================================

def load_weak_labeler(model_id_or_dir: str, device_map: str = "auto"):
    """Load Qwen2.5-3B-Instruct model for batch processing.
    
    Loads the model once and returns a pipeline for repeated inference.
    Optimizes for GPU memory usage with 4-bit quantization when available.
    
    Args:
        model_id_or_dir: Model path or HuggingFace model ID
        device_map: Device mapping for model loading ("auto", "cpu", "cuda")
        
    Returns:
        HuggingFace pipeline configured for text generation with sampling
        
    Raises:
        Exception: If model loading fails or CUDA is unavailable
        
    Example:
        >>> pipe = load_weak_labeler("Qwen/Qwen2.5-3B-Instruct")
        >>> results = pipe(["Case narrative 1", "Case narrative 2"])
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir, trust_remote_code=True)
    
    # Load model with quantization if available
    if torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_dir,
                torch_dtype="auto",
                device_map=device_map,
                quantization_config=bnb_config,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[WARN] 4-bit quantization failed, trying without: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_dir,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_dir,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        )
    
    # Create pipeline for text generation with sampling enabled
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        do_sample=True,  # Enable sampling for diversity
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        return_full_text=False
    )
    return pipe

def _prompt_for_labeling(text: str) -> str:
    """Build formatted prompt for weak labeling.
    
    Creates a structured prompt for the Qwen2.5-3B model to classify movement
    patterns and assess risk levels from case narratives.
    
    Args:
        text: Case narrative text to analyze
        
    Returns:
        Formatted prompt string for the model
        
    Note:
        Input text is automatically trimmed to fit token limits
    """
    # Trim narrative to fit within token limits
    trimmed_narrative = _trim_to_tokens(text, _TOKENIZER, max_input_tokens=1000) if _TOKENIZER else text
    
    return f"""<|im_start|>system
You are a concise risk assessor. Return JSON with keys: movement, risk in {{Low,Medium,High,Critical}}.
<|im_end|>
<|im_start|>user
Narrative:
{trimmed_narrative}

Return JSON only in this exact format:
{{"movement": "Stationary", "risk": "Medium"}}
<|im_end|>
<|im_start|>assistant
"""

def _parse_labeler_json(s: str) -> Dict[str, str]:
    """Parse JSON output from the weak labeler model.
    
    Robustly extracts movement and risk labels from model output, handling
    common JSON formatting issues and malformed responses.
    
    Args:
        s: Raw model output text containing JSON
        
    Returns:
        Dictionary with 'movement' and 'risk' keys, defaults to "Unknown" if parsing fails
        
    Note:
        Uses regex fallback when JSON parsing fails
    """
    import json, re
    
    # Clean up the text first - remove newlines and fix common issues
    cleaned = s.replace('\n', ' ').replace('\r', ' ')
    
    # Try to extract JSON object from the model output robustly
    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not m:
        # Try to extract movement and risk using regex patterns
        movement_match = re.search(r'"movement":\s*"([^"]*)"', cleaned)
        risk_match = re.search(r'"risk":\s*"([^"]*)"', cleaned)
        
        movement = movement_match.group(1) if movement_match else "Unknown"
        risk = risk_match.group(1) if risk_match else "Unknown"
        
        return {"movement": movement, "risk": risk}
    
    # Try to fix common JSON issues before parsing
    json_text = m.group(0)
    
    # Fix missing quotes and values
    json_text = re.sub(r'"movement":\s*,', '"movement": "Unknown",', json_text)
    json_text = re.sub(r'"movement":\s*$', '"movement": "Unknown"', json_text)
    json_text = re.sub(r'"movement":\s*"([^"]*)$', r'"movement": "\1"', json_text)
    
    # Try to parse the fixed JSON
    try:
        obj = json.loads(json_text)
        movement = obj.get("movement", "Unknown")
        risk = obj.get("risk", "Unknown")
        
        return {"movement": str(movement), "risk": str(risk)}
    except:
        # Fall back to regex extraction
        movement_match = re.search(r'"movement":\s*"([^"]*)"', cleaned)
        risk_match = re.search(r'"risk":\s*"([^"]*)"', cleaned)
        
        movement = movement_match.group(1) if movement_match else "Unknown"
        risk = risk_match.group(1) if risk_match else "Unknown"
        
        return {"movement": movement, "risk": risk}

def weak_label_batch(pipe, texts: List[str]) -> List[Dict[str, str]]:
    """Batch process multiple case narratives for movement and risk labeling.
    
    Processes multiple case narratives efficiently using a pre-loaded pipeline.
    Applies rule-based overlays to LLM results for improved accuracy.
    
    Args:
        pipe: HuggingFace pipeline loaded with load_weak_labeler
        texts: List of case narrative texts to process
        
    Returns:
        List of dictionaries containing movement, risk, and timing information
        
    Note:
        Processing time is tracked and included in results
    """
    import time
    
    prompts = [_prompt_for_labeling(t) for t in texts]
    
    # Track processing time for the batch
    start_time = time.time()
    
    try:
        # Batch the pipeline call (Transformers supports list inputs)
        outs = pipe(prompts)
        
        batch_time = time.time() - start_time
        per_case_time = batch_time / len(texts) if texts else 0.0
        
        # Parse results
        results = []
        for i, out in enumerate(outs):
            # typical structure: out[0]["generated_text"]
            gen = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
            parsed = _parse_labeler_json(gen)
            
            # Apply rule overlays after LLM parsing
            original_narrative = texts[i] if i < len(texts) else ""
            movement = parsed.get("movement", "Unknown")
            risk = parsed.get("risk", "Unknown")
            
            # Apply movement rule overlay
            movement = _rule_adjust_movement(original_narrative, movement)
            
            # Apply risk rule overlay 
            entities = {"age": None, "risk_factors": []}
            risk = apply_risk_overlay(risk, entities, original_narrative, debug=bool(os.getenv("GUARDIAN_DEBUG_RULES")))
            
            # Update parsed results with rule overlays
            parsed["movement"] = movement
            parsed["risk"] = risk
            parsed["time"] = per_case_time
            results.append(parsed)
        
        return results
        
    except Exception as e:
        print(f"[ERROR] weak_label_batch failed: {e}")
        # Return fallback results
        return [{"movement": "Unknown", "risk": "Unknown", "time": 0.0} for _ in texts]

# Load model once at import time for maximum efficiency
_MODEL_DIR = None
_TOKENIZER = None
_MODEL = None

def _ensure_loaded():
    """
    Load model once and cache globally.
    
    This function implements lazy loading - the model is only loaded when
    first needed, and then cached for subsequent calls to avoid reloading.
    Uses 4-bit quantization for memory efficiency.
    
    Returns:
        tuple: (tokenizer, model) tuple
        
    Raises:
        FileNotFoundError: If model directory is not found
        RuntimeError: If model loading fails
    """
    global _MODEL_DIR, _TOKENIZER, _MODEL
    
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    
    # Use local model directory
    _MODEL_DIR = DIR_WEAK_LABELER
    
    # Validate model directory exists
    p = Path(_MODEL_DIR)
    if not p.exists() or not (p / "config.json").exists():
        raise FileNotFoundError(
            f"Model dir '{p}' is missing or lacks config.json. "
            "Point to the exact directory that contains the model files."
        )
    
    print(f"[INIT] Using model dir: {Path(_MODEL_DIR).resolve()}")
    
    # Load tokenizer with optimized padding for batch generation
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_DIR, use_fast=True, padding_side="left", local_files_only=True)
        print(f"[CHK]  Tokenizer path: {_TOKENIZER.name_or_path}")
        # Set proper pad token for decoder-only models
        _TOKENIZER.pad_token = _TOKENIZER.eos_token if _TOKENIZER.pad_token is None else _TOKENIZER.pad_token
        _TOKENIZER.padding_side = "left"
    except Exception as e:
        print(f"[ERROR] Tokenizer loading failed: {e}")
        return None
    
    # Set optimized math precision
    torch.set_float32_matmul_precision("high")
    
    # 4-bit quantization for 3B model - simple and effective
    if torch.cuda.is_available():
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        try:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                _MODEL_DIR,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb,
                attn_implementation="eager",  # Use eager attention for Windows compatibility
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Disable FlashAttention 2 on Windows
            if hasattr(_MODEL.config, "use_flash_attention_2"):
                _MODEL.config.use_flash_attention_2 = False
        except Exception as e:
            print(f"[WARN] 4-bit quantization failed, trying without quantization: {e}")
            try:
                # Fallback to unquantized GPU with SDPA attention
                _MODEL = AutoModelForCausalLM.from_pretrained(
                    _MODEL_DIR,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",  # Use eager attention for Windows compatibility
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                # Disable FlashAttention 2 on Windows
                if hasattr(_MODEL.config, "use_flash_attention_2"):
                    _MODEL.config.use_flash_attention_2 = False
            except Exception as e2:
                print(f"[WARN] GPU loading failed, falling back to CPU: {e2}")
                # Final fallback to CPU
                _MODEL = AutoModelForCausalLM.from_pretrained(
                    _MODEL_DIR,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=True
                )
    else:
        # CPU fallback
        _MODEL = AutoModelForCausalLM.from_pretrained(
            _MODEL_DIR,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )
    
    print(f"[CHK]  Model name_or_path: {getattr(_MODEL.config, '_name_or_path', 'unknown')}")
    print(f"[CHK]  4-bit loaded: {getattr(_MODEL, 'is_loaded_in_4bit', False)}")
    print(f"[CHK]  Device map keys: {list(getattr(_MODEL, 'hf_device_map', {}).keys())[:5]}...")
    
    # Set pad token for efficient batching
    _MODEL.generation_config.pad_token_id = _TOKENIZER.eos_token_id
    
    # Optimize for speed
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    _MODEL.eval()
    return _TOKENIZER, _MODEL

# System prompt for consistent behavior
_SYSTEM = "You are a concise risk assessor. Return JSON with keys: movement, risk in {Low,Medium,High,Critical}."
_PROMPT = "Narrative:\n{narr}\n\nReturn JSON only."

def _build_prompt(narr: str) -> str:
    """
    Build optimized prompt for batch processing using Qwen2.5 chat format.
    
    Args:
        narr (str): Case narrative text
        
    Returns:
        str: Formatted prompt with system and user messages
    """
    # Trim narrative to fit within token limits
    trimmed_narrative = _trim_to_tokens(narr, _TOKENIZER, max_input_tokens=1000) if _TOKENIZER else narr
    
    return f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\nNarrative:\n{trimmed_narrative}\n\nReturn JSON only in this exact format:\n{{\"movement\": \"Stationary\", \"risk\": \"Medium\"}}<|im_end|>\n<|im_start|>assistant\n"

def label_case(narrative: str) -> dict:
    """Label a single case narrative for movement and risk.
    
    Convenience function for processing individual case narratives.
    Returns structured results compatible with the zone_qa system.
    
    Args:
        narrative: Case narrative text to analyze
        
    Returns:
        Dictionary containing movement, risk, and metadata
        
    Example:
        >>> result = label_case("Missing person last seen in NYC...")
        >>> print(result['movement'], result['risk'])
    """
    return label_batch([narrative])[0]

@torch.inference_mode()
def label_batch(narratives: list[str], batch_size: int = 64, max_new_tokens: int = 24) -> list[dict]:
    """Batch process multiple narratives using direct model calls.
    
    Processes multiple case narratives efficiently using optimized generation
    settings. Returns structured results compatible with the zone_qa system.
    
    Args:
        narratives: List of case narrative texts to process
        batch_size: Number of narratives to process in each batch
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        List of dictionaries containing:
            - plausibility: Risk score between 0.0 and 1.0
            - rationale: Explanation of the assessment
            - __labeler_source__: Source of the labeling ("real" or "fallback")
        
    Raises:
        RuntimeError: If model loading or generation fails
        
    Note:
        Uses torch.inference_mode() for optimal performance
    """
    try:
        result = _ensure_loaded()
        if result is None:
            raise ValueError("_ensure_loaded returned None")
        tokenizer, model = result
        
        out = []
        for i in range(0, len(narratives), batch_size):
            chunk = narratives[i:i+batch_size]
            prompts = [_build_prompt(n) for n in chunk]
            
            # Add small randomness seed difference per batch for diversity
            import random
            torch.manual_seed(random.randint(1, 10000))

            # Tokenize with optimized padding for batch processing
            toks = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=1024,        # Keep context modest for speed
                return_tensors="pt"
            )
            toks = {k: v.to(model.device) for k, v in toks.items()}

            # Generate with optimized settings for speed and diversity
            gen_cfg = GenerationConfig(
                max_new_tokens=min(max_new_tokens, 64),  # Increased for better responses
                do_sample=True,        # Enable sampling for temperature/top_p to work
                temperature=0.9,       # Higher variety for better diversity
                top_p=0.95,            # Nucleus sampling with higher threshold
                top_k=50,              # Top-k sampling
                num_beams=1,           # No beam search for speed
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,        # KV cache for speed
            )
            gen = model.generate(**toks, generation_config=gen_cfg)

            # Slice off the input portion to get only the new tokens
            out_ids = gen[:, toks["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            # Parse results with fallback and convert to expected format
            for i, t in enumerate(texts):
                result = _parse_json(t)
                if not result:  # Fallback if JSON parsing fails
                    result = _fallback_parse(t)
                
                # Apply rule overlays after LLM parsing
                original_narrative = chunk[i] if i < len(chunk) else ""
                movement = result.get("movement", "Unknown")
                risk = result.get("risk", "Unknown")
                
                # Apply movement rule overlay
                movement = _rule_adjust_movement(original_narrative, movement)
                
                # Apply risk rule overlay (create dummy entities dict for compatibility)
                entities = {"age": None, "risk_factors": []}
                risk = apply_risk_overlay(risk, entities, original_narrative, debug=bool(os.getenv("GUARDIAN_DEBUG_RULES")))
                
                # Convert to expected format for zone_qa.py
                risk2pl = {"Critical": 0.9, "High": 0.7, "Medium": 0.5, "Low": 0.3}
                plausibility = risk2pl.get(risk, 0.5)
                
                out.append({
                    "plausibility": plausibility,
                    "rationale": f"LLM assessment: {risk} risk, movement: {movement}",
                    "__labeler_source__": "real"
                })
        
        return out
        
    except Exception as e:
        print(f"[ERROR] label_batch failed: {e}")
        # Return fallback results for each narrative
        return [
            {
                "plausibility": 0.5,
                "rationale": f"Fallback due to error: {str(e)}",
                "__labeler_source__": "fallback"
            }
            for _ in narratives
        ]

def _parse_json(text: str) -> dict:
    """
    Parse JSON from model output with fallback.
    
    Args:
        text (str): Model output text to parse
        
    Returns:
        dict: Parsed JSON object or empty dict if parsing fails
    """
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end]
            return json.loads(json_str)
    except:
        pass
    return {}

def _fallback_parse(text: str) -> dict:
    """
    Fallback parsing if JSON fails.
    
    This function provides a simple keyword-based fallback when JSON parsing
    fails, extracting risk level from text content.
    
    Args:
        text (str): Model output text to parse
        
    Returns:
        dict: Dictionary with movement and risk fields
    """
    risk = "Medium"
    movement = "Unknown"
    
    text_lower = text.lower()
    if "critical" in text_lower:
        risk = "Critical"
    elif "high" in text_lower:
        risk = "High"
    elif "low" in text_lower:
        risk = "Low"
    
    return {"movement": movement, "risk": risk}

def _risk_by_rules(text: str) -> str:
    """
    Rule-based risk assessment with decisive escalation for high-risk scenarios.
    
    This function implements a comprehensive rule-based scoring system that
    considers victim age, explicit danger indicators, vehicle involvement,
    reporting delays, and highway movement patterns.
    
    Args:
        text (str): Case narrative text to score
        
    Returns:
        str: Risk level ("Low", "Medium", "High", "Critical")
    """
    score = 0
    # victim age
    m = re.search(r'\b(\d{1,2})-year-old\b', text)
    if m:
        age = int(m.group(1))
        if age <= 8:  score += 3  # was 2
        elif age <= 12: score += 2

    # explicit danger words - give higher scores for high-impact signals
    for lvl, pats in _RISK_KEYWORDS.items():
        for p in pats:
            if re.search(p, text, re.I):
                score += 4 if lvl == "Critical" else 3
    
    # Additional high-impact signals that should add 2+ points each
    for pattern in HIGH_SIGNS:
        if re.search(pattern, text, re.I):
            score += 2

    # unknown adult + vehicle lure style phrasing
    if re.search(r'unknown adult|offered? a ride|offering a ride|into (a|an) vehicle', text, re.I):
        score += 3  # was 2
    
    # "entered vehicle" (not only "into a vehicle")
    if re.search(r'entered vehicle|got into (a|the) car|pulled into a car', text, re.I):
        score += 3

    # "offered a ride" variations
    if re.search(r'offered (her|him|them) a ride|asked.*to get in|told.*get in', text, re.I):
        score += 3

    # stronger bump for very young + highway
    if re.search(r'\bI-\d{1,3}\b', text) and re.search(r'\b(\d{1,2})-year-old\b', text):
        age = int(re.search(r'\b(\d{1,2})-year-old\b', text).group(1))
        if age <= 12: score += 1

    # long elapsed reporting for young victims
    # matches "Time to Report: 93 minutes" OR "reported missing 20 hours"
    if (m := re.search(r'Time to Report:\s*(\d+)\s*minutes', text, re.I)):
        mins = int(m.group(1))
        if mins >= 240: score += 2
        elif mins >= 120: score += 1
    elif (m := re.search(r'reported missing\s+(\d+)\s+hours', text, re.I)):
        hrs = int(m.group(1))
        if hrs >= 24: score += 2
        elif hrs >= 12: score += 1

    # high-speed corridor cues help bump severity
    if INTERSTATE.search(text):
        score += 1

    # Convert score to risk level
    if score >= 4:
        return "Critical"
    if score >= 2:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"

def _risk_from_score(score: int) -> str:
   
    # 0 -> Low, 1 -> Medium, 2-3 -> High, 4+ -> Critical
    if score >= 4:
        return "Critical"
    if score >= 2:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"

def rule_risk(entities: dict) -> str:
    """
    Rule-based risk calibration to prevent Medium monotony.
    
    This function provides a simplified risk scoring based on entity risk factors,
    designed to prevent the model from defaulting to "Medium" for all cases.
    
    Args:
        entities (dict): Entity dictionary with risk_factors and age fields
        
    Returns:
        str: Risk level ("Low", "Medium", "High")
    """
    rf = set(entities.get("risk_factors") or [])
    age = entities.get("age")
    score = 0
    if "violence_indicator" in rf: score += 2
    if "vehicle_lure" in rf:       score += 1
    if "very_young_victim" in rf:  score += 1
    if "delayed_report" in rf:     score += 1
    if "nighttime" in rf:          score += 1
    # quick map
    return "High" if score >= 3 else "Medium" if score >= 1 else "Low"

def apply_risk_overlay(llm_risk: str | None, entities: dict | None, text: str, *, debug: bool = False) -> str:
    """
    Apply rule overlay to LLM risk assessment with standardized signature.
    
    This function applies rule-based adjustments to LLM risk assessments to
    prevent monotonous risk labels and ensure proper escalation for high-risk scenarios.
    It considers victim age, explicit danger indicators, vehicle involvement,
    highway movement patterns, and reporting delays.
    
    Args:
        llm_risk (str | None): LLM-generated risk level
        entities (dict | None): Entity dictionary with age and risk_factors
        text (str): Case narrative text
        debug (bool): Enable debug output for rule application
        
    Returns:
        str: Final risk level after rule overlay
    """
    # normalize incoming
    risk = (llm_risk or "Unknown").strip().title()
    if risk not in {"Unknown","Low","Medium","High","Critical"}:
        risk = "Unknown"

    # quick keyword rules
    rules = _risk_by_rules(text)
    if rules == "Critical":
        if debug:
            print(f"[DEBUG] Rule upgrade: {risk} -> Critical (keyword match)")
        return "Critical"
    if rules == "High":
        if risk in {"Unknown","Low","Medium"}:
            if debug:
                print(f"[DEBUG] Rule upgrade: {risk} -> High (keyword match)")
            return "High"
        if debug:
            print(f"[DEBUG] Rule would upgrade to High but LLM already {risk}")

    # fallback
    if debug:
        print(f"[DEBUG] Final risk: {risk} (no rule upgrades)")
    return risk

# Enable TF32 for RTX 40xx speedup and optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load configuration from guardian.config.json
try:
    with open("guardian.config.json", "r") as f:
        config = json.load(f)
    DIR_WEAK_LABELER = config["models"]["weak_labeler"]
except:
    DIR_WEAK_LABELER = r"C:\Users\N0Cir\CS698\Guardian\models\Qwen2.5-3B-Instruct"

# Global model cache variables
_tok = _mdl = None

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


def _rule_adjust_movement(text: str, movement: str) -> str:
    """
    Apply rule-based adjustments to movement classification when clear cues exist.
    
    This function applies movement pattern adjustments based on text content
    to prevent the model from defaulting to "Stationary" when movement is
    clearly indicated in the narrative. Only overrides when clear cues exist.
    
    Args:
        text (str): Case narrative text
        movement (str): LLM-generated movement label
        
    Returns:
        str: Adjusted movement label
    """
    if INTERSTATE.search(text) or re.search(r'\b(interstate|out of state|cross[-\s]state)\b', text, re.I):
        return "Interstate"
    if re.search(r'\b(no travel|remains? in|still in town|same city)\b', text, re.I):
        return "Stationary"
    return movement if movement and movement != "Unknown" else "Unknown"

def classify_movement(narrative: str) -> str:
    """Classify movement pattern in case narrative.
    
    Analyzes a case narrative and classifies the movement pattern of persons or
    vehicles involved. Helps with case categorization and investigative workflow optimization.
    
    Args:
        narrative: The case narrative text to classify
        
    Returns:
        Movement classification from categories: "Stationary", "Local", "Regional",
        "Interstate", "International", or "Unknown"
        
    Raises:
        RuntimeError: If model is not loaded or generation fails
        
    Example:
        >>> narrative = "Suspect traveled from New York to Los Angeles..."
        >>> movement = classify_movement(narrative)
        >>> print(movement)  # "Interstate"
        
    Note:
        If no clear classification is found, returns "Unknown"
    """
    _ensure_loaded()
    
    # Trim input to ~800 tokens for faster processing
    trimmed_narrative = _trim_to_tokens(narrative, _tok, max_input_tokens=800)
    
    # Improved prompt with actual narrative
    improved_prompt = f"""Classify the movement in the following missing-person narrative. 
Choose one: Stationary, Local, Regional, Interstate, International, Unknown.

Narrative: {trimmed_narrative}

Movement:"""
    enc = _tok(improved_prompt, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    
    if torch.cuda.is_available(): 
        ids, mask = ids.to(_mdl.device), mask.to(_mdl.device)
    
    with torch.inference_mode():
        gen_cfg = GenerationConfig(
            max_new_tokens=96,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            pad_token_id=_tok.eos_token_id,
            eos_token_id=_tok.eos_token_id
        )
        out = _mdl.generate(
            input_ids=ids,
            attention_mask=mask,
            generation_config=gen_cfg
        )
    
    response = _tok.decode(out[0], skip_special_tokens=True)
    response = response.split("Movement Classification:")[-1].strip()
    
    # Extract classification from response
    for category in ["Stationary", "Local", "Regional", "Interstate", "International", "Unknown"]:
        if category.lower() in response.lower():
            llm_result = category
            break
    else:
        llm_result = "Unknown"
    
    # Apply rule-based adjustment to prevent obviously wrong defaults
    return _rule_adjust_movement(narrative, llm_result)

def assess_risk(narrative: str) -> str:
    """
    Assess the risk level of a missing person case.

    Returns exactly one of:
      "Low", "Medium", "High", "Critical", or "Unknown" (if unsure)

    Strategy:
      1) Query LLM with few-shot prompt that includes all 4 classes.
      2) Parse STRICTLY one of the allowed labels.
      3) Compute a rules score.
      4) Combine by taking the higher severity; if neither fires, return "Unknown".
    """
    _ensure_loaded()

    trimmed = _trim_to_tokens(narrative, _tok, max_input_tokens=800)

   
    prompt = f"""
You are labeling risk for a missing-person case.
Answer with exactly one word from this set: Low, Medium, High, Critical

Guidance:
- Low: minimal indicators of coercion/violence; likely voluntary/short-term absence
- Medium: some concerning context but no concrete threats or coercion
- High: clear indicators of coercion, grooming, trafficking intent, or credible threats
- Critical: immediate danger (weapons, explicit violent threats, abduction, medical crisis), or ongoing interstate transport

Examples:
Case A: "Wandered from home; returned next morning; no threats; safe contact." → Low
Case B: "Left with friends; phone off; rumors of party; no threats." → Medium
Case C: "Non-custodial adult coerced teen online; threats to harm family." → High
Case D: "Abducted at gunpoint; suspect driving out of state; active threats." → Critical

Case Narrative:
{trimmed}

Answer with one word only:
Risk:
""".strip()

    enc = _tok(prompt, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    if torch.cuda.is_available():
        ids, mask = ids.to(_mdl.device), mask.to(_mdl.device)

    allowed = {"Low", "Medium", "High", "Critical"}
    with torch.inference_mode():
        out = _mdl.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=16,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=_tok.eos_token_id,
        )
    text = _tok.decode(out[0], skip_special_tokens=True)

    # Strict parse: pick the FIRST allowed label that can be found; else "Unknown"
    m = re.search(r"\b(Low|Medium|High|Critical)\b", text, flags=re.IGNORECASE)
    llm_risk = m.group(1).title() if m else "Unknown"

    # Rules overlay 
    score = _risk_by_rules(narrative)
    rule_risk = _risk_from_score(score)

    # Combine: take the maximum severity; if both Unknown/Low signal, return the non-Unknown one
    def rank(x: str) -> int:
        order = {"Unknown": -1, "Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        return order.get(x, -1)

    final = llm_risk if rank(llm_risk) >= rank(rule_risk) else rule_risk

    # Optional: tie movement to a floor for risk when LLM is uncertain
    try:
        mv = classify_movement(narrative)  # or however you fetch movement for this case
    except Exception:
        mv = None

    if final in {"Unknown", "Low"} and mv:
        floor = {"Stationary": "Low", "Local": "Medium", "Regional": "High", "Interstate": "High"}
        floored = floor.get(mv, final)
        if rank(floored) > rank(final):
            final = floored

    # If both are Unknown (or both Low with zero signal), return Unknown, not Medium.
    if final == "Low" and score == 0 and llm_risk in {"Unknown", "Low"}:
        return "Low"
    if final == "Unknown":
        return "Unknown"
    return final


def batch_label_cases(narratives: list, batch_size: int = 2) -> list:
    """
    Label multiple cases in batch for better GPU utilization.
    
    This function processes multiple case narratives in batches and returns labels for each.
    With 4-bit quantization, batch_size=2 is typically safe for 8GB cards.
    
    Args:
        narratives (list): List of case narrative strings to label
        batch_size (int): Number of narratives to process in each batch
        
    Returns:
        list: List of dictionaries, each containing movement and risk labels:
            [
                {"movement": str, "risk": str},
                {"movement": str, "risk": str},
                ...
            ]
            
    Raises:
        RuntimeError: If model is not loaded or generation fails
        
    Example:
        >>> narratives = ["Case 1 narrative...", "Case 2 narrative..."]
        >>> labels = batch_label_cases(narratives, batch_size=2)
        >>> print(labels[0]["movement"])  # "Local"
    """
    _ensure_loaded()
    results = []
    
    for i in range(0, len(narratives), batch_size):
        batch = narratives[i:i + batch_size]
        
        # Process movement classification batch
        movement_prompts = []
        for narrative in batch:
            trimmed_narrative = _trim_to_tokens(narrative, _tok, max_input_tokens=800)
            prompt = f"""Classify the movement in the following missing-person narrative. 
Choose one: Stationary, Local, Regional, Interstate, International, Unknown.

Narrative: {trimmed_narrative}

Movement:"""
            movement_prompts.append(prompt)
        
        # Tokenize and generate movement batch
        enc = _tok(movement_prompts, return_tensors="pt", padding=True, truncation=True, max_length=800)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        
        if torch.cuda.is_available():
            ids, mask = ids.to(_mdl.device), mask.to(_mdl.device)
        
        with torch.inference_mode():
            gen_cfg = GenerationConfig(
                max_new_tokens=96,
                do_sample=True,
            temperature=0.9,
            top_p=0.95,
                top_k=50,
                pad_token_id=_tok.eos_token_id,
                eos_token_id=_tok.eos_token_id
            )
            out = _mdl.generate(
                input_ids=ids, attention_mask=mask,
                generation_config=gen_cfg
            )
        
        # Decode movement results
        movements = []
        for j, generated in enumerate(out):
            response = _tok.decode(generated, skip_special_tokens=True)
            response = response.split("Movement Classification:")[-1].strip()
            
            for category in ["Stationary", "Local", "Regional", "Interstate", "International", "Unknown"]:
                if category.lower() in response.lower():
                    llm_result = category
                    break
            else:
                llm_result = "Unknown"
            
            # Apply rule-based adjustment
            movements.append(_rule_adjust_movement(batch[j], llm_result))
        
        # Process risk assessment batch
        risk_prompts = []
        for narrative in batch:
            trimmed_narrative = _trim_to_tokens(narrative, _tok, max_input_tokens=800)
            prompt = f"""Assess the risk level (Low, Medium, High, Critical) for this case.

Narrative: {trimmed_narrative}

Risk:"""
            risk_prompts.append(prompt)
        
        # Tokenize and generate risk batch
        enc = _tok(risk_prompts, return_tensors="pt", padding=True, truncation=True, max_length=800)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        
        if torch.cuda.is_available():
            ids, mask = ids.to(_mdl.device), mask.to(_mdl.device)
        
        with torch.inference_mode():
            gen_cfg = GenerationConfig(
                max_new_tokens=96,
                do_sample=True,
            temperature=0.9,
            top_p=0.95,
                top_k=50,
                pad_token_id=_tok.eos_token_id,
                eos_token_id=_tok.eos_token_id
            )
            out = _mdl.generate(
                input_ids=ids, attention_mask=mask,
                generation_config=gen_cfg
            )
        
        # Decode risk results
        risks = []
        for j, generated in enumerate(out):
            response = _tok.decode(generated, skip_special_tokens=True)
            response = response.split("Risk Assessment:")[-1].strip()
            
            for level in ["Low", "Medium", "High", "Critical"]:
                if level.lower() in response.lower():
                    llm_level = level
                    break
            else:
                llm_level = "Medium"
            
            # Apply rule-based scoring and choose higher severity
            rule_level = _risk_by_rules(batch[j])
            order = {"Low":0, "Medium":1, "High":2, "Critical":3}
            final_level = max([llm_level, rule_level], key=lambda k: order[k])
            risks.append(final_level)
        
        # Combine results
        for movement, risk in zip(movements, risks):
            results.append({"movement": movement, "risk": risk})
    
    return results
