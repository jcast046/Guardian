"""Guardian LLM Analysis Package.

This package provides comprehensive LLM-based analysis capabilities for the Guardian
missing person case management system. It includes summarization, entity extraction,
weak labeling, and fine-tuning modules optimized for investigative workflows.

Modules:
    summarizer: Case summarization using Llama-3.2-3B-Instruct
    extractor: Entity extraction using Qwen2.5-3B-Instruct  
    weak_labeler: Movement classification and risk assessment
    finetune_qlora: Model fine-tuning utilities 
Author: Joshua Castillo
"""

# Core runtime functions (no heavy dependencies)
from .summarizer import summarize, batch_summarize
from .extractor import (
    extract_entities, extract_persons, extract_vehicles, extract_locations
)
from .weak_labeler import classify_movement, assess_risk, label_case, label_batch

# Optional fine-tuning functions (requires peft library)
try:
    from .finetune_qlora import (
        GuardianFineTuner,
        fine_tune_summarizer,
        fine_tune_extractor,
        fine_tune_weak_labeler,
    )
except Exception:
    GuardianFineTuner = None
    fine_tune_summarizer = None
    fine_tune_extractor = None
    fine_tune_weak_labeler = None
