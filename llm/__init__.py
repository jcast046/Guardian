# llm/__init__.py

# === Runtime (no heavy deps) ===
from .summarizer import summarize
from .extractor import (
    extract_entities, extract_persons, extract_vehicles, extract_locations
)
from .weak_labeler import classify_movement, assess_risk, label_case

# === Optional: fine-tuning (requires peft); safe if missing ===
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
