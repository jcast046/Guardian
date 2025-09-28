"""
Prompt templates for Guardian LLM modules.

This module contains standardized prompt templates for various LLM tasks in the Guardian
system, including entity extraction, summarization, weak labeling, and fine-tuning.
All prompts are designed to work with the specific models configured in guardian.config.json.

Author: Joshua Castillo
"""

# Entity Extraction Prompts
# Template for extracting structured entities from case narratives
# Used by Qwen2-7B-Instruct model for JSON entity extraction
EXTRACTION_PROMPT = """Extract entities from the following case narrative in JSON format. Focus on:
- Persons (names, descriptions, roles)
- Vehicles (make, model, color, license plates)
- Locations (addresses, landmarks, coordinates)
- Timeline (dates, times, sequence of events)
- Evidence (physical evidence, digital evidence)

Case Narrative:
{narrative}

Return only valid JSON with this structure:
{{
  "persons": [{{"name": "string", "description": "string", "role": "string"}}],
  "vehicles": [{{"make": "string", "model": "string", "color": "string", "license": "string"}}],
  "locations": [{{"address": "string", "landmark": "string", "coordinates": "string"}}],
  "timeline": [{{"date": "string", "time": "string", "event": "string"}}],
  "evidence": [{{"type": "string", "description": "string", "location": "string"}}]
}}"""

# Summarization Prompts
# Template for generating investigator-focused case summaries
# Used by Llama-3 8B models (both instruct and base variants)
SUMMARY_PROMPT = """Summarize the following case for investigators in 5 concise bullet points. Be factual, avoid speculation.

Case Narrative:
{narrative}

Summary:"""

# Weak Labeling Prompts
# Template for classifying movement patterns in case narratives
# Used by Mistral-7B-Instruct-v0.2 for movement classification
MOVEMENT_CLASSIFICATION_PROMPT = """Classify the movement pattern described in this case narrative. Choose the most appropriate category:

Categories:
- Stationary: Person/vehicle remained in one location
- Local: Short-distance movement within neighborhood/city
- Regional: Medium-distance movement within state/region
- Interstate: Long-distance movement across state lines
- International: Movement across country borders
- Unknown: Insufficient information to determine

Case Narrative:
{narrative}

Movement Classification:"""

# Template for assessing risk levels in case narratives
# Used by Mistral-7B-Instruct-v0.2 for risk assessment
RISK_ASSESSMENT_PROMPT = """Assess the risk level of this case narrative. Consider factors like:
- Violence indicators
- Weapon involvement
- Suspect behavior patterns
- Victim vulnerability
- Evidence quality

Risk Levels:
- Low: Minimal threat indicators
- Medium: Some concerning factors present
- High: Multiple threat indicators
- Critical: Immediate danger indicators

Case Narrative:
{narrative}

Risk Assessment:"""

# Fine-tuning Prompts
# Conversation templates for fine-tuning models on synthetic data
# Uses chat template format compatible with instruct-tuned models

# Template for fine-tuning summarizer models
FINE_TUNE_SUMMARY_PROMPT = """<|im_start|>user
Summarize this case for investigators: {narrative}
<|im_end|>
<|im_start|>assistant
{summary}
<|im_end|>"""

# Template for fine-tuning entity extraction models
FINE_TUNE_EXTRACTION_PROMPT = """<|im_start|>user
Extract entities from this case: {narrative}
<|im_end|>
<|im_start|>assistant
{extraction}
<|im_end|>"""

# Template for fine-tuning weak labeling models
FINE_TUNE_LABELING_PROMPT = """<|im_start|>user
Classify this case: {narrative}
<|im_end|>
<|im_start|>assistant
Movement: {movement}
Risk: {risk}
<|im_end|>"""
