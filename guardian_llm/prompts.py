"""Prompt templates for Guardian LLM modules.

This module contains standardized prompt templates for various LLM tasks in the Guardian
system, including entity extraction, summarization, weak labeling, and fine-tuning.
All prompts are designed to work with the specific models configured in guardian.config.json.

Author: Joshua Castillo

Constants:
    EXTRACTION_PROMPT: Template for entity extraction tasks
    SUMMARY_PROMPT: Template for case summarization tasks  
    MOVEMENT_CLASSIFICATION_PROMPT: Template for movement pattern classification
    RISK_ASSESSMENT_PROMPT: Template for risk level assessment
    FINE_TUNE_SUMMARY_PROMPT: Template for fine-tuning summarizer models
    FINE_TUNE_EXTRACTION_PROMPT: Template for fine-tuning extractor models
    FINE_TUNE_LABELING_PROMPT: Template for fine-tuning weak labeler models
"""
import json
from pathlib import Path
from typing import Dict, List, Set

def _load_psychology_findings() -> Dict[str, List[str]]:
    """
    Load and aggregate psychology research findings from all psych_examples JSON files.
    
    Returns:
        Dictionary with keys: 'risk_factors', 'behavioral_indicators', 'movement_patterns',
        'motive_patterns', 'temporal_patterns' containing aggregated lists of findings.
    """
    base_path = Path("data/psych_research")
    findings = {
        'risk_factors': [],
        'behavioral_indicators': [],
        'movement_patterns': [],
        'motive_patterns': [],
        'temporal_patterns': []
    }
    
    # Load all psych_examples files (1-13)
    for i in range(1, 14):
        file_path = base_path / f"psych_examples{i}.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    findings_data = data.get('findings', {})
                    
                    # Aggregate risk factors
                    if 'risk_factors' in findings_data:
                        risk_factors = findings_data['risk_factors']
                        if isinstance(risk_factors, list):
                            findings['risk_factors'].extend(risk_factors)
                    
                    # Aggregate behavioral indicators
                    if 'behavioral_indicators' in findings_data:
                        indicators = findings_data['behavioral_indicators']
                        if isinstance(indicators, list):
                            findings['behavioral_indicators'].extend(indicators)
                    
                    # Aggregate movement patterns
                    if 'movement_patterns' in findings_data:
                        movement = findings_data['movement_patterns']
                        if isinstance(movement, dict):
                            if 'typical_distances' in movement and movement['typical_distances']:
                                findings['movement_patterns'].append(movement['typical_distances'])
                            if 'common_routes' in movement and movement['common_routes']:
                                findings['movement_patterns'].extend(movement['common_routes'])
                            if 'transportation_methods' in movement and movement['transportation_methods']:
                                findings['movement_patterns'].extend(movement['transportation_methods'])
                    
                    # Aggregate motive patterns
                    if 'motive_patterns' in findings_data:
                        motives = findings_data['motive_patterns']
                        if isinstance(motives, dict):
                            if 'decision_factors' in motives and motives['decision_factors']:
                                findings['motive_patterns'].extend(motives['decision_factors'])
                            if 'risk_escalation' in motives and motives['risk_escalation']:
                                findings['motive_patterns'].append(motives['risk_escalation'])
                    
                    # Aggregate temporal patterns
                    if 'temporal_patterns' in findings_data:
                        temporal = findings_data['temporal_patterns']
                        if isinstance(temporal, dict):
                            if 'time_of_day' in temporal and temporal['time_of_day']:
                                findings['temporal_patterns'].extend(temporal['time_of_day'])
                            if 'escalation_timeline' in temporal and temporal['escalation_timeline']:
                                findings['temporal_patterns'].append(temporal['escalation_timeline'])
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                # Silently skip files that can't be loaded
                continue
    
    # Deduplicate and clean findings
    for key in findings:
        # Remove duplicates while preserving order
        seen = set()
        unique_findings = []
        for item in findings[key]:
            # Normalize string (lowercase, strip)
            normalized = item.lower().strip() if isinstance(item, str) else str(item)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_findings.append(item)
        findings[key] = unique_findings
    
    return findings

# Load psychology findings once at module import
_PSYCH_FINDINGS = _load_psychology_findings()

# Entity Extraction Prompts
# Template for extracting structured entities from case narratives
# Used by Qwen2.5-3B-Instruct model for JSON entity extraction
# Enhanced with psychology research findings on behavioral indicators
EXTRACTION_PROMPT = """Extract entities from the following case narrative in JSON format. Focus on:
- Persons (names, descriptions, roles, relationship to victim: family/acquaintance/stranger)
- Vehicles (make, model, color, license plates)
- Locations (addresses, landmarks, coordinates)
- Timeline (dates, times, sequence of events)
- Evidence (physical evidence, digital evidence)

Also extract behavioral indicators from psychology research:
- Relationship context: Family member, acquaintance, neighbor, stranger, authority figure
- Grooming indicators: Prior contact, repeated visits, boundary violations, special attention
- Behavioral red flags: Evasion, equivocation, prior familiarity with residence, unsecured entry points
- Temporal markers: Time of day (overnight 12am-5am, afternoon 2pm-9pm), escalation timeline
- Risk factors: Child age, offender relationship, prior criminal history, household instability

Case Narrative:
{narrative}

Return only valid JSON with this structure:
{{
  "persons": [{{"name": "string", "description": "string", "role": "string", "relationship": "string"}}],
  "vehicles": [{{"make": "string", "model": "string", "color": "string", "license": "string"}}],
  "locations": [{{"address": "string", "landmark": "string", "coordinates": "string"}}],
  "timeline": [{{"date": "string", "time": "string", "event": "string"}}],
  "evidence": [{{"type": "string", "description": "string", "location": "string"}}],
  "behavioral_indicators": ["string"],
  "relationship_context": "string"
}}"""

# Summarization Prompts
# Template for generating investigator-focused case summaries
# Used by Llama-3 8B models (both instruct and base variants)
# Enhanced to emphasize psychology-relevant risk factors and behavioral indicators
SUMMARY_PROMPT = """Summarize the following case for investigators in 5 concise bullet points. Be factual, avoid speculation.

Emphasize key psychology-relevant information:
- Offender relationship to victim (family/acquaintance/stranger - known relationships pose higher risk)
- Child age and vulnerability factors
- Behavioral indicators (grooming, prior familiarity, temporal patterns)
- Risk level indicators (motive, prior criminal history, household factors)
- Movement patterns and destination types

Case Narrative:
{narrative}

Summary:"""

# Weak Labeling Prompts
# Template for classifying movement patterns in case narratives
# Used by Qwen2.5-3B-Instruct for movement classification
# Enhanced with psychology research findings on offender movement patterns
MOVEMENT_CLASSIFICATION_PROMPT = """Classify the movement pattern described in this case narrative. Choose the most appropriate category:

Categories:
- Stationary: Person/vehicle remained in one location
- Local: Short-distance movement within neighborhood/city
- Regional: Medium-distance movement within state/region
- Interstate: Long-distance movement across state lines
- International: Movement across country borders
- Unknown: Insufficient information to determine

Psychology Research Context:
- Sexual offenders: Typically local (<5 miles, often within 1 mile of victim's home)
- Residential abductions: Very local, often within quarter mile, to nearby outdoor/indoor secondary locations
- Custodial abductions: Regional to interstate, especially when crossing state lines or moving to another city/state
- Family abductions: Often regional/interstate, moving to extended family locations or home country
- Stranger abductions: Can be local initially, but stereotypical kidnappings involve 50+ miles or overnight detention
- Transportation: On foot (local), vehicle (local to interstate), online (virtual proximity for online exploitation)
- Destination types: Offender's residence, vacant buildings, wooded areas, parks (local); extended family locations, another state/country (regional/interstate)

Case Narrative:
{narrative}

Movement Classification:"""

# Template for assessing risk levels in case narratives
# Used by Qwen2.5-3B-Instruct for risk assessment
# Enhanced with psychology research findings from 13 research sources
RISK_ASSESSMENT_PROMPT = """Assess the risk level of this case narrative. Consider factors like:

Traditional Factors:
- Violence indicators
- Weapon involvement
- Suspect behavior patterns
- Victim vulnerability
- Evidence quality

Psychology Research Indicators:
- Child age: Under 5 (false allegations risk), 6-11 (sexual risk), 12-17 (acquaintance/stranger risk)
- Offender relationship: Known individuals (family, acquaintance) pose higher risk than strangers
- Grooming indicators: Prior contact, repeated visits, boundary violations, 2+ weeks of contact
- Behavioral red flags: Evasion, equivocation, prior familiarity with residence, unsecured entry points
- Temporal risk: Overnight hours (12am-5am) for residential, afternoon (2pm-9pm) for public abductions
- Motive patterns: Sexual gratification (most common), custodial disputes, trafficking, maternal desire
- Prior criminal history: Burglary, assault, prior sex offenses, kidnapping arrests increase risk
- Household factors: Unstable, chaotic, unsecured doors/windows, low guardianship periods
- False allegation indicators: Equivocation, delayed reporting, inappropriate information order, minimal sensory detail

Risk Levels:
- Low: Minimal threat indicators
- Medium: Some concerning factors present
- High: Multiple threat indicators (grooming, prior familiarity, known relationship)
- Critical: Immediate danger indicators (overnight abduction, weapon, prior criminal history, sexual motive)

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
