# Changelog

All notable changes to the Guardian project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Model Management System**: Complete model lifecycle management for Guardian AI
  - `models.lock.json`: Pinned model versions with exact commit hashes for reproducibility
  - `scripts/download_models.ps1`: PowerShell script for automated model downloading
  - `scripts/freeze_lock.py`: Python script to pin model revisions to exact commits
  - `.gitignore`: Comprehensive patterns for model files and cache directories
- **LLM Module System**: Complete LLM module architecture for Guardian AI
  - `llm/summarizer.py`: Auto-switching summarizer with Llama-3 8B models
  - `llm/extractor.py`: Entity extraction using Qwen2-7B-Instruct for JSON output
  - `llm/weak_labeler.py`: Movement/risk classification using Mistral-7B-Instruct
  - `llm/finetune_qlora.py`: QLoRA fine-tuning system for all model types
  - `llm/prompts.py`: Standardized prompt templates for all LLM tasks
- **Model Configuration**: `guardian.config.json` for centralized model management
- **Auto-Switch Logic**: Intelligent fallback between instruct and base models
- **QLoRA Fine-tuning**: Parameter-efficient fine-tuning with LoRA adapters
- **Comprehensive Documentation**: Industry-standard docstrings for all LLM modules
- **Entity Extraction**: Structured JSON extraction for persons, vehicles, locations
- **Weak Labeling**: Automated movement classification and risk assessment
- **Batch Processing**: Efficient batch processing for multiple cases
- **Large-scale case generation**: Default case count increased from 10 to 500
- **Progress tracking**: Intelligent progress reporting with percentage completion and ETA
- **Performance monitoring**: Time tracking and per-case generation statistics
- **Memory management**: Automatic garbage collection every 50 cases for large batches
- **Enhanced user experience**: Detailed progress indicators and performance metrics
- **GPU Memory Optimization**: Advanced memory management for RTX 4060 (8GB VRAM)
  - 4-bit quantization for Mistral-7B-Instruct (weak labeler)
  - 8-bit quantization for Qwen2-7B-Instruct (extractor)
  - Smart offloading for Llama-3.1-8B-Instruct (summarizer)
  - Explicit model unloading between switches
- **Performance Optimizations**: Advanced inference optimizations
  - TF32 acceleration for RTX 40xx GPUs
  - SDPA (Scaled Dot-Product Attention) implementation
  - New token decoding (removes prompt echo from output)
  - Early stopping for summarizer (5th bullet point)
  - Input trimming (last 400-600 words for efficiency)
- **Robust JSON Processing**: Enhanced extractor with regex backfill
  - Intelligent field completion using regex patterns
  - Robust JSON parsing with bracket matching
  - Few-shot examples for improved accuracy
  - Automatic field validation and completion
- **Optional Dependencies**: Graceful handling of missing packages
  - Fine-tuning utilities made optional in `llm/__init__.py`
  - Core functionality available without PEFT installation
  - Robust error handling for missing dependencies
- Industry-standard documentation with comprehensive API reference
- Performance benchmarks and optimization strategies
- Code quality tools integration (Black, flake8, mypy, bandit)
- Comprehensive type hints throughout codebase
- Detailed algorithm documentation with examples
- Mermaid diagrams for architecture visualization

### Changed
- **LLM Architecture**: Implemented modular LLM system with specialized models
- **Model Management**: Centralized configuration through `guardian.config.json`
- **Documentation Standards**: Applied industry-standard docstrings across all LLM modules
- **Virtual Environment**: Organized virtual environment structure at project root
- **Default case generation**: Now generates 500 cases by default instead of 10
- **Progress reporting**: Shows completion percentage, ETA, and performance metrics
- **Memory optimization**: Added garbage collection for large-scale generation
- **Repository Structure**: Added comprehensive `.gitignore` for model files and cache directories
- **Model Workflow**: Implemented reproducible model management with pinned versions
- **Summarizer Output**: Clean bullet point summaries with early stopping after 5th bullet
- **Extractor Processing**: Robust JSON extraction with regex backfill for missing fields
- **Memory Management**: Optimized for RTX 4060 with 4-bit/8-bit quantization and smart offloading
- **Performance**: Enhanced inference speed with TF32, SDPA, and optimized generation parameters
- **Output Quality**: Removed prompt echo from all model outputs, clean token decoding
- **Dependency Management**: Made fine-tuning dependencies optional for core functionality
- Enhanced README.md with badges, quick start guide, and comprehensive sections
- Improved docstring formatting with industry standards
- Updated project structure documentation
- Enhanced error handling and validation

### Fixed
- **Memory management**: Prevented memory buildup during large-scale generation
- **Performance optimization**: Added timing and memory management for 500+ case batches
- **CUDA Out of Memory**: Fixed OOM errors with 4-bit quantization for Mistral and 8-bit for Qwen
- **Model Loading**: Fixed device mapping issues with proper GPU/CPU offloading
- **Output Quality**: Fixed prompt echo in model outputs with proper token slicing
- **JSON Parsing**: Fixed null value extraction with robust JSON parsing and regex backfill
- **Dependency Issues**: Fixed import errors by making fine-tuning dependencies optional
- **UTF-8 Encoding**: Fixed bullet point display issues in PowerShell with proper encoding
- **Early Stopping**: Fixed summarizer truncation with intelligent 5th bullet detection
- Geographic accuracy issues with I-81 road filtering
- City/county classification in generated cases
- Original fields population in case metadata
- Transit distance calculations for distant locations

## [2.0.0] - 2025-01-XX

### Added
- Graph-based road finding using Dijkstra's algorithm
- Comprehensive geographic validation and filtering
- Transit network analysis with distance-based filtering
- Bounded Dijkstra's algorithm for performance optimization
- Geographic filtering for misclassified roads and transit stops
- Enhanced schema validation and error handling
- Caching strategies for improved performance
- Original fields metadata in generated cases

### Changed
- Refactored road finding from simple filtering to graph-based approach
- Enhanced geographic accuracy with multi-layer validation
- Improved performance with caching and early termination
- Updated case generation logic for better realism
- Enhanced error handling and graceful degradation

### Fixed
- I-81 incorrectly appearing in Northern Virginia cases
- Orange, VA showing Richmond transit stops (75+ miles away)
- Prince William county being classified as city
- Geographic inaccuracies in road and transit data
- Data quality issues with misclassified road segments

### Removed
- Dead code and unused functions
- Redundant algorithm implementations
- Outdated filtering logic

## [1.0.0] - 2024-XX-XX

### Added
- Initial release with basic case generation
- Virginia geographic data integration
- Schema validation implementation
- Basic road and transit finding
- Simple geographic filtering
- Case template system
- Reinforcement learning integration

### Features
- Synthetic missing-child case generation
- Virginia gazetteer integration
- Road segments and transit data
- Behavioral and clothing lexicons
- Search zone generation
- Follow-up sighting simulation
- Schema-compliant output

## [0.1.0] - 2024-XX-XX

### Added
- Project initialization
- Basic data structure setup
- Initial data sources
- Basic case generation framework
- Schema definitions
- Data validation system

---

## Development Notes

### Version 2.0.0 Major Changes
- **Algorithm Enhancement**: Replaced simple filtering with graph-based algorithms
- **Geographic Accuracy**: Implemented comprehensive geographic validation
- **Performance**: Added caching and optimization strategies
- **Documentation**: Industry-standard documentation and API reference
- **Code Quality**: Type hints, linting, and testing integration

### Breaking Changes
- Function signatures updated with type hints
- Some internal data structures modified for performance
- Enhanced error handling may change exception types
- Geographic filtering logic completely rewritten

### Migration Guide
- Update function calls to include type hints
- Review geographic filtering logic for custom implementations
- Update error handling for new exception types
- Review performance implications of new algorithms

### Known Issues
- Large datasets may require memory optimization
- Complex geographic queries may be slow on first run
- Some edge cases in geographic validation need refinement

### Future Plans
- **LLM Model Integration**: Download and configure required models (Qwen2-7B-Instruct, Llama-3.1-8B, Mistral-7B-Instruct)
- **Fine-tuning Pipeline**: Implement automated fine-tuning on synthetic case data
- **Model Performance**: Benchmark and optimize LLM performance for Guardian tasks
- **API Integration**: Create REST API endpoints for LLM services
- **Model Versioning**: Implement model version management and A/B testing
- Machine learning integration for case generation
- Real-time data source updates
- Advanced geographic analysis
- Performance optimization for large-scale generation
- Enhanced visualization and analytics tools

## LLM Module Details

### Model Requirements
- **Qwen2-7B-Instruct**: For entity extraction and JSON output
- **Llama-3.1-8B-Instruct**: For case summarization (primary)
- **Llama-3.1-8B**: For case summarization (fallback)
- **Mistral-7B-Instruct-v0.2**: For movement classification and risk assessment

### Model Management Workflow

#### 1. Initial Setup
```powershell
# Set Hugging Face token (required for Llama models)
$env:HUGGINGFACE_HUB_TOKEN = "hf_XXXX"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Download all models
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1
```

#### 2. Pin Model Versions
```bash
# Pin all models to current commit SHAs for reproducibility
python .\scripts\freeze_lock.py
```

#### 3. Model Configuration
The `guardian.config.json` file manages all model paths and settings:
```json
{
  "models": {
    "extractor": ".\\models\\Qwen2-7B-Instruct",
    "summarizer_instruct": ".\\models\\Llama3_1-8B-Instruct", 
    "summarizer_base": ".\\models\\Llama3_1-8B",
    "weak_labeler": ".\\models\\Mistral-7B-Instruct-v0_2"
  },
  "use_summarizer": "instruct",
  "use_llama_as_extractor": false
}
```

#### 4. Model Lock File
The `models.lock.json` file contains pinned model versions:
```json
{
  "models": [
    {
      "repo_id": "Qwen/Qwen2-7B-Instruct",
      "revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
      "local_dir": "models/Qwen2-7B-Instruct",
      "role": "extractor"
    }
  ]
}
```

### Usage Examples
```python
from llm import summarize, extract_entities, classify_movement, assess_risk

# Case summarization
summary = summarize("Case narrative...")

# Entity extraction  
entities = extract_entities("Case narrative...")

# Movement classification
movement = classify_movement("Case narrative...")

# Risk assessment
risk = assess_risk("Case narrative...")
```
