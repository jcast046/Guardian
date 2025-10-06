# Guardian
Guardian is a research proof-of-concept that uses synthetic, schema-conformant case reports to train and evaluate a search-prediction pipeline for missing-child cases in Virginia. It performs LLM-assisted field extraction, enriches records with geospatial context, and outputs ranked, time-windowed search zones designed to directly assist responders in locating victims.


# -> [Guardian_Parser](https://github.com/jcast046/Guardian_parser) <-


The **Guardian Parser** is a pipeline for converting unstructured missing person case PDFs (from **NamUs**, **NCMEC**, **FBI**, and **The Charley Project**) into a unified, structured format based on the **Guardian JSON Schema**. It extracts demographic, spatial, temporal, outcome, and narrative/OSINT fields, normalizes them, and outputs both JSONL and CSV files for downstream analysis and synthetic data generation for this project.

## Data Schemas

This project uses JSON Schema validation to ensure data consistency and quality. All JSON files in the `data/` directory are automatically validated against their appropriate schemas.

### Available Schemas

- **`guardian_schema.json`** - Core schema for Guardian case files containing demographic, spatial, temporal, outcome, and narrative/OSINT data
- **`road_segment.schema.json`** - Schema for Virginia road segment data with geometry, route designations, and operational attributes
- **`transit_line.schema.json`** - Schema for transit lines (rail, metro, bus routes) with geometry and service patterns
- **`transit_stop.schema.json`** - Schema for transit stops and stations with accessibility features and facilities
- **`gazetteer.schema.json`** - Schema for Virginia geographic gazetteer data with location coordinates and regional classifications
- **`case_templates.schema.json`** - Schema for synthetic case generation templates

### Validation

The project includes automatic JSON schema validation:

```bash
# Install validation dependencies
pip install -r requirements-validation.txt

# Run validation (build step)
python build.py
```

The validation system automatically detects the appropriate schema for each JSON file based on file path and content patterns, ensuring all data conforms to the expected structure.

## Features

###  Build System & Validation
- **Automatic JSON Schema Validation**: All JSON files are validated against appropriate schemas
- **Smart Schema Detection**: Automatically detects the correct schema for each file based on path and content
- **Build Integration**: Simple `python build.py` command validates all data files
- **Pre-commit Ready**: Validation can be integrated into development workflows

###  Synthetic Data Generation
- **Large-scale Generation**: Default generation of 500 cases with optimized performance
- **Progress Tracking**: Intelligent progress reporting with completion percentage and ETA
- **Memory Management**: Automatic garbage collection for large-scale generation
- **Performance Monitoring**: Time tracking and per-case generation statistics
- **Realistic Case Generation**: Python script generates schema-valid synthetic missing-child cases for Virginia
- **Geographic Integration**: Uses real Virginia geographic data (gazetteer, regions, roads, transit)
- **Reinforcement Learning Integration**: Incorporates RL search patterns and time windows
- **Transit-Aware**: Generates sightings at real transit stations
- **Realistic Vehicle Data**: Uses actual make/model combinations from vehicle inventory

###  Data Sources
- **Virginia Gazetteer**: 133+ locations with coordinates and regional classifications
- **Road Network**: 247+ road segments with geometry and route designations
- **Transit System**: 2,359+ transit stations across Virginia
- **Regional Boundaries**: GeoJSON boundaries for Virginia regions
- **Lexicons**: Behavioral patterns, clothing, vehicles, witnesses, and time gaps

### Virginia Transportation Data
The system also provides data from the **Guardian_Parser** that is extensive **Virginia transportation data** including statewide transit networks, road segments, and regional transportation infrastructure.

## Usage

### Installation

```bash
# Clone the repository
git clone git@github.com:jcast046/Guardian.git
cd Guardian

# Install dependencies
pip install -r requirements.txt

# Download models (requires Hugging Face token for Llama)
$env:HUGGINGFACE_HUB_TOKEN = "hf_XXXX"   # needed for Llama
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1

# Validate data sources
python build.py
```

### Model Management

Guardian uses two specialized LLM models optimized for RTX 4060 (8GB VRAM):

- **Qwen2.5-3B-Instruct**: Entity extraction and movement classification with 8-bit quantization (GPU)
- **Llama-3.2-3B-Instruct**: Case summarization with smart offloading (GPU+CPU)

#### Performance Optimizations

- **Memory Management**: Advanced quantization and offloading for 8GB VRAM
- **Inference Speed**: TF32 acceleration and SDPA attention for RTX 40xx GPUs
- **Output Quality**: Clean token decoding with early stopping and regex backfill
- **Dependency Handling**: Optional fine-tuning dependencies for core functionality

#### Model Configuration

Models are configured in `guardian.config.json` and versions are pinned in `models.lock.json`:

```json
{
  "models": {
    "extractor": ".\\models\\Qwen2.5-3B-Instruct",
    "summarizer_instruct": ".\\models\\Llama3_2-3B-Instruct",
    "weak_labeler": ".\\models\\Qwen2.5-3B-Instruct"
  },
  "use_summarizer": "instruct",
  "use_llama_as_extractor": false
}
```

#### Downloading Models

```powershell
# Set Hugging Face token (required for Llama models)
$env:HUGGINGFACE_HUB_TOKEN = "hf_XXXX"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Download all models
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1
```

#### Pinning Model Versions

For reproducible builds, pin models to exact commit hashes:

```bash
# Pin all models to current commit SHAs
python .\scripts\freeze_lock.py
```

This updates `models.lock.json` with exact commit hashes instead of "main" branches.

### Basic Usage

```bash
# Generate 500 synthetic cases (default)
python generate_cases.py

# Generate with specific seed for reproducibility
python generate_cases.py --n 100 --seed 42

# Output to custom directory
python generate_cases_organized.py --n 1000 --out data/large_batch_cases

# Generate smaller batches for testing
python generate_cases.py --n 10
```

### LLM Usage

```python
from llm import summarize, extract_json, label_case

# Case summarization (5 bullet points with early stopping)
summary = summarize("15-year-old Alex last seen near I-395 in Arlington at 6:30 PM; blue Corolla heading south toward Alexandria.")
print(summary)

# Entity extraction (structured JSON with regex backfill)
entities = extract_json("John Doe, 12, last seen near I-95 in Henrico County, VA at 10:00 AM.")
print(entities)

# Movement classification (4-bit quantization for efficiency)
classification = label_case("Blue Corolla left via I-395 toward Alexandria; last ping Potomac Yard.")
print(classification)
```

### Large-Scale Generation

```bash
# Generate 500 cases (default) with progress tracking
python generate_cases.py

# Generate 1000+ cases with custom output directory
python generate_cases.py --n 1000 --out data/production_cases

# Monitor progress and performance metrics
# Progress is reported every 5% (25 cases for 500 total)
# Shows completion percentage, ETA, and performance statistics
```

### Performance Features

- **Progress Tracking**: Reports completion every 5% with ETA calculations
- **Memory Management**: Automatic garbage collection every 50 cases
- **Performance Metrics**: Shows total time and per-case generation speed
- **Optimized Algorithms**: Cached transit networks and efficient geographic processing
- **LLM Optimizations**: Advanced memory management and inference optimizations
  - **4-bit Quantization**: Qwen2.5-3B for efficient classification
  - **8-bit Quantization**: Qwen2.5-3B for full GPU extraction
  - **Smart Offloading**: Llama-3.2-3B with GPU+CPU memory management
  - **TF32 Acceleration**: RTX 40xx GPU optimization
  - **SDPA Attention**: Fast attention implementation
  - **Clean Output**: Token-level decoding without prompt echo

### Validate Data
```bash
# Validate all JSON files against schemas
python build.py

# Install validation dependencies
pip install jsonschema
```

## Project Structure

```
Guardian/
├── data/                       # Data storage and templates
│   ├── geo/                    # Geographic data
│   │   ├── va_gazetteer.json   # Virginia locations
│   │   └── va_rl_regions.geojson # Regional boundaries
│   ├── lexicons/               # Behavioral and descriptive data
│   │   ├── behaviors.json
│   │   ├── clothing.json
│   │   ├── routes.json
│   │   ├── time_gaps.json
│   │   ├── vehicles.json
│   │   └── witness.json
│   ├── synthetic_cases/        # Generated synthetic cases
│   │   └── GRD-*.json          # Generated case files
│   ├── templates/              # Case generation templates
│   │   ├── amber_alert.txt
│   │   ├── case_templates.json
│   │   ├── facebook_post.txt
│   │   ├── guardian_case.blank.json
│   │   ├── news_report.txt
│   │   ├── nextdoor_post.txt
│   │   ├── police_presser.txt
│   │   └── youtube_transcripts.txt
│   └── transportation/         # Transportation data
│       ├── va_road_segments.json
│       ├── va_transit.json
│       └── va_transportation_summary.json
├── eda_out/                    # Exploratory Data Analysis outputs
│   ├── age_hist.png            # Age distribution histogram
│   ├── county_topN_bar.png     # County analysis charts
│   ├── distribution_summary.png # Distribution summaries
│   ├── eda_cases_min.jsonl      # Minimal case data for EDA
│   ├── eda_counts.json          # Statistical counts
│   ├── gender_bar.png           # Gender distribution
│   ├── kde_age_*.png           # Kernel density plots by age
│   ├── validation_report.json   # Data validation results
│   ├── zones_review.jsonl       # Zone review data
│   └── zones_reweighted.jsonl   # Reweighted zone data
├── guardian_llm/               # LLM modules and AI components
│   ├── __init__.py
│   ├── extractor.py            # Entity extraction with Qwen2.5-3B
│   ├── summarizer.py           # Case summarization with Llama-3.2-3B
│   ├── weak_labeler.py         # Movement classification with Qwen2.5-3B
│   ├── finetune_qlora.py       # QLoRA fine-tuning system
│   ├── prompts.py              # Standardized prompt templates
│   └── guardian.config.json    # LLM-specific configuration
├── models/                     # Local model storage (gitignored)
│   ├── Qwen2.5-3B-Instruct/   # Entity extraction and classification model
│   └── Llama3_2-3B-Instruct/  # Summarization model
├── reinforcement_learning/     # RL configuration and data
│   ├── ground_truth.json       # Ground truth data for RL
│   └── search_reward_config.json # RL reward configuration
├── schemas/                    # JSON Schema definitions
│   ├── case_templates.schema.json
│   ├── gazetteer.schema.json
│   ├── guardian_case.schema.json
│   ├── guardian_schema.json
│   ├── road_segment.schema.json
│   ├── transit_line.schema.json
│   └── transit_stop.schema.json
├── scripts/                    # Utility scripts
│   ├── download_models.ps1     # PowerShell model downloader
│   ├── freeze_lock.py          # Pin model revisions to commits
│   └── models.lock.json        # Pinned model versions
├── src/                        # Core Guardian modules
│   ├── geography/              # Geographic processing modules
│   │   ├── __init__.py
│   │   ├── distance.py         # Distance calculations
│   │   ├── regions.py          # Regional analysis
│   │   └── validation.py       # Geographic validation
│   ├── transportation/         # Transportation analysis modules
│   │   ├── __init__.py
│   │   └── networks.py         # Network analysis
│   └── guardian_modules.py     # Main Guardian functionality
├── .gitignore                  # Git ignore patterns for models
├── build.py                    # Main validation script
├
├── eda_hotspot.py             # Exploratory data analysis
├── generate_cases.py           # Synthetic case generator
├── generate_cases_organized.py # Organized case generation
├── guardian.config.json        # Main model configuration
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── run_all_llms.py            # LLM testing and evaluation
└── zone_qa.py                  # Zone quality assurance
```

## Key Scripts

### `run_all_llms.py` - LLM Analysis and EDA Data Preparation
Comprehensive LLM analysis script that processes synthetic cases through all Guardian LLM models:

- **Purpose**: Runs summarizer, extractor, and weak labeler models on synthetic cases
- **Outputs**: 
  - `eda_out/eda_cases_min.jsonl` - Normalized case data for EDA
  - `eda_out/eda_counts.json` - Statistical counts and demographics
  - `eda_out/validation_report.json` - Data quality validation results
- **Features**: 
  - Minimal mode (deterministic baseline)
  - Comprehensive case processing with progress tracking
  - Data normalization and validation
  - Memory-optimized processing for large datasets

### `eda_hotspot.py` - Exploratory Data Analysis and KDE Hotspotting
Advanced visualization and hotspot analysis for case data:

- **Purpose**: Creates demographic charts and KDE (Kernel Density Estimation) hotspot maps
- **Outputs**:
  - `distribution_summary.png` - Combined age, gender, and county charts
  - `age_hist.png`, `gender_bar.png`, `county_topN_bar.png` - Individual demographic charts
  - `kde_all.png`, `kde_age_le12.png`, `kde_age_13_17.png` - KDE hotspot maps
  - `eda_hotspot_report.md` - Markdown report with embedded visualizations
- **Features**:
  - Age-band specific hotspot analysis (≤12, 13-17)
  - Fixed geographic extent and shared color scales for comparison
  - Web Mercator projection for accurate distance calculations
  - Optional basemap integration with contextily

### `zone_qa.py` - LLM-Enhanced Zone Plausibility Analysis
LLM-powered search zone evaluation and prioritization:

- **Purpose**: Enhances search zone prioritization using LLM-based plausibility scoring
- **Workflow**:
  1. Reads case JSON files with search zones and narratives
  2. Uses weak labeler LLM to score plausibility (0-1) for each zone
  3. Reweights priorities based on LLM plausibility + RL configuration
  4. Generates evaluation metrics (Geo-hit@K) comparing baseline vs LLM-enhanced
- **Outputs**:
  - `zones_review.jsonl` - Per-case zone plausibility scores and rationale
  - `zones_reweighted.jsonl` - LLM-enhanced zones with priority_llm field
  - `zone_qa_metrics.json` - Evaluation metrics (Geo-hit@K baseline vs LLM)
  - `zone_evaluation_results.json` - Detailed Geo-hit@K evaluation results
- **Features**:
  - Sidecar architecture (doesn't modify core synthetic cases)
  - LLM-enhanced zone analysis for evaluation
  - Comprehensive metrics for search effectiveness

### `generate_cases.py` - Synthetic Case Generator
Comprehensive synthetic missing-child case generation system:

- **Purpose**: Generates realistic, schema-valid synthetic missing-child cases for Virginia
- **Features**:
  - **Geographic Accuracy**: Uses Virginia gazetteer, regional boundaries, and real coordinates
  - **Transportation Integration**: Dijkstra's algorithm for network-based road finding
  - **RL Integration**: Incorporates reinforcement learning search patterns and time windows
  - **Large-scale Generation**: Optimized for 500+ cases with progress tracking and memory management
  - **Schema Validation**: All generated cases conform to Guardian JSON schema
- **Outputs**:
  - `data/synthetic_cases/GRD-*.json` - Individual case files with unique identifiers
  - Progress tracking with completion percentage and ETA calculations
  - Performance metrics (total time, per-case generation speed)
- **Algorithm Details**:
  - **Haversine Distance**: Great-circle distance calculations for geographic accuracy
  - **Bounded Dijkstra**: Network-based road finding with early termination
  - **Geographic Filtering**: Multi-layer validation for spatial accuracy
  - **Transit Network Analysis**: Graph-based station connectivity analysis

## Reinforcement Learning Configuration

### `search_reward_config.json` - RL Search Reward Configuration
Comprehensive reinforcement learning configuration for search zone optimization:

- **Purpose**: Defines reward structures, time windows, and action spaces for RL-based search zone generation
- **Time Windows**: 
  - 0-24 hours (weight: 1.0) - Critical early search period
  - 24-48 hours (weight: 0.7) - Extended search phase
  - 48-72 hours (weight: 0.5) - Long-term search operations
- **Action Space**:
  - 3 zones per time window
  - Zone schema with geographic coordinates, radius, corridors, and priority scoring
  - State boundary enforcement (Virginia only)
- **Reward Structures**:
  - **Distance-based**: Geographic accuracy rewards using Haversine distance
  - **Time-based**: Earlier coverage rewards within time windows
  - **Hybrid**: Balanced scoring with alpha/beta weighting
  - **Regularizers**: Radius penalties and out-of-state penalties
- **Profiles**: Baseline, high-LLM, and risk-heavy configurations for different search strategies

### `ground_truth.json` - Ground Truth Zone Labels
Reference data for RL training and evaluation:

- **Purpose**: Maps case IDs to ground truth zone labels (z01, z02, z03) for RL training
- **Format**: JSON mapping of GRD case IDs to zone identifiers
- **Usage**: Used by `zone_qa.py` for Geo-hit@K evaluation metrics
- **Coverage**: 20+ cases with validated ground truth zone assignments

## Guardian LLM Modules

The `guardian_llm/` directory contains the core LLM functionality for the Guardian system, providing entity extraction, summarization, weak labeling, and fine-tuning capabilities.

### `__init__.py` - LLM Module Interface
Main interface for Guardian LLM functionality:

- **Purpose**: Provides clean API for all LLM operations with optional fine-tuning support
- **Runtime Functions**: `summarize`, `extract_entities`, `classify_movement`, `assess_risk`
- **Fine-tuning Functions**: Optional PEFT fine-tuning capabilities (requires peft library)
- **Safe Imports**: Graceful handling of missing optional dependencies

### `extractor.py` - Entity Extraction Module
Structured entity extraction using Qwen2.5-3B-Instruct:

- **Purpose**: Extracts persons, vehicles, locations, timeline, and evidence from case narratives
- **Key Functions**:
  - `extract_entities()` - Complete entity extraction with JSON output
  - `extract_persons()`, `extract_vehicles()`, `extract_locations()` - Specific entity types
  - `batch_extract_json()` - Batch processing for multiple cases
  - `minimal_entities_from_case()` - Lightweight extraction for EDA
- **Features**: Deterministic scaffolding with LLM backfill, regex validation, memory optimization
- **Output**: Structured JSON with validated entity data

### `summarizer.py` - Case Summarization Module
Investigator-focused case summarization using Llama-3.2-3B-Instruct:

- **Purpose**: Generates concise, factual bullet-point summaries for investigators
- **Key Functions**:
  - `summarize()` - Single case summarization
  - `batch_summarize()` - Batch processing with memory management
  - `release()` - GPU memory cleanup
- **Features**: TF32 acceleration, SDPA attention, early stopping, clean token decoding
- **Output**: 5-bullet point investigator summaries

### `weak_labeler.py` - Movement Classification and Risk Assessment
Automated labeling using Qwen2.5-3B-Instruct:

- **Purpose**: Classifies movement patterns and assesses risk levels for case prioritization
- **Key Functions**:
  - `classify_movement()` - Movement pattern classification
  - `assess_risk()` - Risk level assessment with rule-based overlay
  - `label_case()` - Combined movement and risk labeling
  - `batch_label_cases()` - Batch processing with optimization
- **Features**: Rule-based risk calibration, LLM+rule hybrid scoring, memory management
- **Output**: Movement classifications and risk assessments for case prioritization

### `prompts.py` - Standardized Prompt Templates
Centralized prompt management for all LLM tasks:

- **Purpose**: Provides consistent, optimized prompts for all Guardian LLM operations
- **Templates**:
  - `EXTRACTION_PROMPT` - Entity extraction with JSON structure
  - `SUMMARY_PROMPT` - Case summarization for investigators
  - `MOVEMENT_CLASSIFICATION_PROMPT` - Movement pattern analysis
  - `RISK_ASSESSMENT_PROMPT` - Risk level evaluation
  - Fine-tuning prompts for model training
- **Features**: Model-specific optimization, structured output formatting

### `finetune_qlora.py` - QLoRA Fine-tuning System
Parameter-Efficient Fine-Tuning using QLoRA:

- **Purpose**: Fine-tune Guardian models on synthetic case data using QLoRA
- **Key Functions**:
  - `fine_tune_summarizer()` - Fine-tune summarization model
  - `fine_tune_extractor()` - Fine-tune entity extraction model
  - `fine_tune_weak_labeler()` - Fine-tune classification model
- **Features**: 4-bit quantization, low-rank adaptation, memory-efficient training
- **Output**: Fine-tuned models saved to `./finetuned_*/` directories

### `guardian.config.json` - LLM Configuration
Model configuration and path management:

- **Purpose**: Centralized configuration for all Guardian LLM models
- **Configuration**:
  - Model paths for extractor, weak_labeler, and summarizer
  - Model selection flags (use_llama_as_extractor)
  - Runtime behavior settings
- **Integration**: Used by all LLM modules for consistent model loading

### Core Algorithms

- **Haversine Distance**: Calculates great-circle distances between geographic coordinates
- **Dijkstra's Algorithm**: Finds shortest paths in transit networks
- **Bounded Search**: Optimized graph traversal with distance limits
- **Geographic Filtering**: Multi-layer validation for spatial accuracy

### Optimization Strategies

- **Large-scale Generation**: Optimized for 500+ case generation with progress tracking
- **Memory Management**: Automatic garbage collection every 50 cases
- **Performance Monitoring**: Real-time progress reporting with ETA calculations
- **Caching**: Transit networks and station data are cached
- **Lazy Loading**: Data is loaded only when needed
- **Spatial Indexing**: Efficient geographic proximity searches
- **Early Termination**: Bounded algorithms for distance queries

## Development

### Adding New Data

1. **Place JSON files** in appropriate `data/` subdirectories
2. **Create corresponding schema files** in `schemas/`
3. **Update `build.py`** if needed for new schema detection
4. **Run validation**: `python build.py`

### Schema Validation

The validation system uses intelligent schema detection:

- **Path-based**: Files in `synthetic_cases/` use `guardian_schema.json`
- **Content-based**: Analyzes JSON structure to determine appropriate schema
- **Fallback**: Skips files without detectable schema patterns

### Synthetic Case Generation
The generator creates realistic cases by:
- **Large-scale processing**: Efficiently generates 500+ cases with progress tracking
- **Memory optimization**: Automatic garbage collection prevents memory buildup
- **Performance monitoring**: Real-time progress reporting with ETA calculations
- Selecting locations from Virginia gazetteer
- Using real road segments and transit stations
- Incorporating RL search patterns and time windows
- Generating consistent vehicle and witness information
- Creating geographically sensible movement patterns