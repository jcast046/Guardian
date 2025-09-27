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

Guardian uses three specialized LLM models optimized for RTX 4060 (8GB VRAM):

- **Qwen2-7B-Instruct**: Entity extraction with 8-bit quantization (GPU)
- **Llama-3.1-8B-Instruct**: Case summarization with smart offloading (GPU+CPU)
- **Mistral-7B-Instruct-v0.2**: Movement classification with 4-bit quantization (GPU)

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
    "extractor": ".\\models\\Qwen2-7B-Instruct",
    "summarizer_instruct": ".\\models\\Llama3_1-8B-Instruct",
    "weak_labeler": ".\\models\\Mistral-7B-Instruct-v0_2"
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

#### LLM Performance

- **Summarizer**: ~22s load + generation, clean bullet points
- **Extractor**: ~20s load + generation, structured JSON with backfill
- **Weak Labeler**: ~28s load + generation, 4-bit quantization
- **Memory**: Optimized for RTX 4060 (8GB VRAM)
- **Output**: Clean, professional results without prompt echo

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
  - **4-bit Quantization**: Mistral-7B for efficient classification
  - **8-bit Quantization**: Qwen2-7B for full GPU extraction
  - **Smart Offloading**: Llama-3.1-8B with GPU+CPU memory management
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
├── data/
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
├── llm/                        # LLM modules and AI components
│   ├── __init__.py
│   ├── extractor.py            # Entity extraction with Qwen2-7B
│   ├── summarizer.py           # Case summarization with Llama-3.1-8B
│   ├── weak_labeler.py         # Movement classification with Mistral-7B
│   ├── finetune_qlora.py       # QLoRA fine-tuning system
│   └── prompts.py              # Standardized prompt templates
├── models/                     # Local model storage (gitignored)
│   ├── Qwen2-7B-Instruct/      # Entity extraction model
│   ├── Llama3_1-8B-Instruct/  # Summarization model
│   └── Mistral-7B-Instruct-v0_2/ # Classification model
├── scripts/                    # Utility scripts
│   ├── download_models.ps1     # PowerShell model downloader
│   └── freeze_lock.py          # Pin model revisions to commits
├── src/                        # Core Guardian modules
│   ├── guardian_modules.py     # Main Guardian functionality
│   ├── geography/              # Geographic processing
│   └── transportation/         # Transportation analysis
├── schemas/                    # JSON Schema definitions
│   ├── case_templates.schema.json
│   ├── gazetteer.schema.json
│   ├── guardian_case.schema.json
│   ├── guardian_schema.json
│   ├── road_segment.schema.json
│   ├── transit_line.schema.json
│   └── transit_stop.schema.json
├── reinforcement_learning/      # RL configuration
│   └── search_reward_config.json
├── .gitignore                  # Git ignore patterns for models
├── build.py                    # Main validation script
├── generate_cases.py           # Synthetic case generator
├── generate_cases_organized.py # Organized case generation
├── guardian.config.json        # Model configuration
├── models.lock.json            # Pinned model versions
└── README.md
```

### Core Algorithms

- **Haversine Distance**: Calculates great-circle distances between geographic coordinates
- **Dijkstra's Algorithm**: Finds shortest paths in transit networks
- **Bounded Search**: Optimized graph traversal with distance limits
- **Geographic Filtering**: Multi-layer validation for spatial accuracy

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_geographic.py
python -m pytest tests/test_transportation.py
python -m pytest tests/test_case_generation.py
```

### Integration Tests

```bash
# Test case generation pipeline
python tests/test_integration.py

# Test data validation
python tests/test_validation.py
```

### Performance Tests

```bash
# Benchmark case generation
python tests/test_performance.py

# Memory usage analysis
python tests/test_memory.py
```

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