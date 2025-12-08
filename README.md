# Guardian

## Abstract

Guardian is a research proof-of-concept system for missing-child case analysis and search zone optimization. The system uses synthetic, schema-conformant case reports to train and evaluate a search-prediction pipeline for missing-child cases in Virginia. Guardian performs LLM-assisted field extraction, enriches records with geospatial context, and outputs ranked, time-windowed search zones designed to assist responders in locating victims.

The system integrates multiple subsystems:
- **Synthetic Case Generation**: Schema-valid case generation with geographic and transportation realism
- **LLM Pipeline**: Entity extraction, summarization, and weak labeling using optimized 3B-parameter models
- **Clustering & Hotspot Detection**: K-Means, DBSCAN, and KDE-based spatial analysis
- **Mobility Forecasting**: Markov chain propagation with survival analysis for probability distribution forecasting
- **Reinforcement Learning**: Search zone optimization with reward-based zone prioritization
- **Evaluation Metrics**: Comprehensive metrics for operational, performance, RL, clustering, and predictive consistency evaluation

**⚠️ Research Use Only**: This system is a research proof-of-concept and is not intended for production use in real missing-person cases.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Guardian Pipeline                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │  Data   │          │  Schema │          │  Model  │
   │ Sources │          │Validation│         │Download │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Case Generation  │
                    │  (Synthetic)      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   LLM Processing  │
                    │ (Extract/Summarize)│
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │Clustering│         |Forecast │          │Zone QA  │
   │Hotspots │          │(Markov) │           │(LLM)    │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Search Plan Gen  │
                    │  (Sectors/Rings)  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Metrics & Eval  │
                    │  (Geo-hit@K)      │
                    └────────────────── ┘
```

### Core User Workflow

The primary end-to-end pipeline for generating and evaluating synthetic cases:

1. **Data Validation** → 2. **Case Generation** → 3. **LLM Processing** → 4. **Clustering & Hotspots** → 5. **Forecasting & RL Zones** → 6. **Search Plan Generation** → 7. **Metrics & Evaluation**

### Research / Evaluation Only

- Clustering diagnostics and comparison tools
- Predictive consistency evaluation
- Cluster stability analysis
- Advanced visualization and EDA tools

### Developer / Experimental

- QLoRA fine-tuning system
- Model quantization experiments
- Advanced RL reward configuration
- Custom movement model profiles

## Data Pipeline

### Data Sources

Guardian uses structured data from multiple sources:

- **Guardian Parser**: [Guardian_Parser](https://github.com/jcast046/Guardian_parser) converts unstructured missing person case PDFs (NamUs, NCMEC, FBI, The Charley Project) into unified Guardian JSON Schema format
- **Geographic Data**: Virginia gazetteer, regional boundaries, state boundary shapefiles
- **Transportation Data**: Road segments, transit lines, transit stops
- **Lexicons**: Behavioral patterns, movement profiles, time patterns, offender motives, concealment sites
- **Templates**: Case generation templates (Amber Alert, Facebook posts, news reports, etc.)

### Data Directory Structure

```
data/
├── geo/                          # Geographic data
│   ├── va_gazetteer.json         # Virginia locations database
│   ├── va_rl_regions.geojson     # Regional boundaries
│   ├── va_boundary.geojson        # Virginia state boundary
│   └── cb_2023_us_state_500k/   # US state shapefiles
├── lexicons/                     # Behavioral and descriptive data (12 files)
│   ├── behaviors.json
│   ├── clothing.json
│   ├── concealment_sites.json
│   ├── lures_transport.json
│   ├── movement_profiles.json
│   ├── offender_motives.json
│   ├── psychology_findings.json
│   ├── routes.json
│   ├── time_gaps.json
│   ├── time_patterns.json
│   ├── vehicles.json
│   └── witness.json
├── templates/                    # Case generation templates (8 files)
│   ├── amber_alert.txt
│   ├── case_templates.json
│   ├── facebook_post.txt
│   ├── guardian_case.blank.json
│   ├── news_report.txt
│   ├── nextdoor_post.txt
│   ├── police_presser.txt
│   └── youtube_transcripts.txt
├── transportation/               # Transportation network data (3 files)
│   ├── va_road_segments.json
│   ├── va_transit.json
│   └── va_transportation_summary.json
├── synthetic_cases/               # Generated case files (GRD-*.json)
├── real_cases/                   # Real case data
│   ├── guardian_output.jsonl
│   └── NAMUS_VA 351 Missing Persons Cases.csv
├── training/                     # Training data
│   ├── psychology_weak_labeler_training.json
│   └── real_cases_weak_labeler_training.json
└── psych_research/               # Psychology research examples (13 files)
```

## Schema Validation System

Guardian uses JSON Schema validation to ensure data consistency and quality. All JSON files are validated against appropriate schemas with automatic schema detection.

### Available Schemas

- **`guardian_schema.json`** - Core schema for Guardian case files (demographic, spatial, temporal, outcome, narrative/OSINT)
- **`guardian_case.schema.json`** - Alternative case schema format
- **`road_segment.schema.json`** - Virginia road segment data (geometry, route designations, operational attributes)
- **`transit_line.schema.json`** - Transit lines (rail, metro, bus routes) with geometry and service patterns
- **`transit_stop.schema.json`** - Transit stops and stations with accessibility features
- **`gazetteer.schema.json`** - Virginia geographic gazetteer data (location coordinates, regional classifications)
- **`case_templates.schema.json`** - Synthetic case generation templates

### Validation Script

**Script**: `build.py`

**Purpose**: Validates all JSON files in the repository against appropriate schemas

**CLI Arguments**: None (runs directly)

**Usage**:
```bash
# Install validation dependencies
pip install jsonschema

# Run validation
python build.py
```

**Output**: Console output indicating validation success or errors

**Schema Detection**: The validation system uses intelligent schema detection:
- **Path-based**: Files in `synthetic_cases/` use `guardian_schema.json`
- **Content-based**: Analyzes JSON structure to determine appropriate schema
- **Fallback**: Skips files without detectable schema patterns

## Synthetic Case Generation

**Script**: `generate_cases.py`

**Purpose**: Generates realistic, schema-valid synthetic missing-child cases for Virginia

**CLI Arguments**:
- `--n` (int, default: 500): Number of cases to generate
- `--seed` (int, default: 42): Random seed for reproducibility
- `--out` (Path, default: `data/synthetic_cases`): Output directory

**Usage**:
```bash
# Generate 500 synthetic cases (default)
python generate_cases.py

# Generate smaller test batch
python generate_cases.py --n 50

# Custom output directory
python generate_cases.py --n 100 --out data/test_cases
```

**Outputs**:
- `data/synthetic_cases/GRD-*.json` - Individual case files with unique identifiers
- Progress tracking with completion percentage and ETA calculations
- Performance metrics (total time, per-case generation speed)

**Features**:
- Geographic accuracy using Virginia gazetteer, regional boundaries, and real coordinates
- Transportation integration with Dijkstra's algorithm for network-based road finding
- RL integration with reinforcement learning search patterns and time windows
- Large-scale generation optimized for 500+ cases with progress tracking and memory management
- Schema validation ensuring all generated cases conform to Guardian JSON schema

**Algorithm Details**:
- **Haversine Distance**: Great-circle distance calculations for geographic accuracy
- **Bounded Dijkstra**: Network-based road finding with early termination
- **Geographic Filtering**: Multi-layer validation for spatial accuracy
- **Transit Network Analysis**: Graph-based station connectivity analysis

**Alternative Implementation**: `generate_cases_organized.py` provides an alternative implementation with the same CLI arguments.

## LLM Pipeline

Guardian uses three specialized LLM models for entity extraction, summarization, and weak labeling. All models are optimized for RTX 4060 (8GB VRAM) with 4-bit quantization.

### Model Configuration

**Configuration File**: `guardian.config.json`

```json
{
  "models": {
    "summarizer_instruct": "models/Qwen2.5-3B-Instruct",
    "summarizer_adapter": "./finetuned_summarizer",
    "extractor_instruct": "models/Qwen2.5-3B-Instruct",
    "extractor_adapter": "./finetuned_extractor",
    "weak_labeler_instruct": "models/Qwen2.5-3B-Instruct",
    "weak_labeler_adapter": "./finetuned_weak_labeler"
  },
  "use_llama_as_extractor": false,
  "quantize_4bit": true,
  "device": "cuda",
  "dtype": "bfloat16",
  "batch_size": 16,
  "max_new_tokens": 256,
  "adapter_config": {
    "strict_mode": false,
    "merge_on_load": false
  }
}
```

**Current Active Models**:
- **Qwen2.5-3B-Instruct**: Used for extractor, weak labeler, and summarizer (all three tasks)
- **Llama-3.2-3B-Instruct**: ⚠️ Deprecated (listed in `models.lock.json` but not actively used)

### LLM Processing Script

**Script**: `run_all_llms.py`

**Purpose**: Orchestrates the LLM analysis pipeline, processing cases through geographic normalization, entity extraction, narrative generation, and quality assurance

**CLI Arguments**:
- `--reasoned` (flag): Run reasoned sidecars
- `--do-summary` (flag): Generate summaries
- `--fallback-extractor` (flag): Use fallback extractor

**Usage**:
```bash
# Process all cases through LLM models
python run_all_llms.py

# Generate summaries
python run_all_llms.py --do-summary

# Use fallback extractor
python run_all_llms.py --fallback-extractor
```

**Outputs**:
- `eda_out/eda_cases_min.jsonl` - Normalized case data for EDA
- `eda_out/eda_counts.json` - Statistical counts and demographics
- `eda_out/validation_report.json` - Data quality validation results
- `eda_out/llm_analysis_results.json` - LLM processing results and timings

**Features**:
- Minimal mode (deterministic baseline) for quick testing
- Comprehensive case processing with progress tracking
- Data normalization and validation
- Memory-optimized processing for large datasets

### LLM Modules

#### Entity Extraction (`guardian_llm/extractor.py`)

**Purpose**: Extracts persons, vehicles, locations, timeline events, and evidence from case narratives

**Key Functions**:
- `extract_entities()` - Complete entity extraction with JSON output
- `extract_persons()`, `extract_vehicles()`, `extract_locations()` - Specific entity types
- `batch_extract_json()` - Batch processing for multiple cases
- `minimal_entities_from_case()` - Lightweight extraction for EDA

**Features**:
- Deterministic scaffolding with LLM backfill
- Regex validation and pattern matching
- Memory optimization with 4-bit quantization
- GPU acceleration with CUDA support

#### Case Summarization (`guardian_llm/summarizer.py`)

**Purpose**: Generates concise, factual bullet-point summaries for investigators

**Key Functions**:
- `summarize()` - Single case summarization
- `batch_summarize()` - Batch processing with memory management
- `release()` - GPU memory cleanup

**Features**:
- TF32 acceleration for RTX 40xx GPUs
- SDPA attention for efficient inference
- Early stopping at summary markers
- Clean token decoding

**Output**: 5-bullet point investigator summaries

#### Weak Labeling (`guardian_llm/weak_labeler.py`)

**Purpose**: Classifies movement patterns and assesses risk levels for case prioritization

**Key Functions**:
- `classify_movement()` - Movement pattern classification
- `assess_risk()` - Risk level assessment with rule-based overlay
- `label_case()` - Combined movement and risk labeling
- `batch_label_cases()` - Batch processing with optimization

**Features**:
- Rule-based risk calibration
- LLM+rule hybrid scoring
- Memory management with 4-bit quantization
- GPU acceleration

**Output**: Movement classifications and risk assessments for case prioritization

#### Prompt Templates (`guardian_llm/prompts.py`)

**Purpose**: Centralized prompt management for all LLM tasks

**Templates**:
- `EXTRACTION_PROMPT` - Entity extraction with JSON structure
- `SUMMARY_PROMPT` - Case summarization for investigators
- `MOVEMENT_CLASSIFICATION_PROMPT` - Movement pattern analysis
- `RISK_ASSESSMENT_PROMPT` - Risk level evaluation

#### QLoRA Fine-tuning (`guardian_llm/finetune_qlora.py`)

**Purpose**: Parameter-Efficient Fine-Tuning using QLoRA

**Features**:
- 4-bit quantization + LoRA adapters
- Enables fine-tuning on 8GB VRAM
- Minimal parameter updates (~1% of model parameters)
- Memory-efficient training

**Output**: Fine-tuned models saved to `./finetuned_*/` directories

## Clustering & Hotspot Detection

Guardian provides a unified clustering interface for geographic coordinate analysis using K-Means, DBSCAN, and KDE algorithms.

### Clustering Module (`clustering/`)

**Public API** (`clustering/__init__.py`):
- `make_clusterer(name, **kwargs)` - Factory function to create clusterer instances
- `KMeansClustering`, `DBSCANClustering`, `KDEClustering` - Algorithm implementations
- Utility functions: `to_projected()`, `to_geographic()`, `validate_coordinates()`, etc.

**Modules**:
- `base.py` - Clusterer base class
- `kmeans.py` - K-Means clustering implementation
- `dbscan.py` - DBSCAN clustering implementation
- `kde.py` - KDE clustering implementation
- `utils.py` - Coordinate transformation, validation, I/O utilities

### K-Means Diagnostics

**Script**: `run_kmeans_diagnostics.py`

**Purpose**: Performs K-sweep for K-Means, computes inertia and silhouette scores, generates plots, selects optimal K, and exports hotspots

**CLI Arguments**:
- `--input` (str, default: `eda_out/eda_cases_min.jsonl`): Input JSONL file with case coordinates
- `--out` (str, default: `eda_out`): Output directory
- `--kmin` (int, default: 2): Minimum K value for sweep
- `--kmax` (int, default: 20): Maximum K value for sweep
- `--random-state` (int, default: 42): Random seed for reproducibility

**Usage**:
```bash
python run_kmeans_diagnostics.py --kmin 3 --kmax 15
```

**Outputs**:
- `eda_out/kmeans_hotspots.json` - K-Means hotspots with optimal K
- `eda_out/plots/kmeans_elbow.png` - Elbow plot for K selection
- `eda_out/plots/kmeans_silhouette.png` - Silhouette score plot

### DBSCAN Diagnostics

**Script**: `run_dbscan_diagnostics.py`

**Purpose**: Computes k-distance plot, selects optimal `eps`, fits DBSCAN, and exports hotspots

**CLI Arguments**:
- `--input` (str, default: `eda_out/eda_cases_min.jsonl`): Input JSONL file with case coordinates
- `--out` (str, default: `eda_out`): Output directory
- `--min-samples` (int, default: 5): Minimum samples for DBSCAN
- `--eps-meters` (float, default: None): Epsilon distance in meters (if None, uses k-distance plot)
- `--k` (int, default: 4): K value for k-distance plot

**Usage**:
```bash
python run_dbscan_diagnostics.py --eps-meters 1500.0 --min-samples 5
```

**Outputs**:
- `eda_out/dbscan_hotspots.json` - DBSCAN hotspots
- `eda_out/plots/dbscan_kdist.png` - K-distance plot for eps selection

### KDE Export

**Script**: `run_kde_export.py`

**Purpose**: Exports KDE hotspots using the unified clustering interface, converting them to a JSON format

**CLI Arguments**:
- `--input` (str, default: `eda_out/eda_cases_min.jsonl`): Input JSONL file with case coordinates
- `--out` (str, default: `eda_out`): Output directory
- `--bandwidth-meters` (float, default: 30000.0): KDE bandwidth in meters
- `--iso-mass` (float, default: 0.90): Iso-mass threshold for hotspot extraction

**Usage**:
```bash
python run_kde_export.py --bandwidth-meters 30000.0 --iso-mass 0.90
```

**Outputs**:
- `eda_out/kde_hotspots.json` - KDE hotspots in JSON format

### Cluster Comparison

**Script**: `run_cluster_comparison.py`

**Purpose**: Loads hotspots from K-Means, DBSCAN, and KDE, computes Jaccard overlaps, and generates comparison visualizations

**CLI Arguments**:
- `--in-dir` (str, default: `eda_out`): Input directory containing hotspot JSON files
- `--out` (str, default: `eda_out`): Output directory
- `--topN` (int, default: 10): Number of top hotspots to compare
- `--kde-polygons` (str, default: None): Path to KDE polygon file (optional)

**Usage**:
```bash
python run_cluster_comparison.py --topN 15
```

**Outputs**:
- `eda_out/cluster_compare.json` - Comparison metrics (Jaccard overlaps)
- `eda_out/plots/cluster_compare_side_by_side.png` - Side-by-side comparison visualization
- `eda_out/plots/cluster_compare_overlay.png` - Overlay comparison visualization

## Mobility Forecasting

The mobility forecasting system combines multi-source hotspots, Markov chain movement models, survival analysis, and sequential propagation to forecast probability distributions over geographic space for missing-person cases at future time horizons.

### Forecasting API (`reinforcement_learning/forecast_api.py`)

**Key Functions**:
- `forecast_distribution(case, t_hours, alpha_prior, steps_per_24h)` - Forecast at single time horizon
- `forecast_timeline(case, horizons, alpha_prior, profile)` - Forecast multiple horizons
- `forecast_search_plan(case, horizons, ...)` - Generate complete search plan with sectors and hotspots

**Inputs**:
- **Grid data**: `eda_out/grid_xy.npy` - Grid coordinates (longitude, latitude)
- **Road costs**: `eda_out/road_cost.npy` - Road accessibility costs
- **Seclusion scores**: `eda_out/seclusion.npy` - Seclusion/hiding location scores
- **Hotspots**: `eda_out/kmeans_hotspots.json`, `eda_out/dbscan_hotspots.json`, `eda_out/kde_hotspots.json`
- **Case data**: Case JSON files with `spatial.last_seen_lat/lon` and `temporal.last_seen_ts`

**Outputs**:
- **Probability distributions**: Arrays of shape (N,) summing to 1.0 over grid locations
- **Top locations**: Indices of highest-probability grid cells with coordinates
- **Visualizations**: PNG maps showing risk distribution at different time horizons

**Key Parameters**:
- **`alpha_prior`**: Mixing weight for KDE prior vs case-specific seed (0.0 = only seed, 1.0 = only prior)
- **`steps_per_24h`**: Number of Markov chain propagation steps per 24 hours (default: 3)
- **`method_weights`**: Dict mapping hotspot method names to weights (e.g., `{"kmeans": 1.0, "dbscan": 0.8, "kde": 1.2}`)
- **`profile`**: Survival profile type - `"default"` (half_life=24h), `"runaway"` (48h), `"abduction"` (12h)
- **`beta_cost_day/night`**: Road cost penalty coefficients for day/night transitions
- **`beta_secl_day/night`**: Seclusion reward coefficients for day/night transitions

### Forecast Visualization

**Script**: `reinforcement_learning/visualize_forecast.py`

**Purpose**: Visualize mobility forecasts over time

**CLI Arguments**:
- `--case` (str, default: None): Path to case JSON file (if not provided, uses first available case)
- `--horizons` (int list, default: [24, 48, 72]): Time horizons in hours
- `--output-dir` (str, default: `eda_out/forecast_plots`): Output directory for plots
- `--alpha-prior` (float, default: 0.5): Mixing weight for KDE prior
- `--steps-per-24h` (int, default: 3): Markov steps per 24 hours
- `--cumulative` (flag): Generate cumulative map (0-72h combined)
- `--cumulative-weights` (float list, default: None): Weights for cumulative map (default: [0.5, 0.3, 0.2] for 3 horizons)
- `--cumulative-mode` (str, default: "avg", choices: ["avg", "max"]): Cumulative combination mode

**Usage**:
```bash
python reinforcement_learning/visualize_forecast.py \
    --case data/synthetic_cases/GRD-2025-000001.json \
    --horizons 24 48 72 \
    --output-dir eda_out/forecast_plots
```

**Outputs**: PNG forecast plots in the specified output directory

### Movement Model (`reinforcement_learning/movement_model.py`)

**Key Functions**:
- `kde_prior()` - KDE-based prior probability distribution
- `survival_factor()` - Survival analysis temporal decay

**Features**:
- Multi-source hotspot integration (K-Means, DBSCAN, KDE)
- Markov chain propagation with road costs and seclusion factors
- Day/night pattern modeling
- Sequential propagation (forecasts build on previous horizons)

## Reinforcement Learning & Zone QA

Guardian uses reinforcement learning for search zone optimization with reward-based prioritization and LLM-enhanced plausibility scoring.

### RL Zone Generation

**Script**: `reinforcement_learning/build_rl_zones.py`

**Purpose**: Generates RL search zones from movement model predictions

**CLI Arguments**:
- `--mode` (str, default: "baseline", choices: ["baseline", "propagate"]): Zone generation mode
- `--config` (str, default: `reinforcement_learning/search_reward_config.json`): Path to reward configuration JSON file
- `--outdir` (str, default: `eda_out`): Output directory for results
- `--log-debug` (flag): Enable debug logging and diagnostics
- `--sample` (int, default: 0): Sample N cases for quick testing (0 = all cases)

**Usage**:
```bash
python reinforcement_learning/build_rl_zones.py --mode propagate --sample 10
```

**Outputs**:
- `eda_out/zones_rl.jsonl` - RL-generated search zones

### Search Plan CLI

**Script**: `reinforcement_learning/search_plan_cli.py`

**Purpose**: CLI wrapper for generating complete search plans with sectors, hotspots, containment rings, and visualization maps for SAR teams

**CLI Arguments**:
- `--case` (str, required): Path to case JSON file
- `--horizons` (int list, default: [24, 48, 72]): Time horizons in hours
- `--outdir` (str, default: `eda_out/forecast_plots`): Output directory
- `--use-cumulative` (flag): Use cumulative probability distribution
- `--no-cumulative` (flag): Disable cumulative distribution (opposite of --use-cumulative)
- `--hotspot-pct` (float, default: 0.80): Percentage of probability mass to include in hotspots
- `--sector-path` (str, default: None): Path to sector GeoJSON file (optional)
- `--alpha-prior` (float, default: 0.5): Mixing weight for KDE prior
- `--steps-per-24h` (int, default: 3): Markov steps per 24 hours
- `--beta-corr-day` (float, default: None): Day corridor coefficient
- `--beta-corr-night` (float, default: None): Night corridor coefficient
- `--profile` (str, default: "default"): Survival profile (default, runaway, abduction)
- `--max-hotspots-per-sector` (int, default: 5): Maximum hotspots per sector

**Usage**:
```bash
python reinforcement_learning/search_plan_cli.py \
    --case data/synthetic_cases/GRD-2025-000001.json \
    --horizons 24 48 72 \
    --outdir eda_out/search_plans
```

**Outputs**:
- `eda_out/forecast_plots/<case_id>_search_plan.png` - Main search plan visualization
- `eda_out/forecast_plots/<case_id>_search_plan_t<horizon>h.png` - Per-horizon visualizations
- `eda_out/forecast_plots/<case_id>_search_plan.json` - Search plan data (JSON)
- `eda_out/forecast_plots/<case_id>_search_plan_sectors.csv` - Sector rankings
- `eda_out/forecast_plots/<case_id>_search_plan_sectors_by_horizon.csv` - Sector rankings by horizon

### Zone QA (LLM-Enhanced Zone Analysis)

**Script**: `zone_qa.py`

**Purpose**: LLM-enhanced search zone plausibility analysis and prioritization

**CLI Arguments**:
- `--input` (str, default: `data/synthetic_cases`): Directory containing GRD-*.json case files
- `--config` (str, default: `reinforcement_learning/search_reward_config.json`): Path to search_reward_config.json
- `--outdir` (str, default: `eda_out`): Output directory
- `--profile` (str, default: None): Profile key in search_reward_config.json
- `--evaluate` (flag): Run Geo-hit@K evaluation analysis after zone QA
- `--ttf` (flag): Compute time-to-first-hit metrics
- `--cdf` (flag): Generate CDF of distance gaps
- `--sample` (int, default: 0): Sample N cases for quick testing
- `--selftest` (flag): Run self-test validation checks
- `--force-real` (flag): Force real weak-labeler; error if unavailable
- `--verbose` (flag): Extra logging per case/zone
- `--format` (flag): Display formatted zone results
- `--per-zone` (flag): Call labeler for each zone (slower but more precise)
- `--batch-size` (int, default: 16): LLM batch size for processing multiple cases
- `--print-models` (flag): Print model configuration and exit

**Usage**:
```bash
# Run zone QA analysis
python zone_qa.py --evaluate

# Quick test with sample
python zone_qa.py --sample 10 --verbose

# Self-test validation
python zone_qa.py --selftest
```

**Outputs**:
- `eda_out/zones_review.jsonl` - Per-case zone plausibility scores and rationale
- `eda_out/zones_reweighted.jsonl` - LLM-enhanced zones with priority_llm field
- `eda_out/zone_qa_metrics.json` - Evaluation metrics (Geo-hit@K baseline vs LLM)
- `eda_out/zone_evaluation_results.json` - Detailed Geo-hit@K evaluation results

**Features**:
- Sidecar architecture (doesn't modify core synthetic cases)
- LLM-enhanced zone analysis for evaluation
- Comprehensive metrics for search effectiveness
- Priority reweighting based on plausibility scores

### RL Configuration

**Configuration File**: `reinforcement_learning/search_reward_config.json`

**Time Windows**:
- 0-24 hours (weight: 1.0) - Critical early search period
- 24-48 hours (weight: 0.7) - Extended search phase
- 48-72 hours (weight: 0.5) - Long-term search operations

**Action Space**: 3 zones per time window, zone schema with geographic coordinates, radius, corridors, and priority scoring, state boundary enforcement (Virginia only)

**Reward Structures**:
- **Distance-based**: Geographic accuracy rewards using Haversine distance
- **Time-based**: Earlier coverage rewards within time windows
- **Hybrid**: Balanced scoring with alpha/beta weighting
- **Regularizers**: Radius penalties and out-of-state penalties

**Profiles**: Baseline, high-LLM, and risk-heavy configurations for different search strategies

### Ground Truth Extraction

**Script**: `reinforcement_learning/extract_ground_truth_coordinates.py`

**Purpose**: Extract coordinates from case files to generate coordinate-based ground truth

**CLI Arguments**:
- `--cases-dir` (str, default: `data/synthetic_cases`): Directory containing case JSON files
- `--output` (str, default: `ground_truth.json`): Output path for ground_truth.json
- `--case-ids` (str list, default: None): Optional list of case IDs to extract (if not provided, extracts all)

**Usage**:
```bash
python reinforcement_learning/extract_ground_truth_coordinates.py \
    --cases-dir data/synthetic_cases \
    --output reinforcement_learning/ground_truth.json
```

**Outputs**: `ground_truth.json` - Coordinate-based ground truth mapping

## Visualization & EDA

**Script**: `eda_hotspot.py`

**Purpose**: Advanced visualization and Kernel Density Estimation (KDE) analysis for case data

**CLI Arguments**:
- `--input` (str, default: `eda_out/eda_cases_min.jsonl`): Path to JSONL or CSV cases
- `--outdir` (str, default: `eda_out`): Output directory
- `--state` (str, default: "VA"): Optional state filter (e.g., VA)
- `--bw` (float, default: 30000.0): KDE bandwidth in meters (Web Mercator)
- `--topN` (int, default: 20): Top-N counties bar chart
- `--skip-counts` (flag): Skip count computation (use counts from run_all_llms.py)
- `--fixed-extent` (flag, default: True): Use a shared geographic extent for all KDE maps
- `--shared-scale` (flag, default: True): Use a shared color scale across KDE maps
- `--clip-shape` (str, default: `data/geo/va_boundary.geojson`): Boundary shapefile/geojson path
- `--roads` (str, default: None): Roads shapefile/geojson path
- `--roads-class-col` (str, default: "CLASS"): Road classification column name
- `--roads-cmap` (str, default: "viridis"): Road colormap name
- `--roads-width` (float, default: 0.5): Road line width
- `--roads-alpha` (float, default: 0.6): Road transparency
- `--no-basemap` (flag): Disable basemap tiles
- `--dark-style` (flag): Use dark poster style
- `--cmap` (str, default: "YlOrRd"): Heat colormap name

**Usage**:
```bash
# Generate EDA charts and KDE hotspot maps
python eda_hotspot.py

# Custom bandwidth and output
python eda_hotspot.py --bw 25000.0 --outdir eda_out/custom
```

**Outputs**:
- `eda_out/distribution_summary.png` - Combined demographic charts
- `eda_out/age_hist.png` - Age distribution histogram
- `eda_out/gender_bar.png` - Gender distribution bar chart
- `eda_out/county_topN_bar.png` - Top-N counties bar chart
- `eda_out/kde_*.png` - KDE hotspot maps by age groups
- `eda_out/maps_dark/kde_*.png` - Dark-style KDE maps
- `eda_out/maps_dot/kde_*.png` - Dot-style KDE maps

**Features**:
- Age-band specific hotspot analysis (≤12, 13-17)
- Fixed geographic extent and shared color scales for comparison
- Web Mercator projection for accurate distance calculations
- Optional basemap integration with contextily
- Road network overlay support

## Evaluation Metrics

**Script**: `calculate_metrics.py`

**Purpose**: Comprehensive metrics calculation system for the Guardian pipeline

**CLI Arguments**:
- `stage` (str, optional, choices: ["ops", "perf", "rl", "extractor", "weak", "summarizer", "e2e", "diagnostics", "clustering", "predictive", "all"], default: "all"): Metrics stage to calculate
- `--zones` (str, choices: ["baseline", "llm"], default: "baseline"): Zone type for RL metrics
- `--config` (str, default: `metrics/metrics_config.json`): Path to metrics config file
- `--output` (str, default: None): Output JSON file path (default: print to stdout)

**Usage**:
```bash
# Calculate all metrics
python calculate_metrics.py

# Calculate specific stage
python calculate_metrics.py rl --zones llm

# Save to file
python calculate_metrics.py all --output metrics.json
```

**Outputs**: JSON files containing metrics for each stage (e.g., `metrics.json` if `--output` is specified)

**Metrics Stages**:
- **ops**: Operational metrics (validation, file existence, data quality)
- **perf**: Performance metrics (LLM timings, processing speeds)
- **rl**: RL metrics (Geo-hit@K, ASUH)
- **extractor**: Entity extraction metrics
- **weak**: Weak labeler metrics
- **summarizer**: Summarization metrics (ROUGE)
- **e2e**: End-to-end pipeline metrics
- **diagnostics**: System diagnostics
- **clustering**: Clustering metrics (silhouette, Davies-Bouldin, bootstrap)
- **predictive**: Predictive consistency metrics

### Metrics Configuration

**Configuration File**: `metrics/metrics_config.json`

```json
{
  "paths": {
    "eda_min": "eda_out/eda_cases_min.jsonl",
    "gold_cases": "data/real_cases/guardian_output.jsonl",
    "synthetic_cases": "data/synthetic_cases/",
    "llm_results": "eda_out/llm_analysis_results.json",
    "zones_baseline": "eda_out/zones_rl.jsonl",
    "zones_llm": "eda_out/zones_reweighted.jsonl"
  },
  "ops": {
    "expect_outputs": [
      "eda_out/distribution_summary.png",
      "eda_out/age_hist.png",
      "eda_out/gender_bar.png",
      "eda_out/maps_dark/kde_all.png",
      "eda_out/zones_review.jsonl"
    ]
  },
  "rl": {
    "ks": [1, 3, 5, 10]
  },
  "geo": {
    "hit_buffer_m": 300
  }
}
```

### Cluster Stability Evaluation

**Script**: `metrics/evaluation/evaluate_clusters.py`

**Purpose**: Evaluates cluster stability across multiple bootstrap iterations for K-Means, DBSCAN, and KDE

**CLI Arguments**:
- `--cases` (str, default: `data/synthetic_cases`): Path to directory containing case JSON files
- `--outdir` (str, default: `eda_out`): Output directory for results
- `--methods` (str list, choices: ["kmeans", "dbscan", "kde"], default: ["kmeans", "dbscan", "kde"]): Clustering methods to evaluate
- `--n-iter` (int, default: 10): Number of bootstrap iterations
- `--sample-ratio` (float, default: 0.85): Fraction of points to sample in each iteration

**Usage**:
```bash
python metrics/evaluation/evaluate_clusters.py \
    --methods kmeans dbscan \
    --n-iter 20 \
    --sample-ratio 0.80
```

**Outputs**:
- `eda_out/cluster_stability.json` - Stability metrics (Jaccard overlap, ARI, silhouette, Davies-Bouldin)
- `eda_out/plots/cluster_stability_*.png` - Stability visualization plots

### Predictive Consistency Evaluation

**Script**: `metrics/evaluation/predictive_consistency.py`

**Purpose**: Evaluates predictive consistency across different time horizons and parameter nudges for the mobility forecasting model

**CLI Arguments**:
- `--cases` (str, default: `data/synthetic_cases`): Path to directory containing case JSON files
- `--outdir` (str, default: `eda_out`): Output directory for results
- `--horizons` (int list, default: [24, 48, 72]): Time horizons to evaluate
- `--top-k` (int, default: 100): Number of top cells for Jaccard overlap
- `--no-nudge-test` (flag): Skip parameter nudge testing
- `--alpha-prior` (float, default: 0.5): Mixing weight for KDE prior
- `--steps-per-24h` (int, default: 3): Markov steps per 24 hours
- `--profile` (str, default: "default"): Survival profile (default, runaway, abduction)

**Usage**:
```bash
python metrics/evaluation/predictive_consistency.py \
    --horizons 24 48 72 \
    --top-k 100
```

**Outputs**:
- `eda_out/predictive_consistency.json` - Consistency metrics (Jaccard overlap, Spearman correlation, KL divergence)
- `eda_out/plots/predictive_consistency_*.png` - Consistency visualization plots

## Model Management & Quantization

Guardian uses specialized LLM models optimized for RTX 4060 (8GB VRAM) with 4-bit quantization.

### Model Download

**Script**: `scripts/download_models.ps1` (PowerShell)

**Purpose**: Downloads models from Hugging Face Hub based on `models.lock.json`

**CLI Arguments**:
- `$LockFile` (string, default: `models.lock.json`): Path to model lock file

**Usage** (PowerShell):
```powershell
# Download all models
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1

# Custom lock file
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1 -LockFile "models.lock.json"
```

**Expected Output**: Models downloaded to `models/Qwen2.5-3B-Instruct/` and `models/Llama3_2-3B-Instruct/`

**Time**: ~15-30 minutes depending on internet speed

### Model Locking

**Script**: `scripts/freeze_lock.py`

**Purpose**: Pins model revisions in `models.lock.json` to exact commit SHAs for reproducibility

**CLI Arguments**: None (runs directly)

**Usage**:
```bash
# Pin all models to current commit SHAs
python scripts/freeze_lock.py
```

**Output**: Updates `models.lock.json` with exact commit hashes instead of "latest" or branch names

### Model Lock File

**File**: `scripts/models.lock.json`

```json
{
  "models": [
    {
      "repo_id": "Qwen/Qwen2.5-3B-Instruct",
      "revision": "latest",
      "local_dir": "models/Qwen2.5-3B-Instruct",
      "role": "extractor"
    },
    {
      "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
      "revision": "latest",
      "local_dir": "models/Llama3_2-3B-Instruct",
      "role": "summarizer"
    },
    {
      "repo_id": "Qwen/Qwen2.5-3B-Instruct",
      "revision": "latest",
      "local_dir": "models/Qwen2.5-3B-Instruct",
      "role": "weak_labeler"
    }
  ]
}
```

### Performance Optimizations

- **Memory Management**: Advanced quantization and offloading for 8GB VRAM
- **Inference Speed**: TF32 acceleration and SDPA attention for RTX 40xx GPUs
- **Output Quality**: Clean token decoding with early stopping and regex backfill
- **4-bit Quantization**: NF4 quantization with bfloat16 compute dtype
- **Double Quantization**: Additional compression for memory efficiency

## GPU & Performance Optimization

### GPU Requirements

**Required**: NVIDIA GPU with 8GB+ VRAM (RTX 4060 or better recommended)

**CUDA Version**: Compatible with CUDA 11.8+ (PyTorch 2.0+)

**GPU VRAM Expectations**:
- **4-bit Quantized Models**: ~3-4GB VRAM per model
- **Batch Processing**: Additional 1-2GB VRAM for batch inference
- **Total Recommended**: 8GB+ VRAM for comfortable operation

### CPU-Only Fallback

Guardian supports CPU-only operation with automatic fallback:

- **Model Loading**: Automatically falls back to CPU if CUDA is unavailable
- **Memory Offloading**: Uses CPU memory offloading for large models
- **Performance**: Significantly slower than GPU (10-100x slower inference)

**Configuration**: Set `"device": "cpu"` in `guardian.config.json` to force CPU mode

### Performance Optimizations

- **TF32 Acceleration**: Enabled for RTX 40xx GPUs (`torch.backends.cuda.matmul.allow_tf32 = True`)
- **SDPA Attention**: Scaled Dot-Product Attention for efficient inference
- **KV Caching**: Reuse computed key-value pairs for faster generation
- **Early Stopping**: Terminate generation at specific markers
- **Batch Processing**: Process multiple cases simultaneously for better GPU utilization
- **Memory Management**: Automatic garbage collection and CUDA cache clearing

### Environment Variables

**Required**:
- `HUGGINGFACE_HUB_TOKEN` - Hugging Face access token for model downloads (required for Llama models)

**Optional**:
- `GUARDIAN_SINGLE_MODEL_DIR` - Override model directory for single-model mode
- `GUARDIAN_SUMM_MODEL` - Override summarizer model path
- `PYTORCH_CUDA_ALLOC_CONF` - CUDA memory allocation configuration (default: `expandable_segments:True`)

## Installation & Environment Setup

### Prerequisites

- **Python 3.9+** with pip package manager
- **NVIDIA GPU with 8GB+ VRAM** (RTX 4060 or better recommended) - **Required for optimal performance**
- **20GB+ disk space** for model storage
- **Hugging Face account** and access token for model downloads
- **Windows PowerShell** (for model download script) or **Bash** (Linux/Mac)

### Step 1: Clone Repository

```bash
# Clone repository
git clone git@github.com:jcast046/Guardian.git
cd Guardian
```

### Step 2: Create Virtual Environment

**Windows (PowerShell)**:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac (Bash)**:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Verification**: Check that `pip list` shows all required packages installed.

### Step 4: Set Environment Variables

**Windows (PowerShell)**:
```powershell
# Set Hugging Face token (required for Llama models)
$env:HUGGINGFACE_HUB_TOKEN = "hf_XXXX"
```

**Linux/Mac (Bash)**:
```bash
# Set Hugging Face token (required for Llama models)
export HUGGINGFACE_HUB_TOKEN="hf_XXXX"
```

### Step 5: Download Models

**Windows (PowerShell)**:
```powershell
# Download all models (Qwen2.5-3B and Llama-3.2-3B)
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1
```

**Linux/Mac (Bash)**:
```bash
# Manual download using Python
python -c "
from huggingface_hub import snapshot_download
import json

with open('scripts/models.lock.json', 'r') as f:
    lock = json.load(f)

for model in lock['models']:
    print(f'Downloading {model[\"repo_id\"]}...')
    snapshot_download(
        repo_id=model['repo_id'],
        revision=model['revision'],
        local_dir=model['local_dir'],
        local_dir_use_symlinks=False
    )
    print(f'Downloaded {model[\"local_dir\"]}')
"
```

**Expected Output**: Models downloaded to `models/Qwen2.5-3B-Instruct/` and `models/Llama3_2-3B-Instruct/`

**Time**: ~15-30 minutes depending on internet speed

### Step 6: Validate Installation

```bash
# Validate all JSON files against schemas
python build.py
```

**Expected Output**: "✓ Build successful - all JSON files are valid!"

## Running the Full End-to-End Pipeline

### Canonical Execution Order

The following is the recommended end-to-end pipeline execution order:

#### 1. Data Validation

```bash
# Validate all JSON files against schemas
python build.py
```

**Required Inputs**: All JSON files in `data/` directory
**Generated Outputs**: Validation results (console output)
**Downstream Dependencies**: None (prerequisite for all other steps)

#### 2. Case Generation

```bash
# Generate 500 synthetic cases (default)
python generate_cases.py

# Or generate smaller test batch
python generate_cases.py --n 50
```

**Required Inputs**: 
- `data/geo/va_gazetteer.json`
- `data/geo/va_rl_regions.geojson`
- `data/transportation/va_road_segments.json`
- `data/transportation/va_transit.json`
- `data/templates/` files
- `data/lexicons/` files

**Generated Outputs**: 
- `data/synthetic_cases/GRD-*.json` - Individual case files

**Downstream Dependencies**: Required for all subsequent steps

#### 3. LLM Processing

```bash
# Process all cases through LLM models (summarizer, extractor, weak labeler)
python run_all_llms.py

# Generate summaries
python run_all_llms.py --do-summary
```

**Required Inputs**: 
- `data/synthetic_cases/GRD-*.json` (from Step 2)

**Generated Outputs**: 
- `eda_out/eda_cases_min.jsonl` - Normalized case data
- `eda_out/eda_counts.json` - Statistical counts
- `eda_out/validation_report.json` - Data quality results
- `eda_out/llm_analysis_results.json` - LLM processing results

**Downstream Dependencies**: Required for clustering, visualization, and zone QA

#### 4. Clustering & Hotspot Extraction

```bash
# Run K-Means diagnostics
python run_kmeans_diagnostics.py

# Run DBSCAN diagnostics
python run_dbscan_diagnostics.py

# Export KDE hotspots
python run_kde_export.py

# Compare clustering methods
python run_cluster_comparison.py
```

**Required Inputs**: 
- `eda_out/eda_cases_min.jsonl` (from Step 3)

**Generated Outputs**: 
- `eda_out/kmeans_hotspots.json`
- `eda_out/dbscan_hotspots.json`
- `eda_out/kde_hotspots.json`
- `eda_out/cluster_compare.json`
- `eda_out/plots/kmeans_elbow.png`, `kmeans_silhouette.png`
- `eda_out/plots/dbscan_kdist.png`
- `eda_out/plots/cluster_compare_*.png`

**Downstream Dependencies**: Required for mobility forecasting and search plan generation

#### 5. Forecasting & RL Zone Generation

```bash
# Generate RL zones
python reinforcement_learning/build_rl_zones.py --mode propagate

# Generate search plans for specific cases
python reinforcement_learning/search_plan_cli.py \
    --case data/synthetic_cases/GRD-2025-000001.json \
    --horizons 24 48 72 \
    --outdir eda_out/search_plans
```

**Required Inputs**: 
- `eda_out/eda_cases_min.jsonl` (from Step 3)
- `eda_out/kmeans_hotspots.json`, `dbscan_hotspots.json`, `kde_hotspots.json` (from Step 4)
- `eda_out/grid_xy.npy`, `road_cost.npy`, `seclusion.npy` (⚠️ Configuration required – see `reinforcement_learning/forecast_api.py`)

**Generated Outputs**: 
- `eda_out/zones_rl.jsonl` - RL-generated search zones
- `eda_out/forecast_plots/<case_id>_search_plan.png` - Search plan visualizations
- `eda_out/forecast_plots/<case_id>_search_plan.json` - Search plan data
- `eda_out/forecast_plots/<case_id>_search_plan_sectors.csv` - Sector rankings

**Downstream Dependencies**: Required for zone QA and metrics evaluation

#### 6. Search Plan Generation

```bash
# Run LLM-enhanced zone evaluation
python zone_qa.py --evaluate
```

**Required Inputs**: 
- `data/synthetic_cases/GRD-*.json` (from Step 2)
- `eda_out/zones_rl.jsonl` (from Step 5)
- `reinforcement_learning/search_reward_config.json`

**Generated Outputs**: 
- `eda_out/zones_review.jsonl` - Zone plausibility scores
- `eda_out/zones_reweighted.jsonl` - LLM-enhanced zones
- `eda_out/zone_qa_metrics.json` - Evaluation metrics
- `eda_out/zone_evaluation_results.json` - Detailed evaluation results

**Downstream Dependencies**: Required for metrics evaluation

#### 7. Metrics & Evaluation

```bash
# Calculate all metrics
python calculate_metrics.py all --output metrics.json

# Evaluate cluster stability
python metrics/evaluation/evaluate_clusters.py

# Evaluate predictive consistency
python metrics/evaluation/predictive_consistency.py
```

**Required Inputs**: 
- All outputs from previous steps

**Generated Outputs**: 
- `metrics.json` - Comprehensive metrics (if `--output` specified)
- `eda_out/cluster_stability.json` - Cluster stability metrics
- `eda_out/predictive_consistency.json` - Predictive consistency metrics
- `eda_out/plots/cluster_stability_*.png` - Stability visualizations
- `eda_out/plots/predictive_consistency_*.png` - Consistency visualizations

**Downstream Dependencies**: None (final evaluation step)

### Visualization & EDA (Optional, can run after Step 3)

```bash
# Generate EDA charts and KDE hotspot maps
python eda_hotspot.py
```

**Required Inputs**: 
- `eda_out/eda_cases_min.jsonl` (from Step 3)

**Generated Outputs**: 
- `eda_out/distribution_summary.png`
- `eda_out/age_hist.png`, `gender_bar.png`, `county_topN_bar.png`
- `eda_out/kde_*.png` - KDE hotspot maps
- `eda_out/maps_dark/kde_*.png` - Dark-style maps
- `eda_out/maps_dot/kde_*.png` - Dot-style maps

## Project Structure Tree

```
Guardian/
├── data/                          # Data storage and templates
│   ├── geo/                       # Geographic data
│   │   ├── va_gazetteer.json      # Virginia locations
│   │   ├── va_rl_regions.geojson  # Regional boundaries
│   │   ├── va_boundary.geojson     # Virginia state boundary
│   │   └── cb_2023_us_state_500k/ # US state shapefiles
│   ├── lexicons/                   # Behavioral and descriptive data (12 files)
│   ├── templates/                 # Case generation templates (8 files)
│   ├── transportation/            # Transportation network data (3 files)
│   ├── synthetic_cases/           # Generated case files (GRD-*.json)
│   ├── real_cases/                # Real case data
│   ├── training/                   # Training data files
│   └── psych_research/            # Psychology research examples (13 files)
├── eda_out/                       # Exploratory Data Analysis outputs
│   ├── maps_dark/                 # Dark-style KDE maps
│   ├── maps_dot/                  # Dot-style KDE maps
│   ├── plots/                     # Diagnostic and comparison plots
│   ├── forecast_plots/          # Forecast and search plan visualizations
│   ├── search_plans/             # Search plan outputs
│   ├── *.jsonl                    # Case data files
│   ├── *.json                     # Metrics and configuration files
│   └── *.png                      # Visualization images
├── guardian_llm/                 # LLM modules and AI components
│   ├── __init__.py
│   ├── extractor.py               # Entity extraction with Qwen2.5-3B
│   ├── summarizer.py              # Case summarization with Qwen2.5-3B
│   ├── weak_labeler.py            # Movement classification with Qwen2.5-3B
│   ├── finetune_qlora.py          # QLoRA fine-tuning system
│   ├── prompts.py                 # Standardized prompt templates
│   └── guardian.config.json       # LLM-specific configuration
├── models/                        # Local model storage (gitignored)
│   ├── Qwen2.5-3B-Instruct/      # Entity extraction and classification model
│   └── Llama3_2-3B-Instruct/     # Summarization model (⚠️ Deprecated)
├── reinforcement_learning/        # RL and forecasting modules
│   ├── forecast_api.py            # Forecasting API
│   ├── movement_model.py         # Movement model (KDE prior, survival)
│   ├── build_rl_zones.py         # RL zone generation
│   ├── rewards.py                # Reward calculation
│   ├── rl_env.py                 # RL environment
│   ├── zone_rl.py                # Zone RL agent
│   ├── sectors.py                # Sector generation
│   ├── rings.py                  # Containment rings
│   ├── visualize_forecast.py    # Forecast visualization
│   ├── extract_ground_truth_coordinates.py  # Ground truth extraction
│   ├── search_plan_cli.py        # Search plan CLI
│   └── search_reward_config.json  # RL reward configuration
├── clustering/                    # Clustering module
│   ├── __init__.py                # Unified clustering interface
│   ├── base.py                    # Clusterer base class
│   ├── kmeans.py                  # K-Means implementation
│   ├── dbscan.py                  # DBSCAN implementation
│   ├── kde.py                     # KDE implementation
│   └── utils.py                   # Coordinate transformation utilities
├── metrics/                       # Metrics and evaluation module
│   ├── __init__.py
│   ├── ops.py                     # Operational metrics
│   ├── perf.py                    # Performance metrics
│   ├── rl.py                      # RL metrics (Geo-hit@K, ASUH)
│   ├── extractor.py               # Entity extraction metrics
│   ├── weak.py                    # Weak labeler metrics
│   ├── summarizer.py              # Summarization metrics (ROUGE)
│   ├── diagnostics.py            # System diagnostics
│   ├── clustering.py             # Clustering metrics
│   ├── io.py                      # I/O utilities
│   ├── config.py                  # Configuration loading
│   ├── metrics_config.json       # Metrics configuration
│   └── evaluation/                # Evaluation scripts
│       ├── evaluate_clusters.py  # Cluster stability evaluation
│       └── predictive_consistency.py  # Predictive consistency evaluation
├── schemas/                       # JSON Schema definitions
│   ├── case_templates.schema.json
│   ├── gazetteer.schema.json
│   ├── guardian_case.schema.json
│   ├── guardian_schema.json
│   ├── road_segment.schema.json
│   ├── transit_line.schema.json
│   └── transit_stop.schema.json
├── scripts/                       # Utility scripts
│   ├── download_models.ps1       # PowerShell model downloader
│   ├── freeze_lock.py            # Pin model revisions to commits
│   └── models.lock.json          # Pinned model versions
├── src/                           # Core Guardian modules
│   ├── geography/                 # Geographic processing modules
│   │   ├── __init__.py
│   │   ├── distance.py           # Distance calculations (Haversine, Manhattan)
│   │   ├── regions.py            # Regional analysis
│   │   └── validation.py        # Geographic validation
│   ├── transportation/            # Transportation analysis modules
│   │   ├── __init__.py
│   │   └── networks.py           # Network analysis (transit, roads)
│   ├── priors.py                 # Behavioral prior sampling
│   └── guardian_modules.py       # Main Guardian functionality
├── tests/                         # Test suite
│   ├── unit_tests/                # Unit tests
│   └── integration_tests/        # Integration tests
├── .gitignore                     # Git ignore patterns for models
├── build.py                       # Main validation script
├── calculate_metrics.py          # Metrics calculation script
├── eda_hotspot.py                # Exploratory data analysis
├── generate_cases.py             # Synthetic case generator
├── generate_cases_organized.py   # Organized case generation (alternative)
├── guardian.config.json          # Main model configuration
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── run_all_llms.py               # LLM testing and evaluation
├── run_cluster_comparison.py      # Cluster comparison script
├── run_dbscan_diagnostics.py      # DBSCAN diagnostics
├── run_kde_export.py             # KDE export script
├── run_kmeans_diagnostics.py     # K-Means diagnostics
└── zone_qa.py                    # Zone quality assurance
```

## Key Scripts & Their Exact Outputs

### Root-Level Scripts

1. **`build.py`** - Schema validation
   - **CLI Arguments**: None
   - **Inputs**: All JSON files in `data/` directory
   - **Outputs**: Console validation results

2. **`generate_cases.py`** - Synthetic case generation
   - **CLI Arguments**: `--n` (int, default: 500), `--seed` (int, default: 42), `--out` (Path, default: `data/synthetic_cases`)
   - **Inputs**: Geographic data, templates, lexicons, transportation data
   - **Outputs**: `data/synthetic_cases/GRD-*.json`

3. **`generate_cases_organized.py`** - Alternative case generation
   - **CLI Arguments**: `--n` (int, default: 500), `--seed` (int, default: 42), `--out` (Path, default: `data/synthetic_cases`)
   - **Inputs**: Same as `generate_cases.py`
   - **Outputs**: Same as `generate_cases.py`

4. **`run_all_llms.py`** - LLM processing pipeline
   - **CLI Arguments**: `--reasoned` (flag), `--do-summary` (flag), `--fallback-extractor` (flag)
   - **Inputs**: `data/synthetic_cases/GRD-*.json`
   - **Outputs**: `eda_out/eda_cases_min.jsonl`, `eda_out/eda_counts.json`, `eda_out/validation_report.json`, `eda_out/llm_analysis_results.json`

5. **`eda_hotspot.py`** - Visualization and KDE analysis
   - **CLI Arguments**: `--input`, `--outdir`, `--state`, `--bw`, `--topN`, `--skip-counts`, `--fixed-extent`, `--shared-scale`, `--clip-shape`, `--roads`, `--roads-class-col`, `--roads-cmap`, `--roads-width`, `--roads-alpha`, `--no-basemap`, `--dark-style`, `--cmap`
   - **Inputs**: `eda_out/eda_cases_min.jsonl`
   - **Outputs**: `eda_out/distribution_summary.png`, `eda_out/age_hist.png`, `eda_out/gender_bar.png`, `eda_out/county_topN_bar.png`, `eda_out/kde_*.png`, `eda_out/maps_dark/kde_*.png`, `eda_out/maps_dot/kde_*.png`

6. **`zone_qa.py`** - LLM-enhanced zone analysis
   - **CLI Arguments**: `--input`, `--config`, `--outdir`, `--evaluate`, `--ttf`, `--cdf`, `--profile`, `--sample`, `--selftest`, `--force-real`, `--verbose`, `--format`, `--per-zone`, `--batch-size`, `--print-models`
   - **Inputs**: `data/synthetic_cases/GRD-*.json`, `reinforcement_learning/search_reward_config.json`
   - **Outputs**: `eda_out/zones_review.jsonl`, `eda_out/zones_reweighted.jsonl`, `eda_out/zone_qa_metrics.json`, `eda_out/zone_evaluation_results.json`

7. **`calculate_metrics.py`** - Metrics calculation
   - **CLI Arguments**: `stage` (ops/perf/rl/extractor/weak/summarizer/e2e/diagnostics/clustering/predictive/all), `--zones` (baseline/llm), `--config`, `--output`
   - **Inputs**: Various outputs from pipeline stages
   - **Outputs**: JSON metrics (stdout or file if `--output` specified)

8. **`run_kmeans_diagnostics.py`** - K-Means diagnostics
   - **CLI Arguments**: `--input`, `--out`, `--kmin`, `--kmax`, `--random-state`
   - **Inputs**: `eda_out/eda_cases_min.jsonl`
   - **Outputs**: `eda_out/kmeans_hotspots.json`, `eda_out/plots/kmeans_elbow.png`, `eda_out/plots/kmeans_silhouette.png`

9. **`run_dbscan_diagnostics.py`** - DBSCAN diagnostics
   - **CLI Arguments**: `--input`, `--out`, `--min-samples`, `--eps-meters`, `--k`
   - **Inputs**: `eda_out/eda_cases_min.jsonl`
   - **Outputs**: `eda_out/dbscan_hotspots.json`, `eda_out/plots/dbscan_kdist.png`

10. **`run_kde_export.py`** - KDE export
    - **CLI Arguments**: `--input`, `--out`, `--bandwidth-meters`, `--iso-mass`
    - **Inputs**: `eda_out/eda_cases_min.jsonl`
    - **Outputs**: `eda_out/kde_hotspots.json`

11. **`run_cluster_comparison.py`** - Cluster comparison
    - **CLI Arguments**: `--in-dir`, `--out`, `--topN`, `--kde-polygons`
    - **Inputs**: `eda_out/kmeans_hotspots.json`, `eda_out/dbscan_hotspots.json`, `eda_out/kde_hotspots.json`
    - **Outputs**: `eda_out/cluster_compare.json`, `eda_out/plots/cluster_compare_side_by_side.png`, `eda_out/plots/cluster_compare_overlay.png`

### Reinforcement Learning Scripts

12. **`reinforcement_learning/search_plan_cli.py`** - Search plan generation
    - **CLI Arguments**: `--case`, `--horizons`, `--outdir`, `--use-cumulative`, `--no-cumulative`, `--hotspot-pct`, `--sector-path`, `--alpha-prior`, `--steps-per-24h`, `--beta-corr-day`, `--beta-corr-night`, `--profile`, `--max-hotspots-per-sector`
    - **Inputs**: Case JSON file, hotspot files, grid data
    - **Outputs**: `eda_out/forecast_plots/<case_id>_search_plan.png`, `eda_out/forecast_plots/<case_id>_search_plan_t<horizon>h.png`, `eda_out/forecast_plots/<case_id>_search_plan.json`, `eda_out/forecast_plots/<case_id>_search_plan_sectors.csv`, `eda_out/forecast_plots/<case_id>_search_plan_sectors_by_horizon.csv`

13. **`reinforcement_learning/visualize_forecast.py`** - Forecast visualization
    - **CLI Arguments**: `--case`, `--horizons`, `--output-dir`, `--alpha-prior`, `--steps-per-24h`, `--cumulative`, `--cumulative-weights`, `--cumulative-mode`
    - **Inputs**: Case JSON file
    - **Outputs**: PNG forecast plots in output directory

14. **`reinforcement_learning/build_rl_zones.py`** - RL zone generation
    - **CLI Arguments**: `--mode` (baseline/propagate), `--config`, `--outdir`, `--log-debug`, `--sample`
    - **Inputs**: Case files, reward configuration
    - **Outputs**: `eda_out/zones_rl.jsonl`

15. **`reinforcement_learning/extract_ground_truth_coordinates.py`** - Ground truth extraction
    - **CLI Arguments**: `--cases-dir`, `--output`, `--case-ids`
    - **Inputs**: Case JSON files
    - **Outputs**: `ground_truth.json`

### Metrics Evaluation Scripts

16. **`metrics/evaluation/evaluate_clusters.py`** - Cluster stability evaluation
    - **CLI Arguments**: `--cases`, `--outdir`, `--methods`, `--n-iter`, `--sample-ratio`
    - **Inputs**: Case JSON files
    - **Outputs**: `eda_out/cluster_stability.json`, `eda_out/plots/cluster_stability_*.png`

17. **`metrics/evaluation/predictive_consistency.py`** - Predictive consistency evaluation
    - **CLI Arguments**: `--cases`, `--outdir`, `--horizons`, `--top-k`, `--no-nudge-test`, `--alpha-prior`, `--steps-per-24h`, `--profile`
    - **Inputs**: Case JSON files
    - **Outputs**: `eda_out/predictive_consistency.json`, `eda_out/plots/predictive_consistency_*.png`

### Model Management Scripts

18. **`scripts/download_models.ps1`** - Model download (PowerShell)
    - **CLI Arguments**: `$LockFile` (string, default: `models.lock.json`)
    - **Inputs**: `scripts/models.lock.json`
    - **Outputs**: Models in `models/` directory

19. **`scripts/freeze_lock.py`** - Model version locking
    - **CLI Arguments**: None
    - **Inputs**: `scripts/models.lock.json`
    - **Outputs**: Updated `scripts/models.lock.json` with commit SHAs

## Development & Extension Guide

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

### Testing

Guardian includes comprehensive test suites:

**Unit Tests** (`tests/unit_tests/`):
- Clustering algorithms
- Geography and transportation modules
- LLM modules (extractor, summarizer, weak labeler)
- Metrics calculation
- RL components

**Integration Tests** (`tests/integration_tests/`):
- End-to-end pipeline tests
- Forecast API tests
- Search plan generation
- Metrics evaluation

**Run Tests**:
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit_tests/test_clustering.py

# Run with coverage
pytest tests/ --cov=.
```

### Extending the Clustering System

To add a new clustering algorithm:

1. Create a new class inheriting from `clustering.base.Clusterer`
2. Implement required methods: `fit()`, `get_hotspots()`, `get_labels()`
3. Add factory function in `clustering/__init__.py`
4. Update diagnostics scripts if needed

### Extending the Metrics System

To add new metrics:

1. Create a new module in `metrics/` (e.g., `metrics/custom.py`)
2. Implement metric calculation functions
3. Add to `calculate_metrics.py` with appropriate stage
4. Update `metrics/metrics_config.json` if configuration needed

### Custom Movement Models

To add custom movement model profiles:

1. Update `reinforcement_learning/movement_model.py` with new profile
2. Add profile configuration to `reinforcement_learning/search_reward_config.json`
3. Update `forecast_api.py` to support new profile

## Reproducibility Checklist

To ensure reproducible results:

- [ ] **Python Version**: Use Python 3.9+ (verify with `python --version`)
- [ ] **Dependencies**: Pin dependency versions in `requirements.txt` (consider using `pip freeze > requirements.lock.txt`)
- [ ] **Model Versions**: Run `python scripts/freeze_lock.py` to pin model revisions to exact commit SHAs
- [ ] **Random Seeds**: Use consistent random seeds (default: 42) for case generation and clustering
- [ ] **Data Sources**: Ensure all data files in `data/` are at specific versions (consider git LFS or versioned data)
- [ ] **Configuration**: Document all configuration changes in `guardian.config.json` and `metrics/metrics_config.json`
- [ ] **Environment Variables**: Document all required environment variables (especially `HUGGINGFACE_HUB_TOKEN`)
- [ ] **GPU/CUDA**: Document CUDA version and GPU model for performance reproducibility
- [ ] **Output Verification**: Save all intermediate outputs (`eda_out/`) for reproducibility
- [ ] **Execution Log**: Keep execution logs with timestamps and command-line arguments

## Limitations & Assumptions

### Geographic Scope

- **Virginia-Only**: Guardian is designed specifically for Virginia missing-child cases
- **Regional Boundaries**: Uses Virginia regional boundaries (`va_rl_regions.geojson`)
- **State Boundary**: Enforces Virginia state boundary in search zone generation
- **Transportation Networks**: Limited to Virginia road segments and transit networks

### Data Limitations

- **Synthetic Data**: All cases are synthetically generated (not real missing-person cases)
- **Schema Validation**: Requires strict adherence to Guardian JSON Schema
- **Ground Truth**: Ground truth data is limited to synthetic case outcomes
- **Real Case Integration**: Real case data from Guardian Parser requires schema conversion

### Model Limitations

- **Model Size**: 3B-parameter models may have limitations in complex reasoning
- **Quantization**: 4-bit quantization may slightly reduce model accuracy
- **GPU Requirements**: Optimal performance requires 8GB+ VRAM GPU
- **CPU Fallback**: CPU-only operation is significantly slower (10-100x)

### Algorithm Limitations

- **Markov Chain**: Mobility forecasting uses simplified Markov chain propagation
- **Survival Analysis**: Basic survival models (not advanced Kaplan-Meier or Cox models)
- **Clustering**: K-Means, DBSCAN, and KDE have inherent limitations (cluster shape assumptions, parameter sensitivity)
- **RL Optimization**: RL zone generation uses simplified reward structures

### Performance Limitations

- **Scalability**: Large-scale case generation (1000+ cases) may require significant memory
- **LLM Processing**: LLM inference is the bottleneck (20-30 minutes for 500 cases)
- **Visualization**: Large-scale visualization may be slow without GPU acceleration
- **Disk Space**: Model storage requires 20GB+ disk space

## Ethical & Legal Use Disclaimer

**⚠️ IMPORTANT: Research Use Only**

Guardian is a **research proof-of-concept system** and is **NOT intended for production use** in real missing-person cases. The following disclaimers apply:

### Research Purpose

- Guardian is designed for **academic research** and **proof-of-concept evaluation**
- The system uses **synthetic data** and is not validated for real-world missing-person cases
- Results should **NOT** be used to make decisions in actual search and rescue operations

### Data Privacy

- Guardian processes **synthetic case data** only (not real missing-person cases)
- Real case data from Guardian Parser should be handled according to applicable privacy laws
- Users are responsible for ensuring compliance with data privacy regulations (HIPAA, GDPR, etc.)

### Model Limitations

- LLM models may produce **inaccurate or biased results**
- Entity extraction, summarization, and risk assessment are **not validated** for real-world accuracy
- Model outputs should **NOT** be used as the sole basis for investigative decisions

### Legal Compliance

- Users are responsible for ensuring compliance with all applicable laws and regulations
- Guardian does **NOT** provide legal advice or guarantee legal compliance
- Use of Guardian in real missing-person cases is **at the user's own risk**

### Ethical Considerations

- Missing-person cases involve **sensitive and potentially traumatic situations**
- Users should consider the **ethical implications** of automated case analysis
- **Human judgment** should always be the primary factor in search and rescue decisions

### No Warranty

- Guardian is provided **"as is"** without warranty of any kind
- The authors and contributors are **not responsible** for any consequences of using Guardian
- Users assume **all risks** associated with using this system

**By using Guardian, you acknowledge that you have read, understood, and agree to these disclaimers.**

## Core Algorithms

Guardian implements several core algorithms for geographic analysis, network processing, and machine learning:

- **Haversine Distance**: Calculates great-circle distances between geographic coordinates
- **Dijkstra's Algorithm**: Finds shortest paths in transit networks with bounded search
- **Markov Chain Propagation**: Probabilistic movement model for risk distribution forecasting
- **Kernel Density Estimation**: Spatial hotspot detection from historical case data
- **Survival Analysis**: Temporal decay models for disappearance timelines
- **K-Means Clustering**: Partition-based clustering for hotspot detection
- **DBSCAN Clustering**: Density-based clustering for irregular hotspot shapes
- **4-bit Quantization (NF4)**: Memory-efficient model loading for 8GB VRAM systems
- **QLoRA**: Parameter-efficient fine-tuning with minimal memory requirements

See `Algorithms.md` for detailed algorithm documentation.

## Known Gaps & WIP Components

### Work in Progress (WIP)

- **Advanced Survival Models**: Kaplan-Meier and Cox proportional hazards models (currently basic survival analysis)
- **Corridor-Aware Movement**: Highway preference modeling in movement forecasts
- **Enhanced Visualizations**: GIF generation for forecast timelines
- **Real Case Integration**: Improved schema conversion for Guardian Parser output
- **Advanced RL Training**: Full RL training pipeline (currently zone generation only)

### Known Limitations

- **Grid Data Generation**: `eda_out/grid_xy.npy`, `road_cost.npy`, `seclusion.npy` generation not fully documented (⚠️ Configuration required – see `reinforcement_learning/forecast_api.py`)
- **Sector Generation**: Sector GeoJSON file generation process not fully automated
- **Model Fine-tuning**: QLoRA fine-tuning system exists but training data preparation is manual
- **Real Case Validation**: Limited validation of real case data from Guardian Parser

### Future Improvements

- **Multi-State Support**: Extend beyond Virginia to other US states
- **Advanced RL**: Full reinforcement learning training with policy gradients
- **Real-Time Processing**: Real-time case processing and zone generation
- **Web Interface**: Web-based interface for case management and visualization
- **API Integration**: REST API for programmatic access to Guardian functionality

## Troubleshooting

### Common Issues

**Model Download Failures**:
- Verify `HUGGINGFACE_HUB_TOKEN` is set correctly
- Check internet connection and Hugging Face Hub availability
- Ensure sufficient disk space (20GB+)

**CUDA Out of Memory**:
- Reduce batch size in `guardian.config.json` (`"batch_size": 8`)
- Use CPU fallback mode (`"device": "cpu"`)
- Process cases in smaller batches

**Schema Validation Errors**:
- Check JSON file syntax with a JSON validator
- Verify schema files exist in `schemas/` directory
- Review error messages for specific field issues

**Missing Output Files**:
- Verify all prerequisite steps completed successfully
- Check file paths in configuration files
- Review console output for error messages

**GPU Not Detected**:
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU drivers are up to date
- System will automatically fall back to CPU if GPU unavailable

## License

⚠️ Configuration required – see repository root for license information.

## Citation

If you use Guardian in your research, please cite:

⚠️ Configuration required – see repository for citation information.

## Contact & Support

⚠️ Configuration required – see repository for contact information.

---

**Last Updated**: 2025-12-07

**Version**: 1.0.0

**Status**: Research Proof-of-Concept
