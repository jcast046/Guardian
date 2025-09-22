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

### Generate Synthetic Cases
```bash
# Generate 10 synthetic cases
python generate_cases.py --n 10

# Generate with specific seed for reproducibility
python generate_cases.py --n 5 --seed 42

# Output to custom directory
python generate_cases.py --n 20 --out data/custom_cases
```

### Validate Data
```bash
# Validate all JSON files
python build.py

# Install dependencies first
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
├── build.py                    # Main validation script
├── generate_cases.py           # Synthetic case generator
├── LICENSE
└── README.md
```

## Development

### Adding New Data
1. Place JSON files in appropriate `data/` subdirectories
2. Create corresponding schema files in `schemas/`
3. Update `build.py` if needed for new schema detection
4. Run `python build.py` to validate

### Schema Validation
The validation system uses intelligent schema detection:
- **Path-based**: Files in `synthetic_cases/` use `guardian_schema.json`
- **Content-based**: Analyzes JSON structure to determine appropriate schema
- **Fallback**: Skips files without detectable schema patterns

### Synthetic Case Generation
The generator creates realistic cases by:
- Selecting locations from Virginia gazetteer
- Using real road segments and transit stations
- Incorporating RL search patterns and time windows
- Generating consistent vehicle and witness information
- Creating geographically sensible movement patterns