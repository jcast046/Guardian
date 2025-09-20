# Guardian
Guardian is a lightweight, AI-driven OSINT system focused on consolidating and analyzing reports of missing women and children.  The purpose of this project is to build a proof-of-concept tool that demonstrates how natural language processing (NLP), unsupervised machine learning, and geospatial visualization can be applied to identify trends, patterns, and hotspots to more directly assist in locating victims in missing person cases.

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