#!/usr/bin/env python3
"""
Simple build script for Guardian project.
Validates all JSON files in data/ directory against schemas.
"""

import json
import sys
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError


def load_schemas():
    """Load all schemas from schemas/ directory."""
    schemas = {}
    schemas_dir = Path("schemas")
    
    schema_files = {
        'guardian': 'guardian_schema.json',
        'road_segment': 'road_segment.schema.json', 
        'transit_line': 'transit_line.schema.json',
        'transit_stop': 'transit_stop.schema.json',
        'gazetteer': 'gazetteer.schema.json',
        'templates': 'case_templates.schema.json'
    }
    
    for schema_name, filename in schema_files.items():
        schema_path = schemas_dir / filename
        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                schemas[schema_name] = json.load(f)
    
    return schemas


def detect_schema_type(file_path, json_data):
    """Detect which schema to use for a JSON file."""
    # Path-based detection
    if 'synthetic_samples' in str(file_path):
        return 'guardian'
    elif 'templates' in str(file_path):
        return 'templates'
    elif 'geo' in str(file_path) and 'gazetteer' in str(file_path):
        return 'gazetteer'
    
    # Content-based detection
    if 'case_id' in json_data and 'demographic' in json_data:
        return 'guardian'
    elif 'synthetic_case_templates' in json_data:
        return 'templates'
    elif 'version' in json_data and 'crs' in json_data and 'entries' in json_data:
        return 'gazetteer'
    elif 'segmentId' in json_data and 'localNames' in json_data:
        return 'road_segment'
    elif 'id' in json_data and 'type' in json_data and 'geometry' in json_data:
        if json_data.get('type') in ['rail', 'subway', 'light_rail', 'tram', 'bus', 'ferry', 'cable_car', 'funicular', 'monorail', 'trolleybus']:
            return 'transit_line'
        elif json_data.get('type') in ['station', 'halt', 'stop', 'platform', 'bus_stop', 'tram_stop', 'subway_entrance', 'ferry_terminal']:
            return 'transit_stop'
    
    return None


def validate_json_file(file_path, schemas):
    """Validate a single JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"
    
    schema_type = detect_schema_type(file_path, json_data)
    if not schema_type:
        return False, f"Could not determine schema type"
    
    if schema_type not in schemas:
        return False, f"Schema '{schema_type}' not found"
    
    try:
        validate(instance=json_data, schema=schemas[schema_type])
        return True, None
    except ValidationError as e:
        return False, f"Schema error: {e.message}"


def main():
    """Main build function."""
    print("Guardian Build - Validating JSON files...")
    
    # Load schemas
    schemas = load_schemas()
    if not schemas:
        print("Error: No schemas found")
        sys.exit(1)
    
    # Find all JSON files in data/
    data_dir = Path("data")
    if not data_dir.exists():
        print("Error: data/ directory not found")
        sys.exit(1)
    
    json_files = list(data_dir.rglob("*.json"))
    if not json_files:
        print("No JSON files found in data/")
        sys.exit(0)
    
    # Validate each file
    valid_count = 0
    total_count = len(json_files)
    
    for json_file in sorted(json_files):
        print(f"Validating: {json_file.relative_to(data_dir)}")
        is_valid, error = validate_json_file(json_file, schemas)
        
        if is_valid:
            print("  ✓ Valid")
            valid_count += 1
        else:
            print(f"  ✗ Invalid: {error}")
    
    print(f"\nValidation Summary: {valid_count}/{total_count} files valid")
    
    if valid_count == total_count:
        print("✓ Build successful - all JSON files are valid!")
        sys.exit(0)
    else:
        print(f"✗ Build failed - {total_count - valid_count} files have errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
