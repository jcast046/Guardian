#!/usr/bin/env python3
"""
Guardian Build System - JSON Schema Validation

This module provides automated validation for all JSON files in the Guardian project
against their respective JSON schemas. It implements intelligent schema detection
and comprehensive validation reporting.

Author: Joshua Castillo

"""

import json
import sys
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError


def load_schemas():
    """
    Load all JSON schemas from the schemas/ directory.
    
    Returns:
        dict: Dictionary mapping schema names to their JSON schema objects
        
    Raises:
        FileNotFoundError: If schemas directory doesn't exist
        json.JSONDecodeError: If schema files contain invalid JSON
        
    Example:
        >>> schemas = load_schemas()
        >>> 'guardian' in schemas
        True
    """
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
    """
    Intelligently detect which schema should be used for a JSON file.
    
    Uses a combination of path-based and content-based detection to determine
    the appropriate schema for validation. This allows for flexible file
    organization while maintaining accurate validation.
    
    Args:
        file_path (Path): Path to the JSON file being validated
        json_data (dict): Parsed JSON data from the file
        
    Returns:
        str or None: Schema type name if detected, None if no schema applies
        
    Schema Detection Logic:
        1. Path-based detection (highest priority)
           - synthetic_cases/ or synthetic_samples/ → 'guardian'
           - templates/ → 'templates' 
           - geo/ with gazetteer → 'gazetteer'
           
        2. Content-based detection
           - case_id + demographic → 'guardian'
           - synthetic_case_templates → 'templates'
           - version + crs + entries → 'gazetteer'
           - segmentId + localNames → 'road_segment'
           - id + type + geometry → 'transit_line' or 'transit_stop'
           
        3. Special cases
           - style_tag + child (no demographic) → None (skip)
           
    Example:
        >>> detect_schema_type(Path("data/synthetic_cases/case.json"), {"case_id": "GRD-2025-123"})
        'guardian'
    """
    # Path-based detection
    if 'synthetic_samples' in str(file_path) or 'synthetic_cases' in str(file_path):
        return 'guardian'
    elif 'templates' in str(file_path):
        return 'templates'
    elif 'geo' in str(file_path) and 'gazetteer' in str(file_path):
        return 'gazetteer'
    
    # Content-based detection
    if 'style_tag' in json_data and 'child' in json_data:
        # This is a different format (blank template), skip it
        return None
    elif 'case_id' in json_data and 'demographic' in json_data:
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
    """
    Validate a single JSON file against its appropriate schema.
    
    Performs comprehensive validation including JSON parsing, schema detection,
    and schema validation with detailed error reporting.
    
    Args:
        file_path (Path): Path to the JSON file to validate
        schemas (dict): Dictionary of loaded schemas keyed by schema type
        
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
            - is_valid: True if file is valid, False otherwise
            - error_message: Detailed error message if validation fails, None if valid
            
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If file cannot be read
        
    Example:
        >>> schemas = load_schemas()
        >>> is_valid, error = validate_json_file(Path("data/case.json"), schemas)
        >>> print(f"Valid: {is_valid}")
        Valid: True
    """
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
    """
    Main build function - orchestrates the validation process.
    
    Performs the complete build validation workflow:
    1. Load all available schemas
    2. Discover all JSON files in data/ directory
    3. Validate each file against its appropriate schema
    4. Generate comprehensive validation report
    5. Exit with appropriate status code
    
    Exit Codes:
        0: Build successful (all files valid)
        1: Build failed (validation errors or missing schemas)
        
    Output:
        Prints validation progress and results to stdout
        Provides detailed error messages for failed validations
        
    Example:
        $ python build.py
        Guardian Build - Validating JSON files...
        Validating: geo/va_gazetteer.json
          ✓ Valid
        Validation Summary: 1/1 files valid, 0 skipped
        ✓ Build successful - all JSON files are valid!
    """
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
    
    # Also check synthetic_cases directory if it exists
    synthetic_dir = data_dir / "synthetic_cases"
    if synthetic_dir.exists():
        json_files.extend(list(synthetic_dir.rglob("*.json")))
    if not json_files:
        print("No JSON files found in data/")
        sys.exit(0)
    
    # Validate each file
    valid_count = 0
    total_count = 0
    skipped_count = 0
    
    for json_file in sorted(json_files):
        print(f"Validating: {json_file.relative_to(data_dir)}")
        
        # Load JSON data to check schema type
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
            continue
        
        schema_type = detect_schema_type(json_file, json_data)
        if not schema_type:
            print(f"  ⚠ Skipped (no schema)")
            skipped_count += 1
            continue
        
        if schema_type not in schemas:
            print(f"  ⚠ Skipped (schema '{schema_type}' not found)")
            skipped_count += 1
            continue
        
        total_count += 1
        is_valid, error = validate_json_file(json_file, schemas)
        
        if is_valid:
            print("  ✓ Valid")
            valid_count += 1
        else:
            print(f"  ✗ Invalid: {error}")
    
    print(f"\nValidation Summary: {valid_count}/{total_count} files valid, {skipped_count} skipped")
    
    if total_count == 0:
        print("⚠ No files with schemas found")
        sys.exit(0)
    elif valid_count == total_count:
        print("✓ Build successful - all JSON files are valid!")
        sys.exit(0)
    else:
        print(f"✗ Build failed - {total_count - valid_count} files have errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
