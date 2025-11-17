"""Unit tests for metrics.diagnostics module."""

import pytest
from metrics.diagnostics import calc_diagnostics


class TestDiagnosticsHelpers:
    """Test suite for metrics.diagnostics module.

    Tests diagnostic metrics calculation including counts, ID overlaps,
    and field detection across different data files.
    """

    def test_diagnostics_structure(self, sample_config, tmp_path, monkeypatch):
        """Test that diagnostics returns correct structure."""
        import json
        
        # Create temporary test files
        eda_file = tmp_path / "eda_cases_min.jsonl"
        gold_file = tmp_path / "cases_gold.jsonl"
        truth_file = tmp_path / "zone_truth.jsonl"
        zones_file = tmp_path / "zones_rl.jsonl"
        
        # Write test data
        eda_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        gold_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        truth_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        zones_file.write_text('{"case_id": "GRD-001", "zones": {}}\n', encoding="utf-8")
        
        # Update config paths
        sample_config["paths"]["eda_min"] = str(eda_file)
        sample_config["paths"]["gold_cases"] = str(gold_file)
        sample_config["paths"]["gold_zones"] = str(truth_file)
        sample_config["paths"]["zones_baseline"] = str(zones_file)
        
        result = calc_diagnostics(sample_config)
        
        assert "timestamp" in result
        assert "stage" in result
        assert result["stage"] == "diagnostics"
        assert "diagnostics" in result
        assert "counts" in result["diagnostics"]
        assert "id_overlap" in result["diagnostics"]

    def test_diagnostics_counts(self, sample_config, tmp_path):
        """Test that diagnostics counts cases correctly."""
        import json
        
        # Create test files with multiple cases
        eda_file = tmp_path / "eda_cases_min.jsonl"
        gold_file = tmp_path / "cases_gold.jsonl"
        truth_file = tmp_path / "zone_truth.jsonl"
        zones_file = tmp_path / "zones_rl.jsonl"
        
        eda_file.write_text(
            '{"case_id": "GRD-001"}\n{"case_id": "GRD-002"}\n',
            encoding="utf-8"
        )
        gold_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        truth_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        zones_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        
        sample_config["paths"]["eda_min"] = str(eda_file)
        sample_config["paths"]["gold_cases"] = str(gold_file)
        sample_config["paths"]["gold_zones"] = str(truth_file)
        sample_config["paths"]["zones_baseline"] = str(zones_file)
        
        result = calc_diagnostics(sample_config)
        
        assert result["diagnostics"]["counts"]["eda"] == 2
        assert result["diagnostics"]["counts"]["gold"] == 1

    def test_diagnostics_id_overlap(self, sample_config, tmp_path):
        """Test that diagnostics calculates ID overlaps correctly."""
        import json
        
        eda_file = tmp_path / "eda_cases_min.jsonl"
        gold_file = tmp_path / "cases_gold.jsonl"
        truth_file = tmp_path / "zone_truth.jsonl"
        zones_file = tmp_path / "zones_rl.jsonl"
        
        # Create overlapping IDs
        eda_file.write_text('{"case_id": "GRD-001"}\n{"case_id": "GRD-002"}\n', encoding="utf-8")
        gold_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        truth_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        zones_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        
        sample_config["paths"]["eda_min"] = str(eda_file)
        sample_config["paths"]["gold_cases"] = str(gold_file)
        sample_config["paths"]["gold_zones"] = str(truth_file)
        sample_config["paths"]["zones_baseline"] = str(zones_file)
        
        result = calc_diagnostics(sample_config)
        
        assert result["diagnostics"]["id_overlap"]["zones∩truth"] == 1
        assert result["diagnostics"]["id_overlap"]["gold∩eda"] == 1

    def test_diagnostics_field_detection(self, sample_config, tmp_path):
        """Test that diagnostics detects fields correctly."""
        import json
        
        eda_file = tmp_path / "eda_cases_min.jsonl"
        gold_file = tmp_path / "cases_gold.jsonl"
        truth_file = tmp_path / "zone_truth.jsonl"
        zones_file = tmp_path / "zones_rl.jsonl"
        
        eda_file.write_text(
            '{"case_id": "GRD-001", "summary": "test", "field1": "value1"}\n',
            encoding="utf-8"
        )
        gold_file.write_text(
            '{"case_id": "GRD-001", "entities": {}, "movement_profile": "driving"}\n',
            encoding="utf-8"
        )
        truth_file.write_text('{"case_id": "GRD-001"}\n', encoding="utf-8")
        zones_file.write_text('{"case_id": "GRD-001", "zones": {}}\n', encoding="utf-8")
        
        sample_config["paths"]["eda_min"] = str(eda_file)
        sample_config["paths"]["gold_cases"] = str(gold_file)
        sample_config["paths"]["gold_zones"] = str(truth_file)
        sample_config["paths"]["zones_baseline"] = str(zones_file)
        
        result = calc_diagnostics(sample_config)
        
        assert "eda_fields_present" in result["diagnostics"]
        assert "gold_has_entities" in result["diagnostics"]
        assert "gold_has_movement_profile" in result["diagnostics"]
        assert "eda_has_summary" in result["diagnostics"]
