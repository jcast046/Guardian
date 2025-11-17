"""Unit tests for generate_cases.py module."""

import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock
import generate_cases


class TestGenerateCase:
    """Test suite for generate_cases.generate_case function.

    Tests case generation with demographic, spatial, and temporal data
    including coordinate validity and age range validation.
    """

    def test_generate_case_basic(self):
        """Test basic case generation."""
        case_id = "GRD-001"
        case_data = generate_cases.generate_case(case_id)
        
        assert case_data["case_id"] == case_id
        assert "demographic" in case_data
        assert "spatial" in case_data
        assert "temporal" in case_data

    def test_generate_case_demographic(self):
        """Test case generation includes demographic data."""
        case_id = "GRD-001"
        case_data = generate_cases.generate_case(case_id)
        
        assert "age_years" in case_data["demographic"]
        assert "gender" in case_data["demographic"]
        assert "name" in case_data["demographic"]

    def test_generate_case_spatial(self):
        """Test case generation includes spatial data."""
        case_id = "GRD-001"
        case_data = generate_cases.generate_case(case_id)
        
        assert "last_seen_lat" in case_data["spatial"]
        assert "last_seen_lon" in case_data["spatial"]
        assert "last_seen_city" in case_data["spatial"]
        assert "last_seen_county" in case_data["spatial"]
        assert "last_seen_state" in case_data["spatial"]

    def test_generate_case_temporal(self):
        """Test case generation includes temporal data."""
        case_id = "GRD-001"
        case_data = generate_cases.generate_case(case_id)
        
        assert "reported_missing_ts" in case_data["temporal"]
        assert case_data["temporal"]["reported_missing_ts"] is not None

    def test_generate_case_coordinate_validity(self):
        """Test that generated coordinates are valid."""
        case_id = "GRD-001"
        case_data = generate_cases.generate_case(case_id)
        
        lat = case_data["spatial"]["last_seen_lat"]
        lon = case_data["spatial"]["last_seen_lon"]
        
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180

    def test_generate_case_age_range(self):
        """Test that generated age is within valid range."""
        case_id = "GRD-001"
        case_data = generate_cases.generate_case(case_id)
        
        age = case_data["demographic"]["age_years"]
        
        assert 0 <= age <= 18

    def test_generate_case_state_va(self):
        """Test that generated state is Virginia."""
        case_id = "GRD-001"
        case_data = generate_cases.generate_case(case_id)
        
        state = case_data["spatial"]["last_seen_state"]
        
        assert state == "Virginia" or state == "VA"


class TestGenerateCases:
    """Test suite for generate_cases.generate_cases function.

    Tests batch case generation with unique ID validation and
    various generation parameters.
    """

    def test_generate_cases_count(self):
        """Test generating multiple cases."""
        num_cases = 5
        cases = generate_cases.generate_cases(num_cases)
        
        assert len(cases) == num_cases
        assert all("case_id" in case for case in cases)

    def test_generate_cases_unique_ids(self):
        """Test that generated cases have unique IDs."""
        num_cases = 10
        cases = generate_cases.generate_cases(num_cases)
        
        case_ids = [case["case_id"] for case in cases]
        
        assert len(case_ids) == len(set(case_ids))

    def test_generate_cases_zero_count(self):
        """Test generating zero cases."""
        cases = generate_cases.generate_cases(0)
        
        assert cases == []

    def test_generate_cases_large_count(self):
        """Test generating large number of cases."""
        num_cases = 100
        cases = generate_cases.generate_cases(num_cases)
        
        assert len(cases) == num_cases


class TestSaveCases:
    """Test save_cases function."""

    def test_save_cases_jsonl(self, tmp_path):
        """Test saving cases to JSONL file."""
        cases = [
            {"case_id": "GRD-001", "demographic": {"age_years": 15}},
            {"case_id": "GRD-002", "demographic": {"age_years": 12}}
        ]
        output_file = tmp_path / "cases.jsonl"
        
        generate_cases.save_cases(cases, str(output_file))
        
        assert output_file.exists()
        
        # Verify file contents
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2

    def test_save_cases_json_format(self, tmp_path):
        """Test saving cases in JSON format."""
        cases = [
            {"case_id": "GRD-001", "demographic": {"age_years": 15}}
        ]
        output_file = tmp_path / "cases.json"
        
        generate_cases.save_cases(cases, str(output_file), format="json")
        
        assert output_file.exists()
        
        # Verify JSON format
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1

    def test_save_cases_empty_list(self, tmp_path):
        """Test saving empty cases list."""
        cases = []
        output_file = tmp_path / "cases.jsonl"
        
        generate_cases.save_cases(cases, str(output_file))
        
        assert output_file.exists()
        
        # Verify file is empty or contains empty array
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            assert content == "" or content == "[]"


class TestLoadTemplate:
    """Test load_template function."""

    def test_load_template_exists(self):
        """Test loading template file that exists."""
        # Assuming template file exists
        try:
            template = generate_cases.load_template()
            assert template is not None
            assert isinstance(template, dict)
        except FileNotFoundError:
            pytest.skip("Template file not found")

    def test_load_template_structure(self):
        """Test that loaded template has correct structure."""
        try:
            template = generate_cases.load_template()
            assert "demographic" in template
            assert "spatial" in template
            assert "temporal" in template
        except FileNotFoundError:
            pytest.skip("Template file not found")


class TestApplyTemplate:
    """Test apply_template function."""

    def test_apply_template_basic(self):
        """Test applying template to case data."""
        template = {
            "demographic": {"age_years": 15, "gender": "M"},
            "spatial": {"last_seen_state": "Virginia"},
            "temporal": {}
        }
        case_data = {"case_id": "GRD-001"}
        
        result = generate_cases.apply_template(case_data, template)
        
        assert result["case_id"] == "GRD-001"
        assert result["demographic"]["age_years"] == 15
        assert result["spatial"]["last_seen_state"] == "Virginia"

    def test_apply_template_merge(self):
        """Test that template merges with existing case data."""
        template = {
            "demographic": {"age_years": 15},
            "spatial": {"last_seen_state": "Virginia"}
        }
        case_data = {
            "case_id": "GRD-001",
            "demographic": {"gender": "M"}
        }
        
        result = generate_cases.apply_template(case_data, template)
        
        assert result["demographic"]["age_years"] == 15
        assert result["demographic"]["gender"] == "M"
        assert result["spatial"]["last_seen_state"] == "Virginia"

    def test_apply_template_empty_template(self):
        """Test applying empty template."""
        template = {}
        case_data = {"case_id": "GRD-001"}
        
        result = generate_cases.apply_template(case_data, template)
        
        assert result["case_id"] == "GRD-001"


class TestMain:
    """Test main function."""

    @patch('generate_cases.generate_cases')
    @patch('generate_cases.save_cases')
    def test_main_basic(self, mock_save, mock_generate):
        """Test main function execution."""
        mock_generate.return_value = [
            {"case_id": "GRD-001"},
            {"case_id": "GRD-002"}
        ]
        
        with patch('generate_cases.argparse.ArgumentParser') as mock_parser:
            mock_args = MagicMock()
            mock_args.num_cases = 2
            mock_args.output = "output.jsonl"
            mock_parser.return_value.parse_args.return_value = mock_args
            
            try:
                generate_cases.main()
            except SystemExit:
                pass  # argparse may call sys.exit
            
            mock_generate.assert_called_once_with(2)
            mock_save.assert_called_once()

    @patch('generate_cases.generate_cases')
    @patch('generate_cases.save_cases')
    def test_main_default_output(self, mock_save, mock_generate):
        """Test main function with default output."""
        mock_generate.return_value = [{"case_id": "GRD-001"}]
        
        with patch('generate_cases.argparse.ArgumentParser') as mock_parser:
            mock_args = MagicMock()
            mock_args.num_cases = 1
            mock_args.output = None
            mock_parser.return_value.parse_args.return_value = mock_args
            
            try:
                generate_cases.main()
            except SystemExit:
                pass
            
            mock_save.assert_called_once()
