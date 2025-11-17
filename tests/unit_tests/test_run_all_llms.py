"""Unit tests for run_all_llms.py module."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import run_all_llms


class TestFmtMins:
    """Test suite for run_all_llms._fmt_mins function.

    Tests time formatting in minutes/hours format with various durations.
    """

    def test_fmt_mins_zero(self):
        """Test formatting zero minutes."""
        result = run_all_llms._fmt_mins(0)
        
        assert result == "0m"

    def test_fmt_mins_less_than_hour(self):
        """Test formatting minutes less than an hour."""
        result = run_all_llms._fmt_mins(30)
        
        assert result == "30m"

    def test_fmt_mins_exact_hour(self):
        """Test formatting exact hour."""
        result = run_all_llms._fmt_mins(60)
        
        assert result == "1h"

    def test_fmt_mins_hours_and_minutes(self):
        """Test formatting hours and minutes."""
        result = run_all_llms._fmt_mins(90)
        
        assert result == "1h 30m"

    def test_fmt_mins_multiple_hours(self):
        """Test formatting multiple hours."""
        result = run_all_llms._fmt_mins(150)
        
        assert result == "2h 30m"

    def test_fmt_mins_hours_only(self):
        """Test formatting hours only (no minutes)."""
        result = run_all_llms._fmt_mins(120)
        
        assert result == "2h"


class TestFmtDt:
    """Test suite for run_all_llms._fmt_dt function.

    Tests timestamp formatting with various timezones and edge cases.
    """

    def test_fmt_dt_iso_format(self):
        """Test formatting ISO timestamp."""
        ts_iso = "2025-01-15T10:30:00Z"
        date_str, time_str = run_all_llms._fmt_dt(ts_iso)
        
        assert date_str == "2025-01-15"
        assert time_str == "10:30:00"

    def test_fmt_dt_with_timezone(self):
        """Test formatting timestamp with timezone."""
        ts_iso = "2025-01-15T10:30:00-05:00"
        date_str, time_str = run_all_llms._fmt_dt(ts_iso, "America/New_York")
        
        assert date_str == "2025-01-15"
        assert ":" in time_str  # Should contain time component

    def test_fmt_dt_different_timezone(self):
        """Test formatting timestamp with different timezone."""
        ts_iso = "2025-01-15T10:30:00Z"
        date_str, time_str = run_all_llms._fmt_dt(ts_iso, "America/Los_Angeles")
        
        assert date_str == "2025-01-15"
        assert ":" in time_str

    def test_fmt_dt_invalid_format(self):
        """Test formatting invalid timestamp format."""
        ts_iso = "invalid-date"
        
        try:
            date_str, time_str = run_all_llms._fmt_dt(ts_iso)
            assert isinstance(date_str, str)
            assert isinstance(time_str, str)
        except (ValueError, TypeError):
            pass


class TestAbbrState:
    """Test suite for run_all_llms._abbr_state function.

    Tests state abbreviation conversion with various formats.
    """

    def test_abbr_state_virginia(self):
        """Test Virginia state abbreviation."""
        assert run_all_llms._abbr_state("Virginia") == "VA"
        assert run_all_llms._abbr_state("va") == "VA"
        assert run_all_llms._abbr_state("Va") == "VA"

    def test_abbr_state_other(self):
        """Test other state abbreviation."""
        assert run_all_llms._abbr_state("California") == "California"
        assert run_all_llms._abbr_state("NY") == "NY"

    def test_abbr_state_none(self):
        """Test state abbreviation with None."""
        assert run_all_llms._abbr_state(None) is None

    def test_abbr_state_empty(self):
        """Test state abbreviation with empty string."""
        assert run_all_llms._abbr_state("") == ""


class TestPick:
    """Test _pick function."""

    def test_pick_first_non_none(self):
        """Test picking first non-None value."""
        result = run_all_llms._pick(None, None, "value", None)
        
        assert result == "value"

    def test_pick_all_none(self):
        """Test picking when all values are None."""
        result = run_all_llms._pick(None, None, None)
        
        assert result is None

    def test_pick_with_default(self):
        """Test picking with default value."""
        result = run_all_llms._pick(None, None, default="default")
        
        assert result == "default"

    def test_pick_first_value(self):
        """Test picking first value when not None."""
        result = run_all_llms._pick("first", "second", "third")
        
        assert result == "first"

    def test_pick_empty_string(self):
        """Test picking with empty string (should be considered as value)."""
        result = run_all_llms._pick("", "value")
        
        assert result == ""  # Empty string is a value, not None


class TestRunLlmAnalysis:
    """Test run_llm_analysis function."""

    @patch('run_all_llms.scaffold_from_narrative')
    @patch('run_all_llms.extract_entities')
    @patch('run_all_llms.summarize')
    @patch('run_all_llms.weak_label')
    def test_run_llm_analysis_basic(self, mock_weak_label, mock_summarize, mock_extract, mock_scaffold):
        """Test basic LLM analysis execution."""
        # Mock the functions
        mock_scaffold.return_value = {
            "name": "Child_1234",
            "age": 15,
            "gender": "F",
            "location": {"city": "Richmond", "county": "Richmond", "state": "VA"},
            "lat": 38.88,
            "lon": -77.1,
            "date_reported": "2025-01-15T10:00:00Z"
        }
        mock_extract.return_value = {
            "entities": {
                "data": {
                    "name": "Child_1234",
                    "location": {"city": "Richmond", "county": "Richmond", "state": "VA"}
                }
            }
        }
        mock_summarize.return_value = {
            "summary": {
                "text": "15-year-old female reported missing in Richmond, VA."
            }
        }
        mock_weak_label.return_value = {
            "labels": {
                "data": {
                    "movement": "driving",
                    "risk": "High"
                }
            }
        }
        
        case = {
            "case_id": "GRD-001",
            "narrative": "Missing Person: Child_1234, 15-year-old female. Last Seen: Richmond, Richmond County, Virginia."
        }
        
        result = run_all_llms.run_llm_analysis(case)
        
        assert "case_id" in result
        assert "llm_results" in result
        assert "scaffold" in result["llm_results"]
        assert "entities" in result["llm_results"]
        assert "summary" in result["llm_results"]
        assert "labels" in result["llm_results"]

    @patch('run_all_llms.scaffold_from_narrative')
    @patch('run_all_llms.extract_entities')
    @patch('run_all_llms.summarize')
    @patch('run_all_llms.weak_label')
    def test_run_llm_analysis_error_handling(self, mock_weak_label, mock_summarize, mock_extract, mock_scaffold):
        """Test LLM analysis error handling."""
        # Mock an error in one of the functions
        mock_extract.side_effect = Exception("Extraction failed")
        
        case = {
            "case_id": "GRD-001",
            "narrative": "Test narrative"
        }
        
        # Should handle error gracefully
        result = run_all_llms.run_llm_analysis(case)
        
        assert "case_id" in result
        assert "llm_results" in result
        # Should still have other results even if one fails
        assert "scaffold" in result["llm_results"] or "error" in result["llm_results"]

    @patch('run_all_llms.scaffold_from_narrative')
    @patch('run_all_llms.extract_entities')
    @patch('run_all_llms.summarize')
    @patch('run_all_llms.weak_label')
    def test_run_llm_analysis_missing_narrative(self, mock_weak_label, mock_summarize, mock_extract, mock_scaffold):
        """Test LLM analysis with missing narrative."""
        case = {
            "case_id": "GRD-001"
        }
        
        result = run_all_llms.run_llm_analysis(case)
        
        assert "case_id" in result
        assert "llm_results" in result


class TestRunLlmAnalysisStageByStage:
    """Test run_llm_analysis_stage_by_stage function."""

    @patch('run_all_llms.scaffold_from_narrative')
    @patch('run_all_llms.extract_entities')
    @patch('run_all_llms.summarize')
    @patch('run_all_llms.weak_label')
    def test_run_llm_analysis_stage_by_stage_basic(self, mock_weak_label, mock_summarize, mock_extract, mock_scaffold):
        """Test stage-by-stage LLM analysis."""
        # Mock the functions
        mock_scaffold.return_value = {
            "name": "Child_1234",
            "age": 15,
            "gender": "F"
        }
        mock_extract.return_value = {"entities": {"data": {}}}
        mock_summarize.return_value = {"summary": {"text": "Test summary"}}
        mock_weak_label.return_value = {"labels": {"data": {"movement": "driving"}}}
        
        cases = [
            {
                "case_id": "GRD-001",
                "narrative": "Missing Person: Child_1234, 15-year-old female."
            },
            {
                "case_id": "GRD-002",
                "narrative": "Missing Person: Child_5678, 12-year-old male."
            }
        ]
        
        results = run_all_llms.run_llm_analysis_stage_by_stage(cases)
        
        assert len(results) == 2
        assert all("case_id" in r for r in results)
        assert all("llm_results" in r for r in results)

    @patch('run_all_llms.scaffold_from_narrative')
    @patch('run_all_llms.extract_entities')
    @patch('run_all_llms.summarize')
    @patch('run_all_llms.weak_label')
    def test_run_llm_analysis_stage_by_stage_empty(self, mock_weak_label, mock_summarize, mock_extract, mock_scaffold):
        """Test stage-by-stage LLM analysis with empty cases list."""
        results = run_all_llms.run_llm_analysis_stage_by_stage([])
        
        assert results == []

    @patch('run_all_llms.scaffold_from_narrative')
    @patch('run_all_llms.extract_entities')
    @patch('run_all_llms.summarize')
    @patch('run_all_llms.weak_label')
    def test_run_llm_analysis_stage_by_stage_partial_failure(self, mock_weak_label, mock_summarize, mock_extract, mock_scaffold):
        """Test stage-by-stage LLM analysis with partial failures."""
        # Mock one case to fail
        mock_extract.side_effect = [Exception("Failed"), {"entities": {"data": {}}}]
        mock_scaffold.return_value = {"name": "Test"}
        mock_summarize.return_value = {"summary": {"text": "Test"}}
        mock_weak_label.return_value = {"labels": {"data": {}}}
        
        cases = [
            {"case_id": "GRD-001", "narrative": "Test 1"},
            {"case_id": "GRD-002", "narrative": "Test 2"}
        ]
        
        results = run_all_llms.run_llm_analysis_stage_by_stage(cases)
        
        # Should still process all cases
        assert len(results) == 2
        assert all("case_id" in r for r in results)


class TestMain:
    """Test main function."""

    @patch('run_all_llms.json.load')
    @patch('run_all_llms.json.dump')
    @patch('run_all_llms.run_llm_analysis_stage_by_stage')
    @patch('builtins.open', create=True)
    def test_main_basic(self, mock_open, mock_stage_by_stage, mock_dump, mock_load):
        """Test main function execution."""
        # Mock file operations
        mock_load.return_value = [
            {"case_id": "GRD-001", "narrative": "Test narrative"}
        ]
        mock_stage_by_stage.return_value = [
            {"case_id": "GRD-001", "llm_results": {}}
        ]
        
        # Mock argparse
        with patch('run_all_llms.argparse.ArgumentParser') as mock_parser:
            mock_args = MagicMock()
            mock_args.input = "input.json"
            mock_args.output = "output.json"
            mock_parser.return_value.parse_args.return_value = mock_args
            
            # Should execute without error
            try:
                run_all_llms.main()
            except SystemExit:
                pass  # argparse may call sys.exit

    @patch('run_all_llms.json.load')
    @patch('run_all_llms.run_llm_analysis_stage_by_stage')
    @patch('builtins.open', create=True)
    def test_main_file_not_found(self, mock_open, mock_stage_by_stage, mock_load):
        """Test main function with file not found."""
        mock_open.side_effect = FileNotFoundError("File not found")
        
        with patch('run_all_llms.argparse.ArgumentParser') as mock_parser:
            mock_args = MagicMock()
            mock_args.input = "nonexistent.json"
            mock_args.output = "output.json"
            mock_parser.return_value.parse_args.return_value = mock_args
            
            # Should handle FileNotFoundError
            with pytest.raises(FileNotFoundError):
                run_all_llms.main()
