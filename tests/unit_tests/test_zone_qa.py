"""Unit tests for zone_qa.py module."""

import pytest
import json
import tempfile
import math
from unittest.mock import patch, MagicMock
import zone_qa


class TestRecomputePriority:
    """Test suite for zone_qa.recompute_priority function.

    Tests priority recomputation based on plausibility scores,
    RL scores, risk factors, and reward configuration.
    """

    def test_recompute_priority_basic(self, sample_reward_config):
        """Test basic priority recomputation."""
        zone = {"priority": 0.5, "radius_miles": 3.11}
        qa_result = {"plausibility": 0.7}
        
        result = zone_qa.recompute_priority(zone, qa_result, sample_reward_config)
        
        assert 0.0 <= result <= 1.0

    def test_recompute_priority_high_plausibility(self, sample_reward_config):
        """Test priority recomputation with high plausibility."""
        zone = {"priority": 0.5, "radius_miles": 3.11}
        qa_result = {"plausibility": 0.9}
        
        result = zone_qa.recompute_priority(zone, qa_result, sample_reward_config)
        
        assert result > 0.5

    def test_recompute_priority_low_plausibility(self, sample_reward_config):
        """Test priority recomputation with low plausibility."""
        zone = {"priority": 0.5, "radius_miles": 3.11}
        qa_result = {"plausibility": 0.3}
        
        result = zone_qa.recompute_priority(zone, qa_result, sample_reward_config)
        
        assert 0.0 <= result <= 1.0

    def test_recompute_priority_with_rl_score(self, sample_reward_config):
        """Test priority recomputation with RL score."""
        zone = {"priority": 0.5, "radius_miles": 3.11, "rl_score_norm": 0.8}
        qa_result = {"plausibility": 0.7}
        
        result = zone_qa.recompute_priority(zone, qa_result, sample_reward_config)
        
        assert 0.0 <= result <= 1.0

    def test_recompute_priority_with_risk_boost(self, sample_reward_config):
        """Test priority recomputation with risk boost."""
        zone = {"priority": 0.5, "radius_miles": 3.11, "risk_tier": "high"}
        qa_result = {"plausibility": 0.7}
        
        # Update config to include risk_boost
        sample_reward_config["profiles"]["baseline"]["weights"]["risk_boost"] = 0.1
        
        result = zone_qa.recompute_priority(zone, qa_result, sample_reward_config)
        
        assert 0.0 <= result <= 1.0

    def test_recompute_priority_sigmoid_boundaries(self, sample_reward_config):
        """Test that priority is bounded by sigmoid function."""
        zone = {"priority": 1.0, "radius_miles": 3.11}
        qa_result = {"plausibility": 1.0}
        
        result = zone_qa.recompute_priority(zone, qa_result, sample_reward_config)
        
        assert 0.0 <= result <= 1.0

    def test_recompute_priority_zero_inputs(self, sample_reward_config):
        """Test priority recomputation with zero inputs."""
        zone = {"priority": 0.0, "radius_miles": 3.11}
        qa_result = {"plausibility": 0.0}
        
        result = zone_qa.recompute_priority(zone, qa_result, sample_reward_config)
        
        assert 0.0 <= result <= 1.0


class TestMockLabelCase:
    """Test suite for zone_qa._mock_label_case function.

    Tests mock label case generation from structured case data
    and narrative text for QA evaluation.
    """

    def test_mock_label_case_basic(self):
        """Test basic mock label case functionality."""
        structured_case = {
            "provenance": {
                "search_zones": [
                    {"zone_id": "z01", "type": "school"}
                ]
            }
        }
        narrative = "Child was last seen near a school"
        
        result = zone_qa._mock_label_case(structured_case, narrative)
        
        assert "plausibility" in result
        assert "rationale" in result
        assert 0.0 <= result["plausibility"] <= 1.0

    def test_mock_label_case_no_zones(self):
        """Test mock label case with no zones."""
        structured_case = {"provenance": {"search_zones": []}}
        narrative = "Test narrative"
        
        result = zone_qa._mock_label_case(structured_case, narrative)
        
        assert result["plausibility"] == 0.5
        assert "No zones found" in result["rationale"]

    def test_mock_label_case_school_zone(self):
        """Test mock label case with school zone."""
        structured_case = {
            "provenance": {
                "search_zones": [
                    {"zone_id": "z01", "type": "school"}
                ]
            }
        }
        narrative = "Child was last seen near a school"
        
        result = zone_qa._mock_label_case(structured_case, narrative)
        
        # School zones should have higher plausibility
        assert result["plausibility"] > 0.2

    def test_mock_label_case_residential_zone(self):
        """Test mock label case with residential zone."""
        structured_case = {
            "provenance": {
                "search_zones": [
                    {"zone_id": "z01", "type": "residential"}
                ]
            }
        }
        narrative = "Child was last seen at home"
        
        result = zone_qa._mock_label_case(structured_case, narrative)
        
        # Residential zones should have higher plausibility
        assert result["plausibility"] > 0.2


class TestChooseLabeler:
    """Test _choose_labeler function."""

    def test_choose_mock_labeler_default(self, monkeypatch):
        """Test that mock labeler is chosen by default when real is unavailable."""
        monkeypatch.setattr(zone_qa, "REAL_LABELER", None)
        monkeypatch.setattr(zone_qa, "USE_MOCK_ENV", False)
        
        labeler_fn, source = zone_qa._choose_labeler()
        
        assert source == "mock"
        assert labeler_fn == zone_qa._mock_label_case

    def test_choose_real_labeler_when_available(self, monkeypatch):
        """Test that real labeler is chosen when available."""
        mock_real_labeler = MagicMock()
        monkeypatch.setattr(zone_qa, "REAL_LABELER", mock_real_labeler)
        monkeypatch.setattr(zone_qa, "USE_MOCK_ENV", False)
        
        labeler_fn, source = zone_qa._choose_labeler()
        
        assert source == "real"
        assert labeler_fn == mock_real_labeler

    def test_choose_mock_when_env_set(self, monkeypatch):
        """Test that mock labeler is chosen when USE_MOCK_ENV is set."""
        mock_real_labeler = MagicMock()
        monkeypatch.setattr(zone_qa, "REAL_LABELER", mock_real_labeler)
        monkeypatch.setenv("GUARDIAN_USE_MOCK", "1")
        monkeypatch.setattr(zone_qa, "USE_MOCK_ENV", True)
        
        labeler_fn, source = zone_qa._choose_labeler()
        
        assert source == "mock"

    def test_choose_real_labeler_force_real(self, monkeypatch):
        """Test forcing real labeler raises error if unavailable."""
        monkeypatch.setattr(zone_qa, "REAL_LABELER", None)
        monkeypatch.setattr(zone_qa, "REAL_LABELER_ERR", Exception("Import failed"))
        
        with pytest.raises(RuntimeError):
            zone_qa._choose_labeler(force_real=True)


class TestLoadRewardConfig:
    """Test load_reward_config function."""

    def test_load_reward_config_default_profile(self, tmp_path, sample_reward_config):
        """Test loading reward config with default profile."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        result = zone_qa.load_reward_config(str(config_file))
        
        assert result["__active_profile__"] == "baseline"

    def test_load_reward_config_custom_profile(self, tmp_path, sample_reward_config):
        """Test loading reward config with custom profile."""
        config_file = tmp_path / "reward_config.json"
        config_file.write_text(json.dumps(sample_reward_config), encoding="utf-8")
        
        result = zone_qa.load_reward_config(str(config_file), profile="high_llm")
        
        assert result["__active_profile__"] == "high_llm"

    def test_load_reward_config_missing_file(self):
        """Test loading reward config when file doesn't exist."""
        result = zone_qa.load_reward_config("nonexistent.json")
        
        # Should return default config
        assert "__active_profile__" in result
        assert result["__active_profile__"] == "baseline"


class TestMinmaxNorm:
    """Test _minmax_norm function."""

    def test_minmax_norm_basic(self):
        """Test basic min-max normalization."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = zone_qa._minmax_norm(values)
        
        assert len(result) == len(values)
        assert result[0] == 0.0  # Min value should normalize to 0
        assert result[-1] == 1.0  # Max value should normalize to 1

    def test_minmax_norm_single_value(self):
        """Test min-max normalization with single value."""
        values = [5.0]
        result = zone_qa._minmax_norm(values)
        
        # Single value should normalize to 0.5 (middle)
        assert result[0] == 0.5

    def test_minmax_norm_identical_values(self):
        """Test min-max normalization with identical values."""
        values = [5.0, 5.0, 5.0]
        result = zone_qa._minmax_norm(values)
        
        # All identical values should normalize to 0.5
        assert all(v == 0.5 for v in result)

    def test_minmax_norm_empty(self):
        """Test min-max normalization with empty list."""
        result = zone_qa._minmax_norm([])
        
        assert result == []

    def test_minmax_norm_negative_values(self):
        """Test min-max normalization with negative values."""
        values = [-5.0, -2.0, 0.0, 2.0, 5.0]
        result = zone_qa._minmax_norm(values)
        
        assert len(result) == len(values)
        assert result[0] == 0.0
        assert result[-1] == 1.0
        assert all(0.0 <= v <= 1.0 for v in result)


class TestLoadRlLookup:
    """Test _load_rl_lookup function."""

    def test_load_rl_lookup_basic(self, tmp_path):
        """Test loading RL lookup from JSONL file."""
        jsonl_file = tmp_path / "zones_rl.jsonl"
        jsonl_file.write_text(
            json.dumps({
                "case_id": "GRD-001",
                "zones": {
                    "0-24": [{"zone_id": "z01"}, {"zone_id": "z02"}]
                },
                "zone_scores": {
                    "0-24": [0.8, 0.6]
                }
            }) + "\n",
            encoding="utf-8"
        )
        
        result = zone_qa._load_rl_lookup(jsonl_file)
        
        assert "GRD-001" in result
        assert "z01" in result["GRD-001"]
        assert "z02" in result["GRD-001"]
        assert 0.0 <= result["GRD-001"]["z01"] <= 1.0

    def test_load_rl_lookup_missing_file(self, tmp_path):
        """Test loading RL lookup when file doesn't exist."""
        jsonl_file = tmp_path / "nonexistent.jsonl"
        
        result = zone_qa._load_rl_lookup(jsonl_file)
        
        assert result == {}

    def test_load_rl_lookup_invalid_json(self, tmp_path):
        """Test loading RL lookup with invalid JSON."""
        jsonl_file = tmp_path / "zones_rl.jsonl"
        jsonl_file.write_text("{invalid json}\n", encoding="utf-8")
        
        result = zone_qa._load_rl_lookup(jsonl_file)
        
        # Should handle invalid JSON gracefully
        assert isinstance(result, dict)

    def test_load_rl_lookup_multiple_windows(self, tmp_path):
        """Test loading RL lookup with multiple time windows."""
        jsonl_file = tmp_path / "zones_rl.jsonl"
        jsonl_file.write_text(
            json.dumps({
                "case_id": "GRD-001",
                "zones": {
                    "0-24": [{"zone_id": "z01"}],
                    "24-48": [{"zone_id": "z02"}]
                },
                "zone_scores": {
                    "0-24": [0.8],
                    "24-48": [0.7]
                }
            }) + "\n",
            encoding="utf-8"
        )
        
        result = zone_qa._load_rl_lookup(jsonl_file)
        
        assert "GRD-001" in result
        assert "z01" in result["GRD-001"]
        assert "z02" in result["GRD-001"]


class TestEvaluateGeoHitAtK:
    """Test evaluate_geo_hit_at_k function."""

    def test_evaluate_geo_hit_at_k_hit(self):
        """Test Geo-hit@K evaluation with hit."""
        baseline_zones = [
            {"zone_id": "z01", "priority": 0.8},
            {"zone_id": "z02", "priority": 0.6},
            {"zone_id": "z03", "priority": 0.4}
        ]
        llm_zones = [
            {"zone_id": "z02", "priority_llm": 0.9},
            {"zone_id": "z01", "priority_llm": 0.7},
            {"zone_id": "z03", "priority_llm": 0.5}
        ]
        true_zone_id = "z02"
        k = 3
        
        result = zone_qa.evaluate_geo_hit_at_k(baseline_zones, llm_zones, true_zone_id, k)
        
        assert "baseline_hit" in result
        assert "llm_hit" in result
        assert result["k"] == k
        assert result["true_zone_id"] == true_zone_id

    def test_evaluate_geo_hit_at_k_no_hit(self):
        """Test Geo-hit@K evaluation with no hit."""
        baseline_zones = [
            {"zone_id": "z01", "priority": 0.8},
            {"zone_id": "z02", "priority": 0.6}
        ]
        llm_zones = [
            {"zone_id": "z01", "priority_llm": 0.9},
            {"zone_id": "z02", "priority_llm": 0.7}
        ]
        true_zone_id = "z99"  # Not in top zones
        k = 2
        
        result = zone_qa.evaluate_geo_hit_at_k(baseline_zones, llm_zones, true_zone_id, k)
        
        assert result["baseline_hit"] is False
        assert result["llm_hit"] is False

    def test_evaluate_geo_hit_at_k_different_k(self):
        """Test Geo-hit@K evaluation with different K values."""
        baseline_zones = [
            {"zone_id": "z01", "priority": 0.8},
            {"zone_id": "z02", "priority": 0.6},
            {"zone_id": "z03", "priority": 0.4}
        ]
        llm_zones = [
            {"zone_id": "z03", "priority_llm": 0.9},
            {"zone_id": "z01", "priority_llm": 0.7},
            {"zone_id": "z02", "priority_llm": 0.5}
        ]
        true_zone_id = "z03"
        
        # K=1: z03 should be in LLM top-1, not in baseline top-1
        result_k1 = zone_qa.evaluate_geo_hit_at_k(baseline_zones, llm_zones, true_zone_id, k=1)
        assert result_k1["llm_hit"] is True
        assert result_k1["baseline_hit"] is False
        
        # K=3: z03 should be in both top-3
        result_k3 = zone_qa.evaluate_geo_hit_at_k(baseline_zones, llm_zones, true_zone_id, k=3)
        assert result_k3["llm_hit"] is True
        assert result_k3["baseline_hit"] is True


class TestFormatZoneResults:
    """Test format_zone_results function."""

    def test_format_zone_results_basic(self):
        """Test formatting zone results."""
        zone_data = {
            "case_id": "GRD-001",
            "zones": [
                {
                    "zone_id": "z01",
                    "plausibility": 0.8,
                    "original_priority": 0.5,
                    "new_priority": 0.7,
                    "labeler_source": "mock"
                }
            ]
        }
        
        result = zone_qa.format_zone_results(zone_data)
        
        assert "GRD-001" in result
        assert "z01" in result
        assert "0.800" in result or "0.8" in result
        assert "0.500" in result or "0.5" in result
        assert "0.700" in result or "0.7" in result

    def test_format_zone_results_no_zones(self):
        """Test formatting zone results with no zones."""
        zone_data = {
            "case_id": "GRD-001",
            "zones": []
        }
        
        result = zone_qa.format_zone_results(zone_data)
        
        assert "GRD-001" in result
        assert "No zones found" in result

    def test_format_zone_results_multiple_zones(self):
        """Test formatting zone results with multiple zones."""
        zone_data = {
            "case_id": "GRD-001",
            "zones": [
                {
                    "zone_id": "z01",
                    "plausibility": 0.8,
                    "original_priority": 0.5,
                    "new_priority": 0.7,
                    "labeler_source": "mock"
                },
                {
                    "zone_id": "z02",
                    "plausibility": 0.6,
                    "original_priority": 0.4,
                    "new_priority": 0.5,
                    "labeler_source": "mock"
                }
            ]
        }
        
        result = zone_qa.format_zone_results(zone_data)
        
        assert "z01" in result
        assert "z02" in result
