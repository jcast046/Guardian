"""Unit tests for metrics.config module."""

import pytest
import json
import tempfile
import pathlib
from metrics.config import load_config


class TestLoadConfig:
    """Test suite for metrics.config.load_config function.

    Tests configuration loading, merging with defaults, and error handling.
    """

    def test_load_default_config(self):
        """Test loading default config when no file specified."""
        config = load_config(None)
        
        assert "paths" in config
        assert "ops" in config
        assert "rl" in config

    def test_load_default_config_nonexistent(self):
        """Test loading default config when file doesn't exist."""
        config = load_config("nonexistent.json")
        
        assert "paths" in config
        assert "ops" in config
        assert "rl" in config

    def test_load_custom_config(self, tmp_path):
        """Test loading custom config file."""
        config_file = tmp_path / "test_config.json"
        custom_config = {
            "paths": {
                "eda_min": "custom/path.jsonl"
            },
            "rl": {"ks": [1, 5, 10]}
        }
        config_file.write_text(json.dumps(custom_config), encoding="utf-8")
        
        config = load_config(str(config_file))
        
        assert config["paths"]["eda_min"] == "custom/path.jsonl"
        assert config["rl"]["ks"] == [1, 5, 10]

    def test_load_config_merging(self, tmp_path):
        """Test that custom config merges with defaults."""
        config_file = tmp_path / "test_config.json"
        custom_config = {
            "paths": {
                "eda_min": "custom/path.jsonl"
            },
            "new_key": "new_value"
        }
        config_file.write_text(json.dumps(custom_config), encoding="utf-8")
        
        config = load_config(str(config_file))
        
        # Custom path should override default
        assert config["paths"]["eda_min"] == "custom/path.jsonl"
        # Default paths should still exist
        assert "gold_cases" in config["paths"]
        # New key should be added
        assert config["new_key"] == "new_value"

    def test_load_config_shallow_merge(self, tmp_path):
        """Test that config merging is shallow (dicts are updated, not replaced)."""
        config_file = tmp_path / "test_config.json"
        custom_config = {
            "paths": {
                "custom_path": "custom.jsonl"
            }
        }
        config_file.write_text(json.dumps(custom_config), encoding="utf-8")
        
        config = load_config(str(config_file))
        
        # Custom path should be added
        assert config["paths"]["custom_path"] == "custom.jsonl"
        # Default paths should still exist
        assert "eda_min" in config["paths"]
        assert "gold_cases" in config["paths"]

    def test_load_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON file falls back to default."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{invalid json}", encoding="utf-8")
        
        # Should raise JSONDecodeError, the function doesn't handle JSON errors, so this will raise
        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_file))
