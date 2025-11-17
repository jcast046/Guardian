"""Unit tests for metrics.weak module."""

import pytest
from metrics.weak import _norm_cls, CLS_MAP, CLASSES


class TestNormCls:
    """Test suite for metrics.weak._norm_cls function.

    Tests normalization of movement profile classification labels
    with various input formats, case variations, and edge cases.
    """

    def test_norm_stationary(self):
        """Test normalizing stationary movement profile."""
        assert _norm_cls("stationary") == "stationary"
        assert _norm_cls("Stationary") == "stationary"
        assert _norm_cls("STATIONARY") == "stationary"

    def test_norm_walking(self):
        """Test normalizing walking movement profile."""
        assert _norm_cls("walking") == "walking"
        assert _norm_cls("Walking") == "walking"

    def test_norm_driving(self):
        """Test normalizing driving movement profile."""
        assert _norm_cls("driving") == "driving"
        assert _norm_cls("Driving") == "driving"

    def test_norm_public_transit(self):
        """Test normalizing public transit movement profile."""
        assert _norm_cls("public transit") == "public transit"
        assert _norm_cls("transit") == "public transit"
        assert _norm_cls("bus") == "public transit"
        assert _norm_cls("metro") == "public transit"

    def test_norm_unknown(self):
        """Test normalizing unknown movement profile."""
        assert _norm_cls("unknown") == "unknown"
        assert _norm_cls("") == "unknown"
        assert _norm_cls(None) == "unknown"
        assert _norm_cls("invalid") == "unknown"

    def test_norm_underscore(self):
        """Test normalizing with underscores."""
        assert _norm_cls("public_transit") == "public transit"
        assert _norm_cls("walking_mode") == "unknown"

    def test_norm_whitespace(self):
        """Test normalizing with whitespace."""
        assert _norm_cls("  stationary  ") == "stationary"
        assert _norm_cls(" public transit ") == "public transit"

    def test_norm_case_variations(self):
        """Test normalizing case variations."""
        assert _norm_cls("PUBLIC TRANSIT") == "public transit"
        assert _norm_cls("Public Transit") == "public transit"
        assert _norm_cls("pUbLiC tRaNsIt") == "public transit"

    def test_norm_all_classes(self):
        """Test that all classes in CLASSES are properly normalized."""
        for cls in CLASSES:
            assert _norm_cls(cls) == cls
            assert _norm_cls(cls.upper()) == cls
            assert _norm_cls(cls.lower()) == cls

    def test_norm_mapping(self):
        """Test that CLS_MAP mappings work correctly."""
        for key, value in CLS_MAP.items():
            assert _norm_cls(key) == value
            assert _norm_cls(key.upper()) == value
