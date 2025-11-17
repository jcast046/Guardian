"""Unit tests for guardian_llm.extractor module."""

import pytest
from guardian_llm.extractor import (
    _abbr_state,
    _norm_gender_short,
    minimal_entities_from_case,
    scaffold_from_narrative
)


class TestAbbrState:
    """Test suite for guardian_llm.extractor._abbr_state function.

    Tests state abbreviation conversion with various formats and edge cases.
    """

    def test_abbr_virginia(self):
        """Test Virginia state abbreviation."""
        assert _abbr_state("Virginia") == "VA"
        assert _abbr_state("va") == "VA"
        assert _abbr_state("Va") == "VA"

    def test_abbr_other_state(self):
        """Test other state abbreviation (should return as-is)."""
        assert _abbr_state("California") == "California"
        assert _abbr_state("NY") == "NY"

    def test_abbr_none(self):
        """Test state abbreviation with None."""
        assert _abbr_state(None) is None

    def test_abbr_empty(self):
        """Test state abbreviation with empty string."""
        assert _abbr_state("") == ""

    def test_abbr_whitespace(self):
        """Test state abbreviation with whitespace."""
        assert _abbr_state("  Virginia  ") == "VA"
        assert _abbr_state("  va  ") == "VA"


class TestNormGenderShort:
    """Test suite for guardian_llm.extractor._norm_gender_short function.

    Tests gender normalization to short codes (F/M) with various
    input formats and edge cases.
    """

    def test_gender_female(self):
        """Test female gender normalization."""
        assert _norm_gender_short("female") == "F"
        assert _norm_gender_short("Female") == "F"
        assert _norm_gender_short("FEMALE") == "F"
        assert _norm_gender_short("f") == "F"
        assert _norm_gender_short("F") == "F"

    def test_gender_male(self):
        """Test male gender normalization."""
        assert _norm_gender_short("male") == "M"
        assert _norm_gender_short("Male") == "M"
        assert _norm_gender_short("MALE") == "M"
        assert _norm_gender_short("m") == "M"
        assert _norm_gender_short("M") == "M"

    def test_gender_unclear(self):
        """Test gender normalization for unclear values."""
        assert _norm_gender_short("other") is None
        assert _norm_gender_short("unknown") is None
        assert _norm_gender_short("") is None
        assert _norm_gender_short(None) is None

    def test_gender_whitespace(self):
        """Test gender normalization with whitespace."""
        assert _norm_gender_short("  female  ") == "F"
        assert _norm_gender_short("  male  ") == "M"


class TestMinimalEntitiesFromCase:
    """Test suite for guardian_llm.extractor.minimal_entities_from_case function.

    Tests extraction of minimal entity set from case data including
    age, gender, location, coordinates, and date information.
    """

    def test_extract_minimal_entities(self, sample_gold_case):
        """Test extracting minimal entities from case."""
        result = minimal_entities_from_case(sample_gold_case)
        
        assert "age" in result
        assert "gender" in result
        assert "location" in result
        assert "lat" in result
        assert "lon" in result
        assert "date_reported" in result

    def test_extract_age(self, sample_gold_case):
        """Test extracting age from case."""
        result = minimal_entities_from_case(sample_gold_case)
        
        assert result["age"] == 15

    def test_extract_gender(self, sample_gold_case):
        """Test extracting gender from case."""
        result = minimal_entities_from_case(sample_gold_case)
        
        assert result["gender"] == "M"

    def test_extract_location(self, sample_gold_case):
        """Test extracting location from case."""
        result = minimal_entities_from_case(sample_gold_case)
        
        assert "location" in result
        assert result["location"]["city"] == "Richmond"
        assert result["location"]["county"] == "Richmond"
        assert result["location"]["state"] == "VA"

    def test_extract_coordinates(self, sample_gold_case):
        """Test extracting coordinates from case."""
        result = minimal_entities_from_case(sample_gold_case)
        
        assert result["lat"] == 38.88
        assert result["lon"] == -77.1

    def test_extract_date_reported(self, sample_gold_case):
        """Test extracting date_reported from case."""
        result = minimal_entities_from_case(sample_gold_case)
        
        assert result["date_reported"] == "2025-01-15T10:00:00Z"

    def test_extract_missing_fields(self):
        """Test extracting entities with missing fields."""
        case = {
            "demographic": {},
            "spatial": {},
            "temporal": {}
        }
        result = minimal_entities_from_case(case)
        
        assert result["age"] is None
        assert result["gender"] is None
        assert result["lat"] is None
        assert result["lon"] is None

    def test_extract_location_fallback(self):
        """Test location extraction with fallback fields."""
        case = {
            "demographic": {"age_years": 15},
            "spatial": {
                "last_seen_location": "Richmond, VA",
                "last_seen_state": "Virginia"
            },
            "temporal": {}
        }
        result = minimal_entities_from_case(case)
        
        assert result["location"]["state"] == "VA"
        # Should use last_seen_location as fallback for city
        assert result["location"]["city"] is not None or result["location"]["city"] == "Richmond, VA"


class TestScaffoldFromNarrative:
    """Test suite for guardian_llm.extractor.scaffold_from_narrative function.

    Tests entity extraction from narrative text including name, age, gender,
    location, coordinates, dates, movement cues, and risk factors.
    """

    def test_scaffold_extract_name_age_gender(self):
        """Test extracting name, age, and gender from narrative."""
        narrative = "Missing Person: Child_1234, 15-year-old female"
        result = scaffold_from_narrative(narrative)
        
        assert result["name"] == "Child_1234"
        assert result["age"] == 15
        assert result["gender"] == "F"

    def test_scaffold_extract_location(self):
        """Test extracting location from narrative."""
        narrative = "Last Seen: Richmond, Richmond County, Virginia"
        result = scaffold_from_narrative(narrative)
        
        assert result["location"]["city"] == "Richmond"
        assert result["location"]["county"] == "Richmond County"
        assert result["location"]["state"] == "VA"

    def test_scaffold_extract_coordinates(self):
        """Test extracting coordinates from narrative."""
        narrative = "Coordinates: 37.5407, -77.4360"
        result = scaffold_from_narrative(narrative)
        
        assert result["lat"] == 37.5407
        assert result["lon"] == -77.4360

    def test_scaffold_extract_date_reported(self):
        """Test extracting date_reported from narrative."""
        narrative = "Reported Missing: 2025-01-15T10:00:00Z"
        result = scaffold_from_narrative(narrative)
        
        assert result["date_reported"] == "2025-01-15T10:00:00Z"

    def test_scaffold_extract_movement_cues(self):
        """Test extracting movement cues (highways) from narrative."""
        narrative = "Last seen on I-95 and US-29 heading north"
        result = scaffold_from_narrative(narrative)
        
        assert "I-95" in result["movement_cues"]
        assert "US-29" in result["movement_cues"]

    def test_scaffold_extract_risk_factors(self):
        """Test extracting risk factors from narrative."""
        narrative = "Weapon mentioned in report. Abduction suspected."
        result = scaffold_from_narrative(narrative)
        
        assert "weapon_mentioned" in result.get("risk_factors", [])
        assert "abduction_cue" in result.get("risk_factors", [])

    def test_scaffold_empty_narrative(self):
        """Test scaffold extraction from empty narrative."""
        result = scaffold_from_narrative("")
        
        assert result["name"] is None
        assert result["age"] is None
        assert result["gender"] is None
        assert result["location"]["state"] == "VA"

    def test_scaffold_case_insensitive(self):
        """Test that extraction is case insensitive."""
        narrative = "MISSING PERSON: CHILD_1234, 15-YEAR-OLD FEMALE"
        result = scaffold_from_narrative(narrative)
        
        assert result["name"] == "CHILD_1234"
        assert result["age"] == 15
        assert result["gender"] == "F"
