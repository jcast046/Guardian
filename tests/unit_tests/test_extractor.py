"""Unit tests for metrics.extractor module."""

import pytest
from metrics.extractor import (
    _norm_entities,
    _fold_vehicle,
    _normalize_state,
    _normalize_person_name,
    _normalize_location,
    _extract_persons_from_pred,
    _extract_persons_from_gold,
    _extract_vehicles_from_pred,
    _extract_vehicles_from_gold,
    _extract_locations_from_pred,
    _extract_locations_from_gold,
    _calculate_precision_recall_f1,
    _f1_sets,
)


class TestNormEntities:
    """Test suite for metrics.extractor._norm_entities function.

    Tests entity extraction from various container formats including
    direct entities field, extracted field, and llm field with precedence rules.
    """

    def test_entities_direct(self):
        """Test extracting entities from direct entities field."""
        container = {"entities": {"name": "John"}}
        result = _norm_entities(container)
        
        assert result == {"name": "John"}

    def test_entities_from_extracted(self):
        """Test extracting entities from extracted field."""
        container = {
            "extracted": {
                "entities": {"name": "John"}
            }
        }
        result = _norm_entities(container)
        
        assert result == {"name": "John"}

    def test_entities_from_llm(self):
        """Test extracting entities from llm field."""
        container = {
            "llm": {
                "entities": {"name": "John"}
            }
        }
        result = _norm_entities(container)
        
        assert result == {"name": "John"}

    def test_entities_empty(self):
        """Test extracting entities from empty container."""
        container = {}
        result = _norm_entities(container)
        
        assert result == {}

    def test_entities_precedence(self):
        """Test that direct entities field takes precedence."""
        container = {
            "entities": {"name": "John"},
            "extracted": {"entities": {"name": "Jane"}},
            "llm": {"entities": {"name": "Bob"}}
        }
        result = _norm_entities(container)
        
        assert result == {"name": "John"}


class TestFoldVehicle:
    """Test suite for metrics.extractor._fold_vehicle function.

    Tests vehicle normalization including make/model/color combination,
    F-150 handling, and whitespace normalization.
    """

    def test_vehicle_make_model_color(self):
        """Test vehicle normalization with make, model, color."""
        vehicle = {"make": "Honda", "model": "Civic", "color": "Red"}
        result = _fold_vehicle(vehicle)
        
        assert result == "honda civic red"

    def test_vehicle_make_model_only(self):
        """Test vehicle normalization with make and model only."""
        vehicle = {"make": "Toyota", "model": "Camry"}
        result = _fold_vehicle(vehicle)
        
        assert result == "toyota camry"

    def test_vehicle_make_model_field(self):
        """Test vehicle normalization with make_model field."""
        vehicle = {"make_model": "Honda Civic"}
        result = _fold_vehicle(vehicle)
        
        assert result == "honda civic"

    def test_vehicle_f150_normalization(self):
        """Test F-150 normalization."""
        vehicle = {"make": "Ford", "model": "F-150", "color": "Blue"}
        result = _fold_vehicle(vehicle)
        
        assert "f150" in result
        assert "f-150" not in result

    def test_vehicle_f150_spaced(self):
        """Test F - 150 normalization with spaces."""
        vehicle = {"make": "Ford", "model": "F - 150", "color": "Blue"}
        result = _fold_vehicle(vehicle)
        
        assert "f150" in result

    def test_vehicle_empty(self):
        """Test vehicle normalization with empty fields."""
        vehicle = {"make": "", "model": "", "color": ""}
        result = _fold_vehicle(vehicle)
        
        assert result == ""

    def test_vehicle_extra_spaces(self):
        """Test vehicle normalization with extra spaces."""
        vehicle = {"make": "  Honda  ", "model": "  Civic  ", "color": "  Red  "}
        result = _fold_vehicle(vehicle)
        
        assert result == "honda civic red"


class TestNormalizeState:
    """Test suite for metrics.extractor._normalize_state function.

    Tests state name normalization with various formats and edge cases.
    """

    def test_state_virginia(self):
        """Test Virginia state normalization."""
        assert _normalize_state("Virginia") == "va"
        assert _normalize_state("virginia") == "va"
        assert _normalize_state("VA") == "va"
        assert _normalize_state("va") == "va"

    def test_state_other(self):
        """Test other state normalization."""
        assert _normalize_state("California") == "california"
        assert _normalize_state("NY") == "ny"

    def test_state_none(self):
        """Test state normalization with None."""
        assert _normalize_state(None) == ""

    def test_state_empty(self):
        """Test state normalization with empty string."""
        assert _normalize_state("") == ""

    def test_state_whitespace(self):
        """Test state normalization with whitespace."""
        assert _normalize_state("  VA  ") == "va"


class TestNormalizePersonName:
    """Test suite for metrics.extractor._normalize_person_name function.

    Tests person name normalization with various formats and edge cases.
    """

    def test_name_basic(self):
        """Test basic name normalization."""
        assert _normalize_person_name("John Doe") == "john doe"
        assert _normalize_person_name("JANE SMITH") == "jane smith"

    def test_name_whitespace(self):
        """Test name normalization with whitespace."""
        assert _normalize_person_name("  John Doe  ") == "john doe"

    def test_name_none(self):
        """Test name normalization with None."""
        assert _normalize_person_name(None) == ""

    def test_name_empty(self):
        """Test name normalization with empty string."""
        assert _normalize_person_name("") == ""


class TestNormalizeLocation:
    """Test suite for metrics.extractor._normalize_location function.

    Tests location normalization combining city, county, and state
    with various combinations and edge cases.
    """

    def test_location_full(self):
        """Test location normalization with all components."""
        result = _normalize_location("Richmond", "Richmond County", "Virginia")
        assert result == "richmond richmond county va"

    def test_location_partial(self):
        """Test location normalization with partial components."""
        result = _normalize_location("Richmond", "", "VA")
        assert result == "richmond va"

    def test_location_city_only(self):
        """Test location normalization with city only."""
        result = _normalize_location("Richmond", "", "")
        assert result == "richmond"

    def test_location_empty(self):
        """Test location normalization with all empty."""
        result = _normalize_location("", "", "")
        assert result == ""


class TestExtractPersonsFromPred:
    """Test suite for metrics.extractor._extract_persons_from_pred function.

    Tests person extraction from prediction data including main person
    and persons of interest with various data formats.
    """

    def test_extract_main_person(self):
        """Test extracting main person name."""
        pred_data = {"name": "John Doe"}
        result = _extract_persons_from_pred(pred_data)
        
        assert "john doe" in result

    def test_extract_persons_of_interest(self):
        """Test extracting persons of interest."""
        pred_data = {
            "persons_of_interest": [
                {"name": "Jane Smith"},
                {"name": "Bob Johnson"}
            ]
        }
        result = _extract_persons_from_pred(pred_data)
        
        assert "jane smith" in result
        assert "bob johnson" in result

    def test_extract_both(self):
        """Test extracting both main person and POIs."""
        pred_data = {
            "name": "John Doe",
            "persons_of_interest": [
                {"name": "Jane Smith"}
            ]
        }
        result = _extract_persons_from_pred(pred_data)
        
        assert "john doe" in result
        assert "jane smith" in result

    def test_extract_empty(self):
        """Test extracting from empty data."""
        pred_data = {}
        result = _extract_persons_from_pred(pred_data)
        
        assert result == []

    def test_extract_invalid(self):
        """Test extracting from invalid data."""
        result = _extract_persons_from_pred(None)
        assert result == []
        
        result = _extract_persons_from_pred("not a dict")
        assert result == []


class TestExtractPersonsFromGold:
    """Test suite for metrics.extractor._extract_persons_from_gold function.

    Tests person extraction from gold standard data including
    demographic and narrative_osint fields.
    """

    def test_extract_from_demographic(self):
        """Test extracting from demographic field."""
        gold_case = {
            "demographic": {
                "name": "John Doe"
            }
        }
        result = _extract_persons_from_gold(gold_case)
        
        assert "john doe" in result

    def test_extract_from_narrative_osint(self):
        """Test extracting from narrative_osint field."""
        gold_case = {
            "narrative_osint": {
                "persons_of_interest": [
                    {"name": "Jane Smith"}
                ]
            }
        }
        result = _extract_persons_from_gold(gold_case)
        
        assert "jane smith" in result

    def test_extract_both_sources(self):
        """Test extracting from both sources."""
        gold_case = {
            "demographic": {"name": "John Doe"},
            "narrative_osint": {
                "persons_of_interest": [{"name": "Jane Smith"}]
            }
        }
        result = _extract_persons_from_gold(gold_case)
        
        assert "john doe" in result
        assert "jane smith" in result

    def test_extract_empty(self):
        """Test extracting from empty case."""
        gold_case = {}
        result = _extract_persons_from_gold(gold_case)
        
        assert result == []


class TestExtractVehiclesFromPred:
    """Test suite for metrics.extractor._extract_vehicles_from_pred function.

    Tests vehicle extraction from prediction data with various formats
    including dictionary and string representations.
    """

    def test_extract_vehicle_dict(self):
        """Test extracting vehicle as dictionary."""
        pred_data = {
            "persons_of_interest": [
                {
                    "vehicle": {
                        "make": "Honda",
                        "model": "Civic",
                        "color": "Red"
                    }
                }
            ]
        }
        result = _extract_vehicles_from_pred(pred_data)
        
        assert len(result) == 1
        assert "honda civic red" in result[0]

    def test_extract_vehicle_string(self):
        """Test extracting vehicle as string."""
        pred_data = {
            "persons_of_interest": [
                {"vehicle": "Honda Civic"}
            ]
        }
        result = _extract_vehicles_from_pred(pred_data)
        
        assert "honda civic" in result[0]

    def test_extract_multiple_vehicles(self):
        """Test extracting multiple vehicles."""
        pred_data = {
            "persons_of_interest": [
                {"vehicle": {"make": "Honda", "model": "Civic"}},
                {"vehicle": {"make": "Toyota", "model": "Camry"}}
            ]
        }
        result = _extract_vehicles_from_pred(pred_data)
        
        assert len(result) == 2


class TestExtractVehiclesFromGold:
    """Test suite for metrics.extractor._extract_vehicles_from_gold function.

    Tests vehicle extraction from gold standard data with various formats.
    """

    def test_extract_vehicle_from_gold(self):
        """Test extracting vehicle from gold case."""
        gold_case = {
            "narrative_osint": {
                "persons_of_interest": [
                    {
                        "vehicle": {
                            "make": "Honda",
                            "model": "Civic",
                            "color": "Red"
                        }
                    }
                ]
            }
        }
        result = _extract_vehicles_from_gold(gold_case)
        
        assert len(result) == 1
        assert "honda civic red" in result[0]


class TestExtractLocationsFromPred:
    """Test suite for metrics.extractor._extract_locations_from_pred function.

    Tests location extraction from prediction data with various formats
    including full and partial location information.
    """

    def test_extract_location(self):
        """Test extracting location from predicted data."""
        pred_data = {
            "location": {
                "city": "Richmond",
                "county": "Richmond",
                "state": "VA"
            }
        }
        result = _extract_locations_from_pred(pred_data)
        
        assert len(result) == 1
        assert "richmond richmond va" in result[0]

    def test_extract_location_partial(self):
        """Test extracting location with partial data."""
        pred_data = {
            "location": {
                "city": "Richmond",
                "state": "VA"
            }
        }
        result = _extract_locations_from_pred(pred_data)
        
        assert len(result) == 1


class TestExtractLocationsFromGold:
    """Test suite for metrics.extractor._extract_locations_from_gold function.

    Tests location extraction from gold standard data including
    spatial field with city/county/state information.
    """

    def test_extract_location_from_gold(self):
        """Test extracting location from gold case."""
        gold_case = {
            "spatial": {
                "last_seen_city": "Richmond",
                "last_seen_county": "Richmond",
                "last_seen_state": "Virginia"
            }
        }
        result = _extract_locations_from_gold(gold_case)
        
        assert len(result) == 1
        assert "richmond richmond va" in result[0]


class TestCalculatePrecisionRecallF1:
    """Test suite for metrics.extractor._calculate_precision_recall_f1 function.

    Tests precision, recall, and F1 score calculation with various
    prediction and gold standard combinations including edge cases.
    """

    def test_perfect_match(self):
        """Test precision/recall/F1 with perfect match."""
        pred = ["john", "jane"]
        gold = ["john", "jane"]
        result = _calculate_precision_recall_f1(pred, gold)
        
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_no_match(self):
        """Test precision/recall/F1 with no match."""
        pred = ["john", "jane"]
        gold = ["bob", "alice"]
        result = _calculate_precision_recall_f1(pred, gold)
        
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_partial_match(self):
        """Test precision/recall/F1 with partial match."""
        pred = ["john", "jane", "bob"]
        gold = ["john", "jane", "alice"]
        result = _calculate_precision_recall_f1(pred, gold)
        
        assert result["precision"] == pytest.approx(2/3, abs=0.01)
        assert result["recall"] == pytest.approx(2/3, abs=0.01)
        assert result["f1"] == pytest.approx(2/3, abs=0.01)

    def test_no_predictions(self):
        """Test precision/recall/F1 with no predictions."""
        pred = []
        gold = ["john", "jane"]
        result = _calculate_precision_recall_f1(pred, gold)
        
        assert result["precision"] is None
        assert result["recall"] == 0.0
        assert result["f1"] is None

    def test_no_gold(self):
        """Test precision/recall/F1 with no gold standard."""
        pred = ["john", "jane"]
        gold = []
        result = _calculate_precision_recall_f1(pred, gold)
        
        assert result["precision"] == 0.0
        assert result["recall"] is None
        assert result["f1"] is None

    def test_empty_sets(self):
        """Test precision/recall/F1 with empty sets."""
        pred = []
        gold = []
        result = _calculate_precision_recall_f1(pred, gold)
        
        assert result["precision"] is None
        assert result["recall"] is None
        assert result["f1"] is None

    def test_duplicates(self):
        """Test precision/recall/F1 with duplicates."""
        pred = ["john", "john", "jane"]
        gold = ["john", "jane"]
        result = _calculate_precision_recall_f1(pred, gold)
        
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0


class TestF1Sets:
    """Test suite for metrics.extractor._f1_sets function.

    Tests F1 score calculation using set operations with various
    prediction and gold standard combinations including edge cases.
    """

    def test_perfect_match(self):
        """Test F1 with perfect match."""
        pred = ["john", "jane"]
        gold = ["john", "jane"]
        result = _f1_sets(pred, gold)
        
        assert result == 1.0

    def test_no_match(self):
        """Test F1 with no match."""
        pred = ["john"]
        gold = ["jane"]
        result = _f1_sets(pred, gold)
        
        assert result == 0.0

    def test_partial_match(self):
        """Test F1 with partial match."""
        pred = ["john", "jane"]
        gold = ["john"]
        result = _f1_sets(pred, gold)
        
        assert 0.0 < result < 1.0

    def test_empty_lists(self):
        """Test F1 with empty lists."""
        result = _f1_sets([], [])
        
        assert result == 0.0

    def test_case_insensitive(self):
        """Test F1 is case insensitive."""
        pred = ["John", "Jane"]
        gold = ["john", "jane"]
        result = _f1_sets(pred, gold)
        
        assert result == 1.0
