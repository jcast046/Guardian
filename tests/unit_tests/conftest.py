"""Pytest fixtures for Guardian unit tests.

This module provides shared fixtures for testing individual functions
in isolation across all Guardian modules.
"""

import pytest
import json
import pathlib
import tempfile
from typing import Dict, List, Any


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        "paths": {
            "eda_min": "eda_out/eda_cases_min.jsonl",
            "gold_cases": "gold/cases_gold.jsonl",
            "gold_zones": "gold/zone_truth.jsonl",
            "synthetic_cases": "data/synthetic_cases/",
            "llm_results": "gold/llm_analysis_results.json",
            "zones_baseline": "eda_out/zones_rl.jsonl",
            "zones_llm": "eda_out/zones_reweighted.jsonl",
            "va_boundary": "data/geo/va_boundary.geojson",
        },
        "ops": {
            "llm_timings": "gold/llm_analysis_results.json",
            "validation_report": "eda_out/validation_report.json",
            "expect_outputs": [
                "eda_out/distribution_summary.png",
                "eda_out/age_hist.png",
            ]
        },
        "rl": {"ks": [1, 3, 5, 10]},
        "geo": {"hit_buffer_m": 300}
    }


@pytest.fixture
def sample_gold_case():
    """Sample gold case data for testing."""
    return {
        "case_id": "GRD-2025-000001",
        "demographic": {
            "name": "John Doe",
            "age_years": 15,
            "gender": "male"
        },
        "spatial": {
            "last_seen_lat": 38.88,
            "last_seen_lon": -77.1,
            "last_seen_city": "Richmond",
            "last_seen_county": "Richmond",
            "last_seen_state": "Virginia"
        },
        "temporal": {
            "reported_missing_ts": "2025-01-15T10:00:00Z"
        },
        "narrative_osint": {
            "persons_of_interest": [
                {
                    "name": "Jane Smith",
                    "vehicle": {
                        "make": "Honda",
                        "model": "Civic",
                        "color": "Red"
                    }
                }
            ]
        },
        "movement_profile": "driving"
    }


@pytest.fixture
def sample_predicted_case():
    """Sample predicted case data for testing."""
    return {
        "case_id": "GRD-2025-000001",
        "name": "John Doe",
        "location": {
            "city": "Richmond",
            "county": "Richmond",
            "state": "VA"
        },
        "lat": 38.88,
        "lon": -77.1,
        "date_reported": "2025-01-15T10:00:00Z",
        "persons_of_interest": [
            {
                "name": "Jane Smith",
                "vehicle": {
                    "make": "Honda",
                    "model": "Civic",
                    "color": "Red"
                }
            }
        ],
        "movement": "driving",
        "risk": "High"
    }


@pytest.fixture
def sample_zones():
    """Sample zone data for testing."""
    return [
        {
            "zone_id": "z01",
            "center_lat": 38.88,
            "center_lon": -77.1,
            "radius_miles": 10.0,
            "priority": 0.8,
            "priority_llm": 0.85,
            "score": 0.75
        },
        {
            "zone_id": "z02",
            "center_lat": 38.90,
            "center_lon": -77.15,
            "radius_miles": 15.0,
            "priority": 0.6
        }
    ]


@pytest.fixture
def sample_coordinates():
    """Sample coordinate data for testing."""
    return {
        "richmond": {"lat": 37.5407, "lon": -77.4360},
        "norfolk": {"lat": 36.8468, "lon": -76.2852},
        "alexandria": {"lat": 38.8048, "lon": -77.0469},
        "same_point": {"lat": 38.88, "lon": -77.1}
    }


@pytest.fixture
def sample_geojson_regions():
    """Sample GeoJSON regions data for testing."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-78.0, 38.0],
                        [-77.0, 38.0],
                        [-77.0, 39.0],
                        [-78.0, 39.0],
                        [-78.0, 38.0]
                    ]]
                },
                "properties": {
                    "region_tag": "NoVA"
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-77.5, 36.5],
                        [-76.5, 36.5],
                        [-76.5, 37.5],
                        [-77.5, 37.5],
                        [-77.5, 36.5]
                    ]]
                },
                "properties": {
                    "region_tag": "Tidewater"
                }
            }
        ]
    }


@pytest.fixture
def sample_gazetteer():
    """Sample gazetteer data for testing."""
    return {
        "entries": [
            {
                "name": "Richmond",
                "type": "city",
                "lat": 37.5407,
                "lon": -77.4360,
                "region_tag": "Central Virginia"
            },
            {
                "name": "Norfolk",
                "type": "city",
                "lat": 36.8468,
                "lon": -76.2852,
                "region_tag": "Tidewater"
            },
            {
                "name": "Alexandria",
                "type": "city",
                "lat": 38.8048,
                "lon": -77.0469,
                "region_tag": "Northern Virginia"
            }
        ]
    }


@pytest.fixture
def sample_narrative():
    """Sample narrative text for testing."""
    return """
    Missing Person: John Doe
    Age: 15-year-old male
    Last Seen: Richmond, Richmond, VA
    Reported Missing: 2025-01-15T10:00:00Z
    The child was last seen wearing a blue jacket and jeans.
    """


@pytest.fixture
def sample_entities():
    """Sample entities data for testing."""
    return {
        "entities": {
            "data": {
                "name": "John Doe",
                "location": {
                    "city": "Richmond",
                    "county": "Richmond",
                    "state": "VA"
                },
                "persons_of_interest": [
                    {
                        "name": "Jane Smith",
                        "vehicle": {
                            "make": "Honda",
                            "model": "Civic",
                            "color": "Red"
                        }
                    }
                ]
            }
        }
    }


@pytest.fixture
def sample_llm_results():
    """Sample LLM results for testing."""
    return [
        {
            "case_id": "GRD-2025-000001",
            "llm_results": {
                "summary": {
                    "text": "15-year-old male reported missing in Richmond, VA."
                },
                "entities": {
                    "data": {
                        "name": "John Doe",
                        "location": {
                            "city": "Richmond",
                            "county": "Richmond",
                            "state": "VA"
                        }
                    }
                },
                "labels": {
                    "data": {
                        "movement": "driving",
                        "risk": "High"
                    }
                }
            }
        }
    ]


@pytest.fixture
def sample_transit_data():
    """Sample transit data for testing."""
    return {
        "stations": [
            {
                "id": "station1",
                "name": "Richmond Station",
                "geometry": {
                    "coordinates": [-77.4360, 37.5407]
                },
                "type": "bus_stop",
                "region": "Central Virginia"
            },
            {
                "id": "station2",
                "name": "Norfolk Station",
                "geometry": {
                    "coordinates": [-76.2852, 36.8468]
                },
                "type": "metro_station",
                "region": "Tidewater"
            }
        ]
    }


@pytest.fixture
def sample_reward_config():
    """Sample reward configuration for testing."""
    return {
        "rl_search_config": {
            "time_windows": [
                {"id": "0-24", "start_hr": 0, "end_hr": 24, "weight": 1.0},
                {"id": "24-48", "start_hr": 24, "end_hr": 48, "weight": 0.7}
            ],
            "reward_structures": {
                "zone_level": {
                    "hybrid": {
                        "parameters": {
                            "alpha": 0.7,
                            "beta": 0.3,
                            "inside_threshold": 0.85,
                            "inside_bonus": 0.1,
                            "corridor_bonus": 0.05
                        }
                    },
                    "regularizers": {
                        "radius_penalty": {
                            "parameters": {
                                "lambda_radius": 0.2,
                                "max_radius_miles": 50.0,
                                "p": 2
                            }
                        },
                        "out_of_state": {
                            "penalty_value": -1.0
                        }
                    }
                },
                "window_aggregation": {
                    "penalties": {
                        "overlap_penalty": {
                            "parameters": {
                                "lambda_overlap": 0.1
                            }
                        },
                        "wasted_zone": {
                            "penalty_value": -0.2,
                            "parameters": {
                                "threshold_to_true_by_window": {
                                    "0-24": 15,
                                    "24-48": 25
                                }
                            }
                        }
                    }
                }
            }
        },
        "profiles": {
            "baseline": {
                "weights": {
                    "alpha_orig": 0.6,
                    "beta_plaus": 0.8,
                    "gamma_radius": 0.02,
                    "delta_rl": 0.0
                }
            }
        },
        "default_profile": "baseline"
    }


@pytest.fixture
def sample_hotspots():
    """Sample hotspot data for testing."""
    return [
        (-77.4360, 37.5407, 1.0, 12.43),  # (lon, lat, weight, sigma_miles)
        (-76.2852, 36.8468, 0.8, 10.0)
    ]


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "test_config.json"
    config_data = {
        "paths": {
            "eda_min": "test.jsonl"
        },
        "rl": {"ks": [1, 3, 5]},
        "geo": {"hit_buffer_m": 300}
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f)
    return str(config_path)


@pytest.fixture
def temp_jsonl_file(tmp_path):
    """Create a temporary JSONL file for testing."""
    jsonl_path = tmp_path / "test.jsonl"
    data = [
        {"case_id": "GRD-001", "name": "Test Case 1"},
        {"case_id": "GRD-002", "name": "Test Case 2"}
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return str(jsonl_path)


@pytest.fixture
def temp_reward_config_file(tmp_path, sample_reward_config):
    """Create a temporary reward config file for testing."""
    config_path = tmp_path / "test_reward_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(sample_reward_config, f)
    return str(config_path)


@pytest.fixture
def sample_zone_data():
    """Sample zone data with various formats for testing."""
    return {
        "dict_zones": {
            "0-24": [
                {"zone_id": "z01", "priority": 0.8, "radius_miles": 10.0},
                {"zone_id": "z02", "priority": 0.6, "radius_miles": 15.0}
            ],
            "24-48": [
                {"zone_id": "z03", "priority": 0.7, "radius_miles": 12.0}
            ]
        },
        "list_zones": [
            {"zone_id": "z01", "priority": 0.8, "radius_miles": 10.0},
            {"zone_id": "z02", "priority": 0.6, "radius_miles": 15.0}
        ]
    }


@pytest.fixture
def sample_road_segments():
    """Sample road segments data for testing."""
    return [
        {
            "localNames": ["I-95"],
            "admin": {
                "region": "Northern Virginia"
            }
        },
        {
            "localNames": ["US-29"],
            "admin": {
                "region": "Central Virginia"
            }
        }
    ]


@pytest.fixture
def mock_file_content():
    """Mock file content for testing file I/O."""
    return {
        "jsonl": '{"case_id": "GRD-001"}\n{"case_id": "GRD-002"}\n',
        "json": '{"key": "value"}',
        "pretty_json": '{\n  "key": "value"\n}'
    }
