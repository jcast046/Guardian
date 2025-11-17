from metrics.extractor import calc_extractor_metrics
from metrics.config import load_config


def test_extractor_runs():
    m = calc_extractor_metrics(load_config())
    assert "metrics" in m


