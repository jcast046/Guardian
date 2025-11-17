from metrics.weak import calc_weak_labeler_metrics
from metrics.config import load_config


def test_weak_runs():
    m = calc_weak_labeler_metrics(load_config())
    assert "metrics" in m


