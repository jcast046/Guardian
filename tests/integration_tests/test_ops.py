from metrics.ops import calc_ops_metrics
from metrics.config import load_config


def test_ops_runs():
    m = calc_ops_metrics(load_config())
    assert "metrics" in m and "timestamp" in m


