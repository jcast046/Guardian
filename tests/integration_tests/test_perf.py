from metrics.perf import calc_perf_metrics
from metrics.config import load_config


def test_perf_runs():
    m = calc_perf_metrics(load_config())
    assert "metrics" in m


