from metrics.rl import calc_rl_metrics
from metrics.config import load_config


def test_rl_runs_baseline():
    m = calc_rl_metrics("baseline", load_config())
    assert "metrics" in m and "geo_hit_at_k" in m["metrics"]


