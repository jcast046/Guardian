from metrics.summarizer import calc_summarizer_metrics
from metrics.config import load_config


def test_summarizer_runs():
    m = calc_summarizer_metrics(load_config())
    assert "metrics" in m


