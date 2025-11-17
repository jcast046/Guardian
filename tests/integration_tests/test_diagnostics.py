from metrics.diagnostics import calc_diagnostics
from metrics.config import load_config


def test_diagnostics_runs():
    m = calc_diagnostics(load_config())
    assert "diagnostics" in m


