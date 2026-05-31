'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Statistics / Visualization Module Tests.

Unit tests for cl3o.utils.stats: StatsData path derivation, StatsHelper
loaders, and headless (Agg) plotting smoke tests.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest


# ================================================================================
# StatsData
# ================================================================================

def test_statsdata_path_derivation(tmp_path):
    from cl3o.utils.stats import StatsData

    d = StatsData(
        aircraft     = "DA62",
        sweep        = "sw",
        tools_out    = tmp_path / "tools",
        out_dir      = tmp_path / "out",
        outputs_root = tmp_path / "outs",
    )
    assert d.aircraft == "da62"
    assert d.lhs_results   == tmp_path / "tools" / "sw" / "results.csv"
    assert d.anova_results == tmp_path / "tools" / "sensitivity" / "anova_results.csv"
    assert d.anova_summary == tmp_path / "tools" / "sensitivity" / "anova_summary.csv"
    assert d.rate_pattern  == "da62_sw_LHS-*"
    assert d.run_name      == "da62_sw_LHS-0"
    assert (tmp_path / "out").is_dir()
