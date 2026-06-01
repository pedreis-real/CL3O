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


# ================================================================================
# StatsHelper
# ================================================================================

def test_spearman_matrix_monotonic():
    from cl3o.utils.stats import StatsHelper

    df  = pd.DataFrame({"NP": [1, 2, 3, 4], "best_f_final": [4, 3, 2, 1]})
    rho = StatsHelper.spearman_matrix(df, ["NP"], ["best_f_final"])
    assert rho.loc["NP", "best_f_final"] == pytest.approx(-1.0)


def test_load_rate_curves(tmp_path):
    from cl3o.utils.stats import StatsHelper

    for k in (0, 2):
        d = tmp_path / f"da62_sw_LHS-{k}"
        d.mkdir()
        (d / "rate.csv").write_text(
            "k,best_f,rate,conv\n0,10.0,0.0,N\n1,9.0,0.1,Y\n"
        )
    curves = StatsHelper.load_rate_curves(tmp_path, "da62_sw_LHS-*")
    assert set(curves) == {0, 2}
    assert list(curves[0]["best_f"]) == [10.0, 9.0]


def test_load_last_pkl_picks_highest(tmp_path):
    from cl3o.utils.stats import StatsHelper

    (tmp_path / "gen_0001.pkl").write_bytes(pickle.dumps({"v": 1}))
    (tmp_path / "gen_0009.pkl").write_bytes(pickle.dumps({"v": 9}))
    assert StatsHelper.load_last_pkl(tmp_path)["v"] == 9


def test_load_last_pkl_missing_raises(tmp_path):
    from cl3o.utils.stats import StatsHelper

    with pytest.raises(FileNotFoundError):
        StatsHelper.load_last_pkl(tmp_path)


def test_load_anova_with_and_without_summary(tmp_path):
    from cl3o.utils.stats import StatsHelper

    res = tmp_path / "anova_results.csv"
    res.write_text(
        "group,n_valid,mean_f,std_f,min_f,max_f,SS_within,SS_between,eta_sq\n"
        "A,20,10.0,1.0,9.0,11.0,5.0,2.0,0.3\n"
    )
    summ = tmp_path / "anova_summary.csv"
    summ.write_text(
        "grand_mean,SS_total,df_between,df_within,F_stat,p_value\n"
        "10.0,7.0,4,95,5.4,0.001\n"
    )

    df, summary = StatsHelper.load_anova(res, summ)
    assert list(df["group"]) == ["A"]
    assert summary["F_stat"] == pytest.approx(5.4)
    assert summary["p_value"] == pytest.approx(0.001)

    _, no_summary = StatsHelper.load_anova(res, tmp_path / "missing.csv")
    assert no_summary is None
