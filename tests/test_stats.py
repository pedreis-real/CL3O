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

# ----------------------------------------------------------------------------
# The statistics / visualisation code lives in the standalone tools/stats.py
# script (intentionally not part of the installed cl3o package). Load it by
# file path and expose it under the module path the tests import from. The
# analysis stack (seaborn) is an optional extra, so skip the whole module when
# it is unavailable.
# ----------------------------------------------------------------------------
import sys
import importlib.util

pytest.importorskip("seaborn", reason="stats tools require the [analysis] extra")

_STATS_PATH = Path(__file__).resolve().parents[1] / "tools" / "stats.py"
_spec = importlib.util.spec_from_file_location("cl3o.utils.stats", _STATS_PATH)
_stats_mod = importlib.util.module_from_spec(_spec)
sys.modules["cl3o.utils.stats"] = _stats_mod
_spec.loader.exec_module(_stats_mod)

# The plotting tests assert the figures tools/stats.py actually produces today:
# the real figure names and its <out_dir>/<RUN_NAME>/ save layout (RUN_NAME is a
# module-level constant in tools/stats.py, referenced here as _stats_mod.RUN_NAME).


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


# ================================================================================
# RunStats - LHS figures
# ================================================================================

def _write_lhs_fixtures(tmp_path):
    '''Create a minimal tools_out/results.csv and outputs/ rate.csv tree.'''
    tools_out = tmp_path / "tools"
    sweep_dir = tools_out / "sw"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "results.csv").write_text(
        "sample_idx,NP,CR,F,lambda,k_max_budget,n_gens,converged,"
        "best_f_final,feasible_f,gen_to_half,elapsed_s\n"
        "0,65,0.63,0.88,0.31,400,251,True,35.5,35.5,251,825.5\n"
        "1,73,0.74,1.50,0.01,400,193,True,36.5,36.5,193,926.6\n"
        "2,17,0.78,0.33,0.10,400,232,True,38.1,38.1,155,233.6\n"
    )
    outs = tmp_path / "outs"
    for k in (0, 1, 2):
        d = outs / f"da62_sw_LHS-{k}"
        d.mkdir(parents=True)
        (d / "rate.csv").write_text(
            "k,best_f,rate,conv\n0,55.0,0.0,N\n1,40.0,0.27,N\n2,35.5,0.11,Y\n"
        )
    return tools_out, outs


def test_plot_lhs_writes_pdfs(tmp_path):
    from cl3o.utils.stats import StatsData, RunStats

    tools_out, outs = _write_lhs_fixtures(tmp_path)
    data = StatsData(
        aircraft     = "da62",
        sweep        = "sw",
        tools_out    = tools_out,
        outputs_root = outs,
        out_dir      = tmp_path / "fig",
    )
    RunStats(data, enable_logging=False).plot_lhs()

    out = tmp_path / "fig" / _stats_mod.RUN_NAME
    for name in ("lhs_param_scatter", "lhs_convergence",
                 "lhs_convergence_expand", "lhs_convergence_expand_simple",
                 "lhs_speed_ecdf"):
        assert (out / f"{name}.pdf").is_file(), name


# ================================================================================
# RunStats - sensitivity figures
# ================================================================================

def test_plot_sensitivity_writes_pdfs(tmp_path):
    from cl3o.utils.stats import StatsData, RunStats

    sens = tmp_path / "tools" / "sensitivity"
    sens.mkdir(parents=True)
    (sens / "anova_results.csv").write_text(
        "group,n_valid,mean_f,std_f,min_f,max_f,SS_within,SS_between,eta_sq\n"
        "Mesas,20,795.2,426.5,91.5,1092.3,3638310.3,5000.0,0.101\n"
        "Revestimento,20,910.2,345.2,82.8,1092.2,2382899.7,3000.0,0.017\n"
    )
    (sens / "anova_summary.csv").write_text(
        "grand_mean,SS_total,df_between,df_within,F_stat,p_value\n"
        "991.6,7629585.4,4,95,5.456,0.000535\n"
    )
    data = StatsData(
        aircraft  = "da62",
        sweep     = "sw",
        tools_out = tmp_path / "tools",
        out_dir   = tmp_path / "fig",
    )
    RunStats(data, enable_logging=False).plot_sensitivity()

    out = tmp_path / "fig" / _stats_mod.RUN_NAME
    assert (out / "anova_eta_sq.pdf").is_file()
    assert (out / "anova_group_means.pdf").is_file()


def test_plot_sensitivity_legacy_footer(tmp_path):
    # The legacy producer writes "nan" string cells and a trailing ANOVA
    # summary row whose eta_sq packs "F=.. p=..". plot_sensitivity must drop
    # the footer and coerce the strings rather than render a phantom group.
    from cl3o.utils.stats import StatsData, RunStats

    sens = tmp_path / "tools" / "sensitivity"
    sens.mkdir(parents=True)
    (sens / "anova_results.csv").write_text(
        "group,n_valid,mean_f,std_f,min_f,max_f,SS_within,eta_sq\n"
        "Mesas,20,795.2,426.5,91.5,1092.3,3638310.3,0.101\n"
        "Almas,0,nan,nan,nan,nan,nan,nan\n"
        "ANOVA,100,991.6438,,,,7629585.4,F=5.4562 p=0.0005351\n"
    )
    data = StatsData(
        aircraft  = "da62",
        sweep     = "sw",
        tools_out = tmp_path / "tools",
        out_dir   = tmp_path / "fig",
    )
    RunStats(data, enable_logging=False).plot_sensitivity()

    out = tmp_path / "fig" / _stats_mod.RUN_NAME
    assert (out / "anova_eta_sq.pdf").is_file()
    assert (out / "anova_group_means.pdf").is_file()


# ================================================================================
# RunStats - best-design figures  (uses the heavy `runtime` fixture)
# ================================================================================

@pytest.mark.slow
def test_plot_best_design_writes_pdfs(runtime, tmp_path):
    from cl3o.utils.stats import StatsData, RunStats

    run_dir = tmp_path / "outs" / "da62_sw_LHS-0"
    run_dir.mkdir(parents=True)
    with open(run_dir / "gen_0005.pkl", "wb") as fh:
        pickle.dump(runtime, fh)

    data = StatsData(
        aircraft     = "da62",
        sweep        = "sw",
        run_name     = "da62_sw_LHS-0",
        tools_out    = tmp_path / "tools",
        outputs_root = tmp_path / "outs",
        out_dir      = tmp_path / "fig",
    )
    RunStats(data, enable_logging=False).plot_best_design()

    out = tmp_path / "fig" / _stats_mod.RUN_NAME
    for name in ("design_margins_evolution", "design_margins_dist"):
        assert (out / f"{name}.pdf").is_file(), name


# ================================================================================
# RunStats - run_all degrades gracefully on partial data
# ================================================================================

def test_run_all_partial_data(tmp_path):
    from cl3o.utils.stats import StatsData, RunStats

    # Only LHS results present; ANOVA + pkl absent. run_all must not raise.
    tools_out, outs = _write_lhs_fixtures(tmp_path)
    data = StatsData(
        aircraft     = "da62",
        sweep        = "sw",
        tools_out    = tools_out,
        outputs_root = outs,
        out_dir      = tmp_path / "fig",
    )
    RunStats(data, enable_logging=False).run_all()

    out = tmp_path / "fig" / _stats_mod.RUN_NAME
    assert (out / "lhs_param_scatter.pdf").is_file()
    assert (out / "lhs_speed_ecdf.pdf").is_file()
    # ANOVA inputs absent and best-design is disabled in run_all, so those
    # figures must not be produced.
    assert not (out / "anova_eta_sq.pdf").is_file()
    assert not (out / "design_margins_evolution.pdf").is_file()
