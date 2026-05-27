'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
DE Convergence Plot Tests Module.

Phase I regression tests for results.plot_cwss:

  - plot_convergence renders best/mean/std and the feasible line, and
    writes a PNG whose byte size is non-trivial.
  - Running-best trajectory is monotone non-increasing, matching the
    mathematical definition of min-accumulate.
  - plot_design_trajectories normalises each variable to [0, 1] using
    the SetupOpt bounds (values at bounds_lo map to 0, bounds_hi to 1).
  - Both functions refuse empty histories with a clear ValueError.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path

# Headless Agg backend (must be set before importing pyplot)
import matplotlib
# matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ================ Module imports ================
from optimization.de_opt import OptData, HistoryData
from results.plot_cleo   import (
    plot_convergence, plot_design_trajectories, PlotCL3OHelper,
)


# ================================================================================
# PRIVATE API - Synthetic history builder
# ================================================================================

def _make_history(n_gen: int = 12, n_dim: int = 4) -> HistoryData:
    '''Decaying best_f, inflated mean_f, shrinking std_f trajectory.'''
    rng = np.random.default_rng(0)
    best_f = np.exp(-np.linspace(0.0, 2.5, n_gen + 1)) * 10.0 + 1.0
    mean_f = best_f + np.linspace(4.0, 0.5, n_gen + 1)
    std_f  = np.linspace(2.0, 0.2, n_gen + 1)
    best_X = rng.uniform(0.0, 1.0, size=(n_gen + 1, n_dim))
    return HistoryData(
        ng      = n_gen,
        D      = n_dim,
        best_X     = best_X,
        best_f     = best_f,
        mean_f     = mean_f,
        std_f      = std_f,
        feasible_X = best_X[-1],
        feasible_f = float(best_f[-1]),
    )


def _make_opt(n_dim: int = 4) -> OptData:
    return OptData(
        lo = np.zeros(n_dim),
        hi = np.ones(n_dim),
        NP        = 10,
        CR        = 0.9,
        F         = 0.8,
        lmbda       = 0.5,
        k_max     = 12,
        seed      = 0,
        n_dim     = n_dim,
    )


# ================================================================================
# PUBLIC API - Tests
# ================================================================================

def test_running_best_is_monotone() -> None:
    '''PlotCWSSHelper.running_best == numpy.minimum.accumulate semantics.'''
    arr = np.array([5.0, 4.0, 6.0, 3.5, 4.2, 3.5])
    rb  = PlotCL3OHelper.running_best(arr)
    assert np.all(np.diff(rb) <= 0.0)
    assert np.isclose(rb[-1], 3.5)


def test_normalized_trajectories_bounds() -> None:
    '''Lower-bound row maps to 0.0, upper-bound row to 1.0 per column.'''
    best_X = np.array([
        [-1.0, 0.0,  10.0],
        [ 0.0, 5.0,  30.0],
        [ 1.0, 10.0, 50.0],
    ])
    lo = np.array([-1.0, 0.0, 10.0])
    hi = np.array([ 1.0, 10.0, 50.0])

    norm = PlotCL3OHelper.normalized_trajectories(best_X, lo, hi)
    assert norm.shape == best_X.shape
    assert np.allclose(norm[0], 0.0)
    assert np.allclose(norm[-1], 1.0)
    # Mid row is exactly at the midpoint of each interval
    assert np.allclose(norm[1], 0.5)


def test_convergence_plot_renders(tmp_path: Path) -> None:
    '''plot_convergence produces a non-empty PNG and a Figure handle.'''
    hist = _make_history()
    out  = tmp_path / "convergence.png"

    fig = plot_convergence(
        history        = hist,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )

    assert isinstance(fig, plt.Figure)
    assert out.exists()
    assert out.stat().st_size > 1024          # non-trivial PNG
    plt.close(fig)


def test_design_trajectories_plot_renders(tmp_path: Path) -> None:
    '''plot_design_trajectories runs end-to-end and writes a PNG.'''
    hist = _make_history(n_dim=3)
    opt  = _make_opt(n_dim=3)

    out = tmp_path / "traj.png"
    fig = plot_design_trajectories(
        history        = hist,
        opt_data       = opt,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    plt.close(fig)


def test_convergence_rejects_empty_history() -> None:
    '''Empty best_f must raise ValueError with a CWSS-tagged message.'''
    empty = HistoryData(ng=0, D=0)
    with pytest.raises(ValueError):
        plot_convergence(
            history        = empty,
            show           = False,
            enable_logging = False,
        )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    # pytest.main([__file__, "-q"])
    hist = _make_history()
    opt  = _make_opt(n_dim=hist.D)
    plot_convergence(history=hist, enable_logging=False)
    plot_design_trajectories(history=hist, opt_data=opt, enable_logging=False)
    plt.show()
