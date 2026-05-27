'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Convergence Plot Smoke Tests.

Headless smoke tests for the DE convergence and design-trajectory
plotters against a real HistoryData from a tiny optimization run.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import matplotlib.pyplot as plt
import pytest

# ================ Module imports ================
from cl3o.results.plot_cleo import plot_convergence, plot_design_trajectories

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Plot smoke tests
# ================================================================================

def test_plot_convergence(de_history) -> None:
    '''Convergence plotter returns a Figure.'''
    fig = plot_convergence(de_history, show=False, enable_logging=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_design_trajectories(de_history, opt_data) -> None:
    '''Design-trajectory plotter returns a Figure.'''
    fig = plot_design_trajectories(
        de_history, opt_data, show=False, enable_logging=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
