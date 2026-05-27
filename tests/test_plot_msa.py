'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
FEA Plot Smoke Tests.

Headless smoke tests for the stiffness-matrix and deformed-mesh plotters
against the real assembled mesh and static-solve results.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import matplotlib.pyplot as plt
import pytest

# ================ Module imports ================
from cl3o.results.plot_msa import (
    plot_K_sparsity, plot_K_spectrum, plot_deformed_mesh,
)

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Plot smoke tests
# ================================================================================

def test_plot_K_sparsity(runtime) -> None:
    '''Spy plot of the global stiffness matrix returns a Figure.'''
    fig = plot_K_sparsity(runtime.mesh.K, show=False, enable_logging=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_K_spectrum(runtime) -> None:
    '''Eigenvalue-spectrum plot returns a Figure.'''
    fig = plot_K_spectrum(runtime.mesh, show=False, enable_logging=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_deformed_mesh(runtime) -> None:
    '''Deformed-mesh plot returns a Figure.'''
    fig = plot_deformed_mesh(
        coord          = runtime.mesh.coord,
        conn           = runtime.mesh.conn,
        results        = runtime.fea_rts,
        show           = False,
        enable_logging = False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
