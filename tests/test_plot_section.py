'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Section Plot Smoke Tests.

Headless smoke tests for the cross-section plotters against a real solved
section set.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import matplotlib.pyplot as plt
import pytest

# ================ Module imports ================
from cl3o.results.plot_section import plot_section, plot_all_sections

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Plot smoke tests
# ================================================================================

def test_plot_section_single(runtime) -> None:
    '''plot_section draws one station and returns a Figure.'''
    geom = runtime.sections.sec_data[0]
    fig = plot_section(geom, show=False, enable_logging=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_all_sections(runtime) -> None:
    '''plot_all_sections tiles every station onto one Figure.'''
    fig = plot_all_sections(
        list(runtime.sections.sec_data), show=False, enable_logging=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
