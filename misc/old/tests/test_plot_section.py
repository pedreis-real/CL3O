'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Cross-Section Plot Tests Module.

Phase I regression tests for results.plot_section:

  - Single-station plot_section renders against the box_section fixture
    (booms + centroid + shear center) and writes a PNG.
  - PlotSectionHelper.boom_marker_sizes maps area 0 -> min_size and the
    largest area -> max_size (linear interpolation).
  - plot_all_sections builds a multi-subplot grid for a list of
    GeomData records.
  - T1 / T3 polylines are drawn only when populated; an empty section
    does not raise.

Notes
-----
GeomData records produced by the current section_builder do populate
T1 / T3. To keep this suite decoupled from the full SectionBuilder
pipeline, synthetic T1/T3 dicts are attached directly. Once
SectionBuilder output is stable in the test fixtures, the synthetic
attachment can be replaced with a real GeomData instance.

@ CWSS Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path

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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ================ Module imports ================
from geometry.geom_properties import GeomData
from results.plot_section     import (
    plot_section, plot_all_sections, PlotSectionHelper,
)


# ================================================================================
# PRIVATE API - Section augmentation helpers
# ================================================================================

def _attach_synthetic_outline(gd: GeomData) -> GeomData:
    '''Attach a 100x100 mm box outline (T1, T3, P_XZ) to gd.'''
    half = 50.0
    upper = {"x": np.array([-half, +half]), "z": np.array([+half, +half])}
    lower = {"x": np.array([-half, +half]), "z": np.array([-half, -half])}
    gd.T1 = [upper, lower]
    gd.T3 = [
        {"label": "cell",
         "x": np.array([-half, +half, +half, -half]),
         "z": np.array([+half, +half, -half, -half])},
    ]
    gd.P_XZ = np.array([
        [-half, +half],
        [+half, +half],
        [+half,   0.0],
        [+half, -half],
        [-half, -half],
        [-half,   0.0],
    ])
    return gd


# ================================================================================
# PUBLIC API - Helper unit tests
# ================================================================================

def test_boom_marker_sizes_linear_scaling() -> None:
    '''Area 0 maps to min_size and max(A) maps to max_size.'''
    A = np.array([0.0, 50.0, 100.0, 25.0])
    sizes = PlotSectionHelper.boom_marker_sizes(A, min_size=20.0, max_size=100.0)

    assert np.isclose(sizes[0], 20.0)
    assert np.isclose(sizes[2], 100.0)
    # Midpoint of area range maps to midpoint of size range
    assert np.isclose(sizes[1], 60.0)


def test_boom_marker_sizes_handles_zero_array() -> None:
    '''An all-zero area array returns min_size for every boom.'''
    sizes = PlotSectionHelper.boom_marker_sizes(
        np.zeros(4), min_size=10.0, max_size=80.0,
    )
    assert np.allclose(sizes, 10.0)


# ================================================================================
# PUBLIC API - Section rendering tests
# ================================================================================

def test_plot_section_single_station(
    tmp_path   : Path,
    box_section: GeomData,
) -> None:
    '''Single-station section renders with booms + centroid + SC.'''
    gd  = _attach_synthetic_outline(box_section)
    out = tmp_path / "section.png"

    fig = plot_section(
        geom_data      = gd,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    plt.close(fig)


def test_plot_section_tolerates_empty_outline(
    tmp_path   : Path,
    box_section: GeomData,
) -> None:
    '''With no T1 or T3 entries, booms + centre markers still render.'''
    gd = box_section
    gd.T1 = []
    gd.T3 = []
    out = tmp_path / "section_empty.png"

    fig = plot_section(
        geom_data      = gd,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    plt.close(fig)


def test_plot_all_sections_grid(
    tmp_path   : Path,
    box_section: GeomData,
) -> None:
    '''A 3-station list produces the expected Figure with >= 3 Axes.'''
    sections = []
    for Y in (0.0, 2000.0, 6000.0):
        # Build a fresh copy to avoid shared-reference mutation.
        import copy
        gd = copy.deepcopy(box_section)
        gd.Y_sta = float(Y)
        sections.append(_attach_synthetic_outline(gd))

    out = tmp_path / "all_sections.png"
    fig = plot_all_sections(
        sec_data       = sections,
        n_cols         = 2,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_plot_all_sections_rejects_empty_list() -> None:
    '''An empty section list must raise ValueError with CWSS tag.'''
    with pytest.raises(ValueError):
        plot_all_sections(
            sec_data       = [],
            show           = False,
            enable_logging = False,
        )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    # pytest.main([__file__, "-q"])
    import copy
    import sys as _sys
    _sys.path.insert(0, str(_HERE))
    from misc.old.tests.conftest import _make_box_section
    gd = _attach_synthetic_outline(_make_box_section())
    plot_section(geom_data=gd, enable_logging=False)
    sections = []
    for Y in (0.0, 2000.0, 6000.0):
        s = copy.deepcopy(_make_box_section())
        s.Y_sta = float(Y)
        sections.append(_attach_synthetic_outline(s))
    plot_all_sections(sec_data=sections, n_cols=2, enable_logging=False)
    plt.show()
