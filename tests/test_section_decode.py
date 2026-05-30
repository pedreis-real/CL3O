'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Section Builder Decoder Tests.

The cross-section decoder (SectionBuilder._cp_index / _interp_opt_vars)
must be side-agnostic: the design vector is defined over |Y| (Y_cp is
always stored positive in lerp_wing_db), so a left-wing station at -Y and
a right-wing station at +Y of equal magnitude must decode to identical
spar fractions, flange widths, layups and control-point index.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from types import SimpleNamespace

import numpy as np

# ================ Module imports ================
from cl3o.geometry.section_builder import SectionBuilder


# ================================================================================
# PRIVATE API - Lightweight stubs
# ================================================================================

def _make_decoder() -> SectionBuilder:
    '''
    Build a SectionBuilder without running its DB-heavy __init__, wiring
    only the attributes the decoder helpers read: a positive Y_cp and a
    set of per-cpt design variables that taper from root to tip.
    '''
    sb = object.__new__(SectionBuilder)

    Y_cp = np.array([0.0, 600.0, 2000.0, 6400.0])  # always positive (|Y|)
    sb.st = SimpleNamespace(lerp_wing_db=SimpleNamespace(Y_cp=Y_cp))

    # Continuous vars vary root -> tip so a clamp-to-root bug is visible.
    sb.opt = SimpleNamespace(
        xw1=np.array([0.20, 0.24, 0.30, 0.40]),
        xw2=np.array([0.55, 0.58, 0.62, 0.70]),
        bf1=np.array([0.02, 0.03, 0.04, 0.05]),
        bf2=np.array([0.02, 0.03, 0.04, 0.05]),
        bf3=np.array([0.02, 0.03, 0.04, 0.05]),
        bf4=np.array([0.02, 0.03, 0.04, 0.05]),
        ls1=np.array([1, 2, 3, 4]),
        ls2=np.array([1, 2, 3, 4]),
        lw1=np.array([1, 2, 3, 4]),
        lw2=np.array([1, 2, 3, 4]),
        lf1=np.array([1, 2, 3, 4]),
        lf2=np.array([1, 2, 3, 4]),
        lf3=np.array([1, 2, 3, 4]),
        lf4=np.array([1, 2, 3, 4]),
    )
    return sb


# ================================================================================
# PUBLIC API - Decoder symmetry tests
# ================================================================================

def test_cp_index_is_side_agnostic() -> None:
    '''cp_index at -Y (left) must equal cp_index at +Y (right).'''
    sb = _make_decoder()
    for y in (155.7, 764.2, 2452.1, 6345.0):
        assert sb._cp_index(+y) == sb._cp_index(-y)


def test_interp_opt_vars_is_side_agnostic() -> None:
    '''Continuous design vars must decode identically at -Y and +Y.'''
    sb = _make_decoder()
    for y in (155.7, 764.2, 2452.1, 6345.0):
        right = sb._interp_opt_vars(+y)
        left = sb._interp_opt_vars(-y)
        assert right == left


def test_step_opt_vars_is_side_agnostic() -> None:
    '''Discrete layup indices must decode identically at -Y and +Y.'''
    sb = _make_decoder()
    for y in (155.7, 764.2, 2452.1, 6345.0):
        assert sb._step_opt_vars(+y) == sb._step_opt_vars(-y)


def test_left_wing_tapers_off_root() -> None:
    '''Left-wing spar fractions must not be frozen at the root value.'''
    sb = _make_decoder()
    root = sb._interp_opt_vars(-155.7)['xw1']
    tip = sb._interp_opt_vars(-6345.0)['xw1']
    assert tip > root
