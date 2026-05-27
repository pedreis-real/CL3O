'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Displacement Margins Smoke Tests Module.

Regression tests for DisplacementHelper and DisplacementMargins.

Covered:
  * Flat 6n vector reshape to (6, n) column-major.
  * margin() saturates to _LARGE_MS when peak is zero.
  * Cantilever fixture produces positive MS_disp under a weak tip load.
  * Over-tight u_limit drives MS_disp_u negative and bumps n_violations.

@ CWSS Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path

import numpy as np

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ================ Module imports ================
from fem.solver.mesh_builder import MeshBuilder
from fem.solver.msa_solver   import MSASolver
from fem.post.disp_margins   import (
    DisplacementMargins,
    DisplacementHelper,
    _LARGE_MS,
)


# ================================================================================
# PRIVATE API - Test helper
# ================================================================================

def _solve_cantilever(box_section, cantilever_mesh) -> np.ndarray:
    '''Run MeshBuilder + MSASolver on the fixture; return d vector.'''
    coord, conn, restraints, F_nodal = cantilever_mesh
    fem = MeshBuilder(
        sec_data       = [box_section, box_section],
        coord          = coord,
        conn           = conn,
        restraints     = restraints,
        F_nodal        = F_nodal,
        enable_logging = False,
    ).arrays
    return MSASolver(fem, enable_logging=False).results.d


# ================================================================================
# PUBLIC API - Helper-level tests
# ================================================================================

def test_reshape_dof_matrix_shape() -> None:
    '''Reshape yields (6, n) with column-major DOF-per-node layout.'''
    d = np.arange(18, dtype=float)
    mat = DisplacementHelper.reshape_dof_matrix(d)
    assert mat.shape == (6, 3)
    np.testing.assert_allclose(mat[:, 0], d[0:6])
    np.testing.assert_allclose(mat[:, 1], d[6:12])
    np.testing.assert_allclose(mat[:, 2], d[12:18])


def test_reshape_dof_matrix_rejects_bad_size() -> None:
    '''Vector size not divisible by 6 must raise ValueError.'''
    import pytest
    with pytest.raises(ValueError):
        DisplacementHelper.reshape_dof_matrix(np.zeros(7))


def test_margin_saturates_when_peak_zero() -> None:
    '''Zero peak displacement returns the _LARGE_MS sentinel.'''
    assert DisplacementHelper.margin(50.0, 0.0) == _LARGE_MS


def test_margin_matches_formula() -> None:
    '''MS = limit / peak - 1 for positive peak.'''
    assert np.isclose(DisplacementHelper.margin(50.0, 10.0), 4.0)


# ================================================================================
# PUBLIC API - Full-pipeline tests
# ================================================================================

def test_cantilever_positive_margin(box_section, cantilever_mesh) -> None:
    '''Loose u_limit keeps MS_disp_u positive on the fixture cantilever.'''
    d = _solve_cantilever(box_section, cantilever_mesh)
    data = DisplacementMargins(
        d_vector       = d,
        u_limit        = 100.0,
        th_limit       = 0.10,
        enable_logging = False,
    ).data

    assert data.MS_disp    > 0.0
    assert data.MS_disp_u  > 0.0
    assert data.MS_disp_th > 0.0
    assert data.n_violations == 0


def test_cantilever_tight_limit_flags_violation(
    box_section, cantilever_mesh,
) -> None:
    '''A u_limit below the analytical tip deflection produces MS_u < 0.'''
    d = _solve_cantilever(box_section, cantilever_mesh)
    data = DisplacementMargins(
        d_vector       = d,
        u_limit        = 0.01,
        th_limit       = 0.10,
        enable_logging = False,
    ).data

    assert data.MS_disp_u < 0.0
    assert data.MS_disp   < 0.0
    assert data.n_violations >= 1


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
