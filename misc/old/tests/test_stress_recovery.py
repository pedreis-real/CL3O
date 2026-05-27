'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Stress Recovery Smoke Tests Module.

Phase D regression tests for StressRecovery. At the root section of a
cantilever with tip load P in -Z, the bending moment is M = -P * L.
Upper booms (+w) fall in compression and lower booms (-w) in tension,
with magnitude  |sigma| = |M| * |w| / I_1. For the conftest fixture
M = 100000 N*mm, w = 50 mm, I = 1e6 mm^4 giving |sigma| = 5 MPa. The
sign pattern is validated against stress_recovery.py documentation.

@ CWSS Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path
import pytest

import numpy as np

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ================ Module imports ================
from fea.solver.mesh_builder  import MeshBuilder
from fea.solver.static_analysis    import LinearStaticSolver
from fea.post.stress_recovery import StressRecovery
from misc.old.tests.conftest                 import HALF, I_BEAM, L_TOT, TIP_P


# ================================================================================
# PRIVATE API - Test helpers
# ================================================================================

def _recover(box_section, cantilever_mesh):
    '''Run MeshBuilder, MSASolver, and StressRecovery end-to-end.'''
    coord, conn, restraints, F_nodal = cantilever_mesh
    fem = MeshBuilder(
        sec_data       = [box_section, box_section],
        coord          = coord,
        conn           = conn,
        restraints     = restraints,
        F_nodal        = F_nodal,
        enable_logging = False,
    ).arrays
    res = LinearStaticSolver(fem, enable_logging=False).results
    return StressRecovery(
        sections       = [box_section, box_section],
        elem_sec_idx   = np.array([0, 1], dtype=int),
        fea_results    = res,
        enable_logging = False,
    ).data


# ================================================================================
# PUBLIC API - Test cases
# ================================================================================

def test_boom_stress_opposite_signs(
    box_section, cantilever_mesh,
) -> None:
    '''Upper booms (w > 0) and lower booms (w < 0) carry opposite signs.'''
    data       = _recover(box_section, cantilever_mesh)
    sigma_root = data.sigma_booms_begin[0, :]
    upper      = sigma_root[:2]
    lower      = sigma_root[2:]

    assert np.sign(upper[0]) == np.sign(upper[1])
    assert np.sign(lower[0]) == np.sign(lower[1])
    assert np.sign(upper[0]) != np.sign(lower[0])


def test_boom_stress_magnitude_matches_bending(
    box_section, cantilever_mesh,
) -> None:
    '''|sigma| at the root equals |M| * c / I for pure bending.'''
    data       = _recover(box_section, cantilever_mesh)
    sigma_root = data.sigma_booms_begin[0, :]
    M          = abs(TIP_P) * L_TOT
    expected   = M * HALF / I_BEAM

    assert np.allclose(np.abs(sigma_root), expected, atol=5.0e-3)


def test_tip_element_end_has_zero_moment(
    box_section, cantilever_mesh,
) -> None:
    '''At the free tip the bending moment vanishes -> zero boom stress.'''
    data      = _recover(box_section, cantilever_mesh)
    sigma_tip = data.sigma_booms_end[-1, :]
    assert np.max(np.abs(sigma_tip)) < 1.0e-3


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pytest.main([__file__, "-q"])
