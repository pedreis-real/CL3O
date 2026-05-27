'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Mesh Builder Smoke Tests Module.

Phase B regression tests for MeshBuilder. Verifies that tip nodal
loads are stored in FemArraysData.F_nodal and do NOT leak into
FemArraysData.Pf (fixed-end reactions), because MSASolver assembles
the free-DOF right-hand-side as  rhs = F_free - Pf_free. Merging the
two vectors would flip the sign of the applied load.

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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ================ Module imports ================
from fea.solver.mesh_builder import MeshBuilder
from misc.old.tests.conftest                import TIP_P

# ================ Global variables ================
_TIP_DOF_Z = 14       # node 2 (tip) * 6 + component 2 (u_Z)


# ================================================================================
# PRIVATE API - Test helpers
# ================================================================================

def _build(box_section, cantilever_mesh):
    '''Assemble FemArraysData for the fixture cantilever.'''
    coord, conn, restraints, F_nodal = cantilever_mesh
    return MeshBuilder(
        sec_data       = [box_section, box_section],
        coord          = coord,
        conn           = conn,
        restraints     = restraints,
        F_nodal        = F_nodal,
        enable_logging = False,
    ).arrays


# ================================================================================
# PUBLIC API - Test cases
# ================================================================================

def test_F_nodal_stored_on_tip_dof(box_section, cantilever_mesh) -> None:
    '''F_nodal[14] holds the tip u_Z load; no other entry is populated.'''
    arrays = _build(box_section, cantilever_mesh)

    assert arrays.F.shape == (18,)
    assert np.isclose(arrays.F[_TIP_DOF_Z], TIP_P)

    other = np.delete(arrays.F, _TIP_DOF_Z)
    assert np.count_nonzero(other) == 0


def test_Pf_does_not_absorb_nodal_load(
    box_section, cantilever_mesh,
) -> None:
    '''Pf holds fixed-end reactions only; no distributed load -> all zero.'''
    arrays = _build(box_section, cantilever_mesh)
    assert np.count_nonzero(arrays.Pf) == 0


def test_K_symmetry(box_section, cantilever_mesh) -> None:
    '''Global stiffness matrix must be symmetric.'''
    arrays = _build(box_section, cantilever_mesh)
    assert np.allclose(arrays.K, arrays.K.T, atol=1.0e-6)


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
