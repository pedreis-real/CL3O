'''
================================================================================
CWSS - Composite Wing Structural Sizing.
MSA Solver Smoke Tests Module.

Phase C regression tests for MSASolver. A two-element cantilever with
a tip point load in -Z is solved and compared against the analytical
Euler-Bernoulli tip deflection  u_Z = P * L^3 / (3 E I). With the
conftest fixture (P = -100 N, L = 1000 mm, E = 70 GPa, I = 1e6 mm^4)
the result is -0.47619 mm, which matches the value logged during the
Phase C smoke run within rounding tolerance.

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
from fea.solver.static_analysis   import LinearStaticSolver
from misc.old.tests.conftest                import E_ALU, I_BEAM, L_TOT, TIP_P


# ================================================================================
# PRIVATE API - Test helpers
# ================================================================================

def _solve(box_section, cantilever_mesh):
    '''Run MeshBuilder then MSASolver on the fixture cantilever.'''
    coord, conn, restraints, F_nodal = cantilever_mesh
    fem = MeshBuilder(
        sec_data       = [box_section, box_section],
        coord          = coord,
        conn           = conn,
        restraints     = restraints,
        F_nodal        = F_nodal,
        enable_logging = False,
    ).arrays
    return LinearStaticSolver(fem, enable_logging=False).results


# ================================================================================
# PUBLIC API - Test cases
# ================================================================================

def test_tip_deflection_matches_analytical(
    box_section, cantilever_mesh,
) -> None:
    '''Tip u_Z matches P * L^3 / (3 E I) within rounding tolerance.'''
    res        = _solve(box_section, cantilever_mesh)
    analytical = TIP_P * L_TOT**3 / (3.0 * E_ALU * I_BEAM)
    tip_uZ     = float(res.dmatrix[2, -1])

    assert np.isclose(tip_uZ, analytical, atol=5.0e-4), (
        f"tip u_Z = {tip_uZ} vs analytical {analytical}"
    )


def test_root_reactions_balance_tip_load(
    box_section, cantilever_mesh,
) -> None:
    '''Z-reaction at the root equals -TIP_P (global equilibrium).'''
    res      = _solve(box_section, cantilever_mesh)
    R_Z_root = float(res.reaction_mat[2, 0])
    assert np.isclose(R_Z_root, -TIP_P, atol=1.0e-3)


def test_free_dofs_exclude_root(
    box_section, cantilever_mesh,
) -> None:
    '''The six root-node DOFs (0..5) must all be constrained.'''
    res = _solve(box_section, cantilever_mesh)
    assert np.all(res.f >= 6)


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
