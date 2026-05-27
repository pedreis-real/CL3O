'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Tsai-Wu Failure Smoke Tests Module.

Phase E regression tests for TsaiWuHelper and TsaiWuFailure.

Helper-level checks (no FEM required):
  * sigma_1 = Xt           -> FI = 1.0 exactly, MS = 0.0
  * sigma_1 past Xc        -> FI > 1, MS < 0
  * FI <= 0 (unloaded)    -> margin_of_safety returns the _LARGE_MS sentinel

Full-pipeline check:
  * The cantilever fixture with a single-ply carbon/epoxy laminate
    produces no Tsai-Wu violations (displacement margin checked
    separately in test_disp_margins.py).

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
from fem.solver.mesh_builder  import MeshBuilder
from fem.solver.msa_solver    import MSASolver
from fem.post.stress_recovery import StressRecovery
from fem.post.tsw_failure     import TsaiWuHelper, TsaiWuFailure, _LARGE_MS

# ================ Global variables ================
_XT_EXACT = 2000.0         # MPa, must match conftest._XT
_XC_PAST  = -1800.0        # MPa, 20% past conftest._XC = -1500


# ================================================================================
# PRIVATE API - Test helpers
# ================================================================================

def _build_failure_data(
    box_section,
    cantilever_mesh,
    single_ply_bundle,
):
    '''Run the full FEM -> Tsai-Wu pipeline on the fixture cantilever.'''
    coord, conn, restraints, F_nodal = cantilever_mesh
    fem = MeshBuilder(
        sec_data       = [box_section, box_section],
        coord          = coord,
        conn           = conn,
        restraints     = restraints,
        F_nodal        = F_nodal,
        enable_logging = False,
    ).arrays
    res    = MSASolver(fem, enable_logging=False).results
    stress = StressRecovery(
        sec_data       = [box_section, box_section],
        elem_sec_idx   = np.array([0, 1], dtype=int),
        msa_results    = res,
        enable_logging = False,
    ).data

    panels  = [[single_ply_bundle] * 7, [single_ply_bundle] * 7]
    flanges = [[single_ply_bundle] * 4, [single_ply_bundle] * 4]

    return TsaiWuFailure(
        sec_data       = [box_section, box_section],
        elem_sec_idx   = np.array([0, 1], dtype=int),
        stress_data    = stress,
        panel_bundles  = panels,
        flange_bundles = flanges,
        enable_logging = False,
    ).data


# ================================================================================
# PUBLIC API - Helper-level test cases
# ================================================================================

def test_tsw_at_Xt_gives_FI_one(single_ply) -> None:
    '''At sigma_1 = Xt the quadratic form collapses to FI = 1.0.'''
    sigma = np.array([_XT_EXACT, 0.0, 0.0])
    FI    = TsaiWuHelper.tsai_wu_index(single_ply, sigma)
    MS    = TsaiWuHelper.margin_of_safety(FI)

    assert np.isclose(FI, 1.0, atol=1.0e-6)
    assert np.isclose(MS, 0.0, atol=1.0e-6)


def test_tsw_past_Xc_violates(single_ply) -> None:
    '''Beyond compressive strength FI exceeds 1 and MS is negative.'''
    sigma = np.array([_XC_PAST, 0.0, 0.0])
    FI    = TsaiWuHelper.tsai_wu_index(single_ply, sigma)
    MS    = TsaiWuHelper.margin_of_safety(FI)

    assert FI > 1.0
    assert MS < 0.0


def test_margin_saturates_for_non_positive_FI() -> None:
    '''FI <= 0 returns the _LARGE_MS sentinel, flagged as safe.'''
    assert TsaiWuHelper.margin_of_safety( 0.0) == _LARGE_MS
    assert TsaiWuHelper.margin_of_safety(-0.1) == _LARGE_MS


# ================================================================================
# PUBLIC API - Full-pipeline test cases
# ================================================================================

def test_cantilever_has_no_violations(
    box_section, cantilever_mesh, single_ply_bundle,
) -> None:
    '''Weak tip load -> all plies safe and displacement margin positive.'''
    data = _build_failure_data(
        box_section, cantilever_mesh, single_ply_bundle,
    )
    assert data.n_violations == 0


def test_shape_matches_11_components(
    box_section, cantilever_mesh, single_ply_bundle,
) -> None:
    '''FI_ply has shape (n_elements, 2, 11, n_ply_max); here (2, 2, 11, 1).'''
    data = _build_failure_data(
        box_section, cantilever_mesh, single_ply_bundle,
    )
    assert data.FI_ply.shape == (2, 2, 11, 1)


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
