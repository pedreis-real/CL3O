'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Displacement Margins Tests.

Verifies DisplacementMargins turns the global displacement tensor into
well-shaped per-DOF margins, that an over-tight deflection limit drives a
violation, and that a fresh evaluation reproduces the cached margins.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.fea.post.displacement_ms import DisplacementMargins, DisplacementData

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Displacement margin tests
# ================================================================================

def test_margin_shapes(static, runtime) -> None:
    '''MS tensors follow (3, n, nc) and the per-node minimum is (n,).'''
    dd = runtime.displ
    assert isinstance(dd, DisplacementData)
    n, nc = runtime.mesh.n, runtime.mesh.nc
    assert dd.MS_u.shape == (3, n, nc)
    assert dd.MS_th.shape == (3, n, nc)
    assert dd.MS_min_node.shape == (n,)
    assert np.isclose(dd.MS_min, float(np.min(dd.MS_min_node)))


def test_recompute_matches_runtime(static, runtime) -> None:
    '''Re-running on the same dmatrix reproduces the cached margins.'''
    dd2 = DisplacementMargins(
        mesh           = runtime.mesh,
        dmatrix        = runtime.fea_rts.dmatrix,
        b              = static.wing_db.b,
        enable_logging = False,
    ).data
    np.testing.assert_allclose(dd2.MS_min_node, runtime.displ.MS_min_node,
                               rtol=0.0, atol=0.0)


def test_tight_limit_flags_violation(static, runtime) -> None:
    '''A near-zero allowable deflection (tiny b) forces MS < 0 and nv > 0.'''
    dd = DisplacementMargins(
        mesh           = runtime.mesh,
        dmatrix        = runtime.fea_rts.dmatrix,
        b              = 1e-6,
        enable_logging = False,
    ).data
    assert dd.MS_min < 0.0
    assert dd.nv >= 1
