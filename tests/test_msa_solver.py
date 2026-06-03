'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Linear Static Solver Tests.

Drives LinearStaticSolver on the real assembled mesh and verifies the
displacement/reaction containers are well-shaped, that constrained DOFs
stay clamped, and that a fresh solve reproduces the cached results.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.fea.solver.static_analysis import LinearStaticSolver, FeaResults

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Static solve tests
# ================================================================================

def test_solver_result_shapes(static, runtime) -> None:
    '''FeaResults carries well-shaped displacement and reaction tensors.'''
    res = runtime.fea_rts
    assert isinstance(res, FeaResults)

    n, nc = runtime.mesh.n, runtime.mesh.nc
    assert res.dmatrix.shape == (6, n, nc)
    assert res.d.shape == (6 * n, nc)
    assert res.Rmatrix.shape == (6, n, nc)
    assert np.all(np.isfinite(res.dmatrix))


def test_constrained_dofs_clamped(static, runtime) -> None:
    '''DOFs flagged as restrained carry ~zero displacement.'''
    res = runtime.fea_rts
    re_flat = np.asarray(runtime.mesh.re_flat, dtype=bool)
    clamped = res.d[re_flat, :]
    assert np.allclose(clamped, 0.0, atol=1e-9)


def test_resolve_matches_runtime(static, runtime) -> None:
    '''A fresh solve on the same mesh reproduces the cached displacements.'''
    res2 = LinearStaticSolver(
        mesh           = runtime.mesh,
        loads          = static.fem_setup.loads,
        enable_logging = False,
    ).results
    np.testing.assert_allclose(res2.d, runtime.fea_rts.d, rtol=0.0, atol=0.0)
