'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Mesh Builder Tests.

Exercises MeshBuilder against a real solved section set: checks the
assembled global stiffness matrix is square, symmetric and consistently
sized, and that a fresh assembly reproduces the cached RuntimeData mesh.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.fea.solver.mesh_builder import MeshBuilder, MeshData

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Assembly tests
# ================================================================================

def test_mesh_dimensions_consistent(runtime) -> None:
    '''Node/element/DOF counts obey the cantilever-chain relations.'''
    mesh = runtime.mesh
    assert isinstance(mesh, MeshData)
    assert mesh.m == mesh.n - 1
    assert mesh.dof == 6 * mesh.n
    assert mesh.coord.shape == (mesh.n, 3)


def test_global_stiffness_square_symmetric(runtime) -> None:
    '''[K] is (dof, dof) and symmetric within numerical tolerance.'''
    K = np.asarray(runtime.mesh.K, dtype=float)
    assert K.shape == (runtime.mesh.dof, runtime.mesh.dof)
    assert np.allclose(K, K.T, rtol=0.0, atol=1e-6 * np.max(np.abs(K)))


def test_reassembly_matches_runtime(static, runtime) -> None:
    '''Re-running MeshBuilder on the same inputs reproduces [K] exactly.'''
    mesh2 = MeshBuilder(
        data           = (static.fem_setup, runtime.sections),
        enable_logging = False,
    ).data
    np.testing.assert_allclose(mesh2.K, runtime.mesh.K, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(mesh2.conn, runtime.mesh.conn)
