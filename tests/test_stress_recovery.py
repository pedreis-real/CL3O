'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Stress Recovery Tests.

Verifies StressRecovery packs finite boom-axial and panel-shear stress
tensors of the documented shape at both element ends, and that a fresh
recovery reproduces the cached RuntimeData stresses.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.Constants import N_BOOMS, N_PANELS
from cl3o.fea.post.stress_recovery import StressRecovery, StressData

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Stress recovery tests
# ================================================================================

def test_stress_shapes_and_finite(runtime) -> None:
    '''Boom and panel stress arrays match (m, nb/ns, 2) and are finite.'''
    sd = runtime.stress
    assert isinstance(sd, StressData)
    m, nc = runtime.mesh.m, runtime.mesh.nc

    sigmaA, sigmaB = sd.sigma
    tauA, tauB     = sd.tau
    assert sigmaA.shape == (m, N_BOOMS, nc)
    assert sigmaB.shape == (m, N_BOOMS, nc)
    assert tauA.shape == (m, N_PANELS, nc)
    assert tauB.shape == (m, N_PANELS, nc)

    for arr in (sigmaA, sigmaB, tauA, tauB):
        assert np.all(np.isfinite(arr))


def test_recovery_matches_runtime(runtime) -> None:
    '''Re-running StressRecovery reproduces the cached boom stresses.'''
    sd2 = StressRecovery(
        sections        = runtime.sections,
        element_idx     = runtime.mesh.conn[:, :2],
        fea_results     = runtime.fea_rts,
        use_local_in_sr = True,
        enable_logging  = False,
    ).data
    np.testing.assert_allclose(sd2.sigma[0], runtime.stress.sigma[0],
                               rtol=0.0, atol=0.0)
