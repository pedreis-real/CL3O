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


def test_local_vs_global_normal_stress(runtime) -> None:
    '''
    use_local selects the beam-local bending constants: the local direct
    stress reproduces N/A + z/I_2 * My + y/I_1 * Mz, and it differs from
    the global-frame stress whenever the section frame is rotated
    (c_rad != 0).
    '''
    sec = runtime.sections.sec_data[0]
    N, MX, MZ = 1.0e3, 5.0e5, 2.0e5

    sig_g = StressRecovery._compute_boom_normal_stress(
        sec, N, MX, MZ, use_local=False)
    sig_l = StressRecovery._compute_boom_normal_stress(
        sec, N, MX, MZ, use_local=True)

    expected = (N / sec.A
                + sec.boom_z / sec.I_2 * MX
                + sec.boom_y / sec.I_1 * MZ)
    np.testing.assert_allclose(sig_l, expected, rtol=1e-12)

    if abs(float(sec.c_rad)) > 1e-6:
        assert np.max(np.abs(sig_g - sig_l)) > 0.0
