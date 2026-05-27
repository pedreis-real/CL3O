'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Tsai-Wu Failure Tests.

Drives TsaiWuFailure on a real solved section and verifies the strength
ratio / margin tensors are well-shaped and internally consistent
(MS = R - 1, global aggregates derived from the per-component fields).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.Constants import N_BOOMS, N_PANELS
from cl3o.fea.post.tsw_failure import TsaiWuFailure, FailureData

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Tsai-Wu tests
# ================================================================================

def test_failure_shapes(runtime) -> None:
    '''Strength-ratio and margin tensors follow (m, nb/ns, 2, nc).'''
    fd = runtime.tsw
    assert isinstance(fd, FailureData)
    m, nc = runtime.mesh.m, runtime.mesh.nc
    assert fd.R_panels.shape == (m, N_PANELS, 2, nc)
    assert fd.R_booms.shape  == (m, N_BOOMS,  2, nc)
    assert fd.MS_panels.shape == fd.R_panels.shape
    assert fd.MS_booms.shape  == fd.R_booms.shape


def test_margin_equals_ratio_minus_one(runtime) -> None:
    '''Global MS_min must equal R_min - 1.'''
    fd = runtime.tsw
    assert np.isclose(fd.MS_min, fd.R_min - 1.0)
    assert np.isfinite(fd.R_min)
    assert fd.nv >= 0


def test_resolve_matches_runtime(static, runtime) -> None:
    '''A fresh Tsai-Wu pass reproduces the cached global minimum.'''
    fd2 = TsaiWuFailure(data=(static, runtime), enable_logging=False).data
    assert np.isclose(fd2.R_min, runtime.tsw.R_min)
    assert fd2.nv == runtime.tsw.nv
