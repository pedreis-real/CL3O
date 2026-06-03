'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Optimization Layer Tests.

Covers the pure-math pieces of the DE optimizer that do not need the FEA
pipeline (bound construction, input validators, the logistic penalty)
plus a structural-mass check driven against a real solved RuntimeData.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.optimization.de_opt   import OptHelper, SetupOpt
from cl3o.optimization.fpenalty import Penalty, PenaltyData
from cl3o.optimization.fscore   import StructuralMass, ScoreData
from cl3o.fea.post.tsw_failure     import FailureData
from cl3o.fea.post.displacement_ms import DisplacementData


# ================================================================================
# PUBLIC API - Input validators
# ================================================================================

def test_validate_bounds_accepts_ordered() -> None:
    '''Strictly ordered, equal-length bounds pass validation.'''
    OptHelper.validate_bounds(np.zeros(5), np.ones(5))


def test_validate_bounds_rejects_unordered() -> None:
    '''hi <= lo anywhere must raise.'''
    with pytest.raises(ValueError):
        OptHelper.validate_bounds(np.ones(3), np.zeros(3))


def test_validate_bounds_rejects_length_mismatch() -> None:
    '''Mismatched lengths must raise.'''
    with pytest.raises(ValueError):
        OptHelper.validate_bounds(np.zeros(3), np.ones(4))


def test_validate_hyperpar_guards() -> None:
    '''CR out of [0,1] or NP < 4 must raise; a valid pair passes.'''
    OptHelper.validate_hyperpar(NP=10, CR=0.9)
    with pytest.raises(ValueError):
        OptHelper.validate_hyperpar(NP=10, CR=1.5)
    with pytest.raises(ValueError):
        OptHelper.validate_hyperpar(NP=3, CR=0.5)


# ================================================================================
# PUBLIC API - DE bounds construction
# ================================================================================

def test_build_de_bounds_layout() -> None:
    '''_build_de_bounds yields D = 11*n_cpts + 3 strictly ordered bounds.'''
    n_cpts, n_mats = 4, 8
    lo, hi = SetupOpt._build_de_bounds(n_cpts=n_cpts, n_mats=n_mats)

    expected_D = 11 * n_cpts + 3
    assert lo.size == hi.size == expected_D
    assert np.all(hi > lo)

    # The 8 * n_cpts discrete layup tail must address a valid MAT* index.
    tail_hi = hi[-8 * n_cpts:]
    assert float(np.max(tail_hi)) <= n_mats


# ================================================================================
# PUBLIC API - Logistic penalty
# ================================================================================

def test_penalty_feasible_is_zero_flag() -> None:
    '''No violations -> feasible and a finite, non-negative penalty.'''
    fd = FailureData(nv=0, MS_min_component=np.array([0.5, 1.2, 0.3]))
    dd = DisplacementData(nv=0, MS_min=0.4)

    p = Penalty(data=(fd, dd), enable_logging=False).data

    assert isinstance(p, PenaltyData)
    assert p.is_feasible is True
    assert p.nv_total == 0
    assert np.isfinite(p.total) and p.total >= 0.0


def test_penalty_infeasible_is_positive() -> None:
    '''Violations drive the penalty strictly positive and flag infeasible.'''
    fd = FailureData(nv=2, MS_min_component=np.array([-0.3, 0.1]))
    dd = DisplacementData(nv=1, MS_min=-0.2)

    p = Penalty(data=(fd, dd), enable_logging=False).data

    assert p.is_feasible is False
    assert p.nv_total == 3
    assert p.total > 0.0


def test_penalty_monotonic_in_violations() -> None:
    '''More severe violations cannot decrease the penalty.'''
    mild = Penalty(
        data=(FailureData(nv=1, MS_min_component=np.array([-0.05])),
              DisplacementData(nv=0, MS_min=0.5)),
        enable_logging=False,
    ).data.total
    severe = Penalty(
        data=(FailureData(nv=5, MS_min_component=np.array([-0.8])),
              DisplacementData(nv=3, MS_min=-0.6)),
        enable_logging=False,
    ).data.total
    assert severe >= mild


# ================================================================================
# PUBLIC API - Structural mass (real solved section)
# ================================================================================

@pytest.mark.slow
def test_structural_mass_positive(runtime, static) -> None:
    '''StructuralMass on a real section gives a positive, additive mass.'''
    score = StructuralMass(
        sections       = runtime.sections,
        element_idx    = runtime.mesh.conn[:, :2],
        laminate_db    = static.laminate_db,
        enable_logging = False,
    ).data

    assert isinstance(score, ScoreData)
    assert score.total > 0.0
    assert score.per_elem.shape == (score.m,)
    np.testing.assert_allclose(
        score.per_elem, score.panels + score.flanges, rtol=1e-9, atol=1e-9,
    )
    np.testing.assert_allclose(
        score.total, float(np.sum(score.per_elem)), rtol=1e-6,
    )
