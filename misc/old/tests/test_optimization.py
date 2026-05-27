'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Optimization Smoke Tests Module.

Phase G regression tests for the optimization layer:

  - StructuralMass (fscore): mass of a 1 m cantilever with two boom
    flanges of 100 mm^2 each, density 1600 kg/m^3 = 1.6e-6 t/mm^3,
    returns 1000 mm * 4 * 100 mm^2 * 1.6e-6 t/mm^3 * 1000 kg/t =
    0.64 kg. Panels are zero-thickness, so panel mass must be 0.
  - Penalty (fpenalty): p = 1 when feasible; formula
    exp((a*MS_sum+b)/n) verified for known inputs; overflow clipped.
  - TotalScore (fobjective): mass_coef * mass + penalty; feasible flag
    mirrors the input penalty_data.feasible.
  - SetupOpt + RunOpt (de_opt): a convex quadratic minimised to ~0 in
    a few generations confirms the DE-3 loop converges.

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
from fea.post.tsw_failure     import FailureData
from fea.post.displacement_ms import DisplacementData
from optimization.fscore      import StructuralMass, ScoreHelper
from optimization.fpenalty    import Penalty, PenaltyHelper
from optimization.fobjective  import TotalScore
from optimization.de_opt      import SetupOpt, RunOpt

from misc.old.tests.conftest import (
    _make_box_section, _make_single_ply_bundle, L_TOT, BOOM_A,
)


# ================================================================================
# PRIVATE API - Optimization fixtures
# ================================================================================

def _single_section_mass_inputs():
    '''
    Two-element cantilever along +Y sharing one section. Every panel
    bundle is the single-ply laminate from conftest; flange bundles
    share the same ply. Panel t_k in the fixture section is 1 mm but
    we override it to 0 to isolate the flange mass contribution.
    '''
    sec    = _make_box_section()
    sec.t_k = np.zeros(7)
    coord  = np.array([
        [0.0,    0.0, 0.0],
        [0.0,  500.0, 0.0],
        [0.0, L_TOT,  0.0],
    ])
    conn   = np.array([[0, 1], [1, 2]], dtype=int)
    sec.A_flange = np.full(4, BOOM_A)

    bundle = _make_single_ply_bundle()
    panels  = [[bundle] * 7]
    flanges = [[bundle] * 4]
    elem_sec_idx = np.array([0, 0], dtype=int)
    return sec, coord, conn, elem_sec_idx, panels, flanges


# ================================================================================
# PUBLIC API - Mass tests
# ================================================================================

def test_structural_mass_flanges_only() -> None:
    '''Mass matches Sum_j rho * A_fj * L for zero-thickness panels.'''
    sec, coord, conn, elem_sec_idx, panels, flanges = (
        _single_section_mass_inputs()
    )
    sm = StructuralMass(
        sec_data       = [sec],
        coord          = coord,
        conn           = conn,
        elem_sec_idx   = elem_sec_idx,
        panel_bundles  = panels,
        flange_bundles = flanges,
        enable_logging = False,
    ).data

    rho     = flanges[0][0].laminate.rho
    mu_f    = 4 * rho * BOOM_A
    expected = 1000.0 * mu_f * L_TOT
    assert np.isclose(sm.mass_total, expected, atol=1.0e-3)
    assert np.all(np.isclose(sm.mass_panels, 0.0, atol=1.0e-6))


def test_structural_mass_per_element_sum() -> None:
    '''Sum of per-element mass equals mass_total.'''
    sec, coord, conn, elem_sec_idx, panels, flanges = (
        _single_section_mass_inputs()
    )
    sm = StructuralMass(
        sec_data       = [sec],
        coord          = coord,
        conn           = conn,
        elem_sec_idx   = elem_sec_idx,
        panel_bundles  = panels,
        flange_bundles = flanges,
        enable_logging = False,
    ).data
    assert np.isclose(float(np.sum(sm.mass_per_elem)), sm.mass_total, atol=1.0e-3)


def test_element_length_helper() -> None:
    '''ScoreHelper.element_length matches Euclidean distance.'''
    coord = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    conn  = np.array([[0, 1]], dtype=int)
    assert np.isclose(ScoreHelper.element_length(coord, conn, 0), 5.0)


# ================================================================================
# PUBLIC API - Penalty tests
# ================================================================================

def test_penalty_no_violations_is_zero() -> None:
    '''P(X) = 0 when there are no violations (logistic penalty, Eq. 3.63).'''
    fd = FailureData(nv=0)
    dd = DisplacementData(n_violations=0)
    pd = Penalty(fd, dd, enable_logging=False).data
    assert pd.n_violations == 0
    assert np.isclose(pd.penalty, 0.0, atol=1.0e-6)
    assert pd.feasible is True


def test_penalty_reaches_psi_fractions_at_thresholds() -> None:
    '''P(v1)/L ~ psi1 and P(v2)/L ~ psi2 under the g(-k*v0) ~ 0 approximation.'''
    L, psi1, psi2 = 1000.0, 0.10, 0.90
    k, v0 = PenaltyHelper.derive_k_v0(psi1, psi2)
    p1 = PenaltyHelper.penalty_value(0.05, L, k, v0, n=1)
    p2 = PenaltyHelper.penalty_value(0.20, L, k, v0, n=1)
    # Tolerance 3% absorbs the g(-k*v0) ~ 0 approximation baked into
    # Eqs. 3.67-3.68; k*v0 is O(1) in this regime.
    assert np.isclose(p1 / L, psi1, atol=3.0e-2)
    assert np.isclose(p2 / L, psi2, atol=3.0e-2)


def test_penalty_is_capped() -> None:
    '''Penalty is clipped to _PENALTY_CAP to avoid numeric overflow.'''
    k, v0 = PenaltyHelper.derive_k_v0(0.10, 0.90)
    p = PenaltyHelper.penalty_value(1.0e6, 1.0e20, k, v0, n=1)
    assert np.isfinite(p)


# ================================================================================
# PUBLIC API - Total score tests
# ================================================================================

def test_total_score_combines_mass_and_penalty() -> None:
    '''total = mass_coef * mass + penalty.'''
    from optimization.fscore   import ScoreData
    from optimization.fpenalty import PenaltyData

    ts = TotalScore(
        mass_data      = ScoreData(mass_total=2.5),
        penalty_data   = PenaltyData(n_violations=0, penalty=1.0, feasible=True),
        mass_coef      = 1000.0,
        enable_logging = False,
    ).data
    assert np.isclose(ts.total, 1000.0 * 2.5 + 1.0, atol=1.0e-6)
    assert ts.feasible is True


def test_total_score_infeasible_flag() -> None:
    '''feasible mirrors the penalty_data flag.'''
    from optimization.fscore   import ScoreData
    from optimization.fpenalty import PenaltyData

    ts = TotalScore(
        mass_data      = ScoreData(mass_total=1.0),
        penalty_data   = PenaltyData(n_violations=3, penalty=20.0, feasible=False),
        mass_coef      = 1000.0,
        enable_logging = False,
    ).data
    assert ts.feasible is False


# ================================================================================
# PUBLIC API - DE solver tests
# ================================================================================

def test_de_converges_on_sphere() -> None:
    '''
    Minimise f(X) = Sum X_i^2 in [-5, 5]^3. With k_max=30 the best-so-
    far fitness must be near zero.
    '''
    lo = np.full(3, -5.0)
    hi = np.full(3,  5.0)

    def evaluator(X: np.ndarray) -> float:
        return float(np.sum(np.asarray(X, dtype=float) ** 2))

    setup = SetupOpt(
        bounds_lo = lo, bounds_hi = hi,
        evaluator = evaluator,
        NP = 20, CR = 0.9, F = 0.8, lam = 0.5,
        k_max = 30, seed = 7,
        enable_logging = False,
    )
    run = RunOpt(setup, enable_logging=False).history
    assert run.best_f[-1] < 1.0e-2
    assert run.best_f[-1] <= run.best_f[0]


def test_setup_bounds_validation() -> None:
    '''SetupOpt rejects degenerate bounds.'''
    lo = np.array([1.0, 2.0])
    hi = np.array([1.0, 3.0])

    try:
        SetupOpt(
            bounds_lo = lo, bounds_hi = hi,
            evaluator = lambda X: 0.0,
            enable_logging = False,
        )
    except ValueError:
        return
    raise AssertionError("SetupOpt should have raised ValueError.")


def test_de_history_shapes() -> None:
    '''HistoryData arrays have the expected (k_max + 1) length.'''
    lo = np.array([-1.0, -1.0])
    hi = np.array([ 1.0,  1.0])
    setup = SetupOpt(
        bounds_lo = lo, bounds_hi = hi,
        evaluator = lambda X: float(np.sum(X ** 2)),
        NP = 8, k_max = 5, seed = 1,
        enable_logging = False,
    )
    h = RunOpt(setup, enable_logging=False).history
    assert h.best_X.shape == (6, 2)
    assert h.best_f.shape == (6,)
    assert h.mean_f.shape == (6,)
    assert h.std_f.shape  == (6,)


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
