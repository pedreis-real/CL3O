'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
DE Outer-Loop Tests.

Runs a tiny end-to-end DE optimization (via the de_history fixture) and
asserts the HistoryData arrays are well-shaped, finite, and that the
running-best fitness is monotonically non-increasing across generations.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.optimization.de_opt import HistoryData

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - HistoryData tests
# ================================================================================

def test_history_well_shaped(de_history) -> None:
    '''HistoryData arrays share the (ng+1) record length and dimension D.'''
    hist = de_history
    assert isinstance(hist, HistoryData)

    n_records = hist.best_f.size
    assert hist.best_X.shape == (n_records, hist.D)
    assert hist.mean_f.size == n_records
    assert hist.std_f.size == n_records


def test_history_finite_and_feasible_dim(de_history) -> None:
    '''All fitness arrays are finite; std is non-negative.'''
    hist = de_history
    assert np.all(np.isfinite(hist.best_f))
    assert np.all(np.isfinite(hist.mean_f))
    assert np.all(hist.std_f >= -1e-12)
    assert hist.feasible_X.size == hist.D


def test_running_best_monotonic(de_history) -> None:
    '''The best-so-far fitness never increases generation to generation.'''
    rbest = np.minimum.accumulate(de_history.best_f)
    assert np.all(np.diff(rbest) <= 1e-9)
