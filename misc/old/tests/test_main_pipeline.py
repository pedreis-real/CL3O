'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Main Pipeline Smoke Tests Module.

Phase H regression tests for the main.RunCWSS optimization driver.
The test bypasses the static-database loader (which needs on-disk
JSONs) and exercises run_optimization directly with a convex
quadratic evaluator. It confirms that the method wires SetupOpt ->
RunOpt correctly and stores both on RuntimeData.

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
import main as cwss_main
from main                  import RunCL3O, RuntimeData
from optimization.de_opt   import SetupOpt, RunOpt, HistoryData


# ================================================================================
# PRIVATE API - Bypass __init__ helper
# ================================================================================

def _make_runner() -> RunCL3O:
    '''
    Construct a RunCWSS instance without loading any database by
    bypassing __init__ and injecting the minimum attributes needed by
    run_optimization: self.logger and self.runtime.
    '''
    runner = RunCL3O.__new__(RunCL3O)
    import logging
    runner.logger  = logging.getLogger("RunCWSS-test")
    runner.runtime = RuntimeData()
    return runner


# ================================================================================
# PUBLIC API - Test cases
# ================================================================================

def test_main_module_imports_cleanly() -> None:
    '''main.py imports without error and exposes expected top classes.'''
    assert hasattr(cwss_main, "RunCWSS")
    assert hasattr(cwss_main, "BuildDatabase")
    assert hasattr(cwss_main, "PostProcessing")
    assert hasattr(cwss_main, "StaticData")
    assert hasattr(cwss_main, "RuntimeData")


def test_run_optimization_stores_results_on_runtime() -> None:
    '''run_optimization wires SetupOpt and RunOpt into RuntimeData.'''
    runner = _make_runner()
    lo = np.full(3, -5.0)
    hi = np.full(3,  5.0)

    history = runner.run_optimization(
        bounds_lo = lo,
        bounds_hi = hi,
        evaluator = lambda X: float(np.sum(X ** 2)),
        NP        = 16,
        k_max     = 10,
        seed      = 3,
    )

    assert isinstance(history, HistoryData)
    assert isinstance(runner.runtime.opt_setup,  SetupOpt)
    assert isinstance(runner.runtime.opt_result, RunOpt)
    assert history.best_f[-1] < history.best_f[0]


def test_run_optimization_respects_feasible_check() -> None:
    '''
    Mark only the subset |X| <= 2 as feasible. The best feasible
    design returned by RunOpt must satisfy that bound.
    '''
    runner = _make_runner()
    lo = np.full(2, -5.0)
    hi = np.full(2,  5.0)

    def feasible(X: np.ndarray) -> bool:
        return bool(np.all(np.abs(X) <= 2.0))

    history = runner.run_optimization(
        bounds_lo      = lo,
        bounds_hi      = hi,
        evaluator      = lambda X: float(np.sum(X ** 2)),
        feasible_check = feasible,
        NP             = 12,
        k_max          = 12,
        seed           = 5,
    )

    assert np.all(np.abs(history.feasible_X) <= 2.0 + 1.0e-9)
    assert history.feasible_f < float("inf")


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
