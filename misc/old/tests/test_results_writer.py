'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Results Writer Tests Module.

Phase I regression tests for results.results_writer.ResultsWriter:

  - Packing: ResultsHelper.pack_results folds OptData + HistoryData into
    a ResultsData archive with consistent shapes and scalar metadata.
  - Disk I/O: ResultsWriter.write() creates the JSON file on disk and
    the resulting payload round-trips back into a ResultsData instance.
  - Custom filepath override: the write() target can be redirected per
    call without mutating the stored default path.

@ CWSS Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path

import numpy as np
import pytest

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ================ Module imports ================
from utils                  import io_utils as io
from optimization.de_opt    import OptData, HistoryData
from results.results_writer import (
    ResultsData, ResultsHelper, ResultsWriter,
)


# ================================================================================
# PRIVATE API - Synthetic fixtures
# ================================================================================

def _make_opt_history(
    n_dim: int = 3,
    n_gen: int = 8,
    NP   : int = 10,
    seed : int = 11,
) -> tuple[OptData, HistoryData]:
    '''Build a matched OptData / HistoryData pair for archive testing.'''
    rng = np.random.default_rng(seed)
    lo  = np.full(n_dim, -1.0)
    hi  = np.full(n_dim,  1.0)
    opt = OptData(
        lo = lo,
        hi = hi,
        NP        = NP,
        CR        = 0.9,
        F         = 0.8,
        lmbda       = 0.5,
        k_max     = n_gen,
        seed      = seed,
        n_dim     = n_dim,
        X0        = rng.uniform(lo, hi, size=(NP, n_dim)),
    )
    # Monotonic-ish best to mimic a convergent run
    best_f = np.exp(-np.linspace(0.0, 3.0, n_gen + 1)) * 5.0 + 0.5
    mean_f = best_f + np.linspace(2.0, 0.2, n_gen + 1)
    std_f  = np.linspace(1.5, 0.1, n_gen + 1)

    hist = HistoryData(
        ng      = n_gen,
        D      = n_dim,
        best_X     = rng.uniform(lo, hi, size=(n_gen + 1, n_dim)),
        best_f     = best_f,
        mean_f     = mean_f,
        std_f      = std_f,
        feasible_X = rng.uniform(lo, hi, size=n_dim),
        feasible_f = float(best_f[-1]),
    )
    return opt, hist


# ================================================================================
# PUBLIC API - Packing tests
# ================================================================================

def test_pack_results_preserves_scalars() -> None:
    '''Scalar DE hyper-parameters are copied verbatim into ResultsData.'''
    opt, hist = _make_opt_history()
    packed = ResultsHelper.pack_results(
        aircraft_name = "DA62",
        opt_name      = "smoke",
        opt           = opt,
        history       = hist,
    )

    assert isinstance(packed, ResultsData)
    assert packed.aircraft == "DA62"
    assert packed.opt_name == "smoke"
    assert packed.n_dim    == hist.D
    assert packed.n_gen    == hist.ng
    assert packed.NP       == opt.NP
    assert packed.k_max    == opt.k_max
    assert np.isclose(packed.CR,  opt.CR)
    assert np.isclose(packed.F,   opt.F)
    assert np.isclose(packed.lam, opt.lmbda)


def test_pack_results_shapes_match_history() -> None:
    '''Array shapes survive the pack step with n_dim / n_gen semantics.'''
    opt, hist = _make_opt_history(n_dim=4, n_gen=5)
    packed = ResultsHelper.pack_results("DA62", "shape", opt, hist)

    assert packed.best_X.shape    == (hist.ng + 1, hist.D)
    assert packed.best_f.shape    == (hist.ng + 1,)
    assert packed.mean_f.shape    == (hist.ng + 1,)
    assert packed.std_f.shape     == (hist.ng + 1,)
    assert packed.feasible_X.shape == (hist.D,)
    assert packed.bounds_lo.shape == (hist.D,)


def test_default_filepath_uses_lowercase_tokens() -> None:
    '''Default archive path lowercases the aircraft/opt names.'''
    out_dir = Path("outputs_test")
    p = ResultsHelper.default_filepath("DA62", "OptRun", out_dir=out_dir)
    assert p.parent == out_dir
    assert p.name.startswith("da62_optrun_")
    assert p.suffix == ".json"


# ================================================================================
# PUBLIC API - Disk I/O tests
# ================================================================================

def test_writer_writes_and_round_trips(tmp_path: Path) -> None:
    '''write() produces a JSON that round-trips back into ResultsData.'''
    opt, hist = _make_opt_history(n_dim=3, n_gen=4)

    writer = ResultsWriter(
        aircraft_name  = "DA62",
        opt_name       = "rt",
        opt_data       = opt,
        history        = hist,
        out_dir        = tmp_path,
        enable_logging = False,
    )
    target = writer.write()

    assert target.exists()
    assert target.parent == tmp_path

    loaded = io.read_json(target, ResultsData)
    assert isinstance(loaded, ResultsData)
    assert loaded.aircraft == "DA62"
    assert loaded.n_dim    == opt.n_dim
    assert loaded.n_gen    == hist.ng
    assert np.allclose(np.asarray(loaded.best_f), np.asarray(writer.data.best_f))
    assert np.allclose(
        np.asarray(loaded.best_X), np.asarray(writer.data.best_X)
    )


def test_writer_honours_filepath_override(tmp_path: Path) -> None:
    '''An override path is used without mutating the stored default.'''
    opt, hist = _make_opt_history()

    writer = ResultsWriter(
        aircraft_name  = "DA62",
        opt_name       = "override",
        opt_data       = opt,
        history        = hist,
        out_dir        = tmp_path,
        enable_logging = False,
    )
    override = tmp_path / "custom" / "archive.json"
    actual   = writer.write(filepath=override)

    assert actual == override
    assert override.exists()
    # Default path must remain unchanged and must NOT have been written.
    assert writer.filepath != override
    assert not writer.filepath.exists()


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pytest.main([__file__, "-q"])
