'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Results Writer Tests.

Checks ResultsHelper.pack_results folds an OptData + HistoryData pair into
a ResultsData archive with matching scalars, and that ResultsWriter.write
round-trips the archive through JSON without loss.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.utils import io_utils as io
from cl3o.results.results_writer import (
    ResultsWriter, ResultsData, ResultsHelper,
)

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Packing + round-trip tests
# ================================================================================

def test_pack_results_scalars(opt_data, de_history) -> None:
    '''pack_results copies the run scalars verbatim from OptData/History.'''
    rd = ResultsHelper.pack_results(
        aircraft_name = "DA62",
        opt_name      = "WriterTest",
        opt           = opt_data,
        history       = de_history,
    )
    assert isinstance(rd, ResultsData)
    assert rd.n_dim == int(de_history.D)
    assert rd.n_gen == int(de_history.ng)
    assert rd.NP == int(opt_data.NP)
    assert rd.bounds_lo.size == rd.n_dim


def test_write_roundtrip(tmp_path, opt_data, de_history) -> None:
    '''A written ResultsData reloads with identical core arrays.'''
    writer = ResultsWriter(
        aircraft_name  = "DA62",
        opt_name       = "WriterTest",
        opt_data       = opt_data,
        history        = de_history,
        out_dir        = tmp_path,
        enable_logging = False,
    )
    path = writer.write()
    assert path.exists()

    reloaded = io.read_json(filepath=path, dcls=ResultsData)
    assert reloaded.n_dim == writer.data.n_dim
    assert reloaded.n_gen == writer.data.n_gen
    np.testing.assert_allclose(
        np.asarray(reloaded.best_f, dtype=float),
        np.asarray(writer.data.best_f, dtype=float),
    )
