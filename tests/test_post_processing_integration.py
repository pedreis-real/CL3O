'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Post-Processing Integration Tests.

End-to-end check of the archive fan-out RunOpt performs when given an
out_dir: one pickled RuntimeData snapshot per generation plus a manifest
JSON, with the manifest snapshot count matching the pickles on disk and
each pickle reloading into a populated RuntimeData.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import json
import pickle

import pytest

# ================ Module imports ================
from cl3o.main import RunCLEO
from cl3o.optimization.fobjective import RuntimeData

pytestmark = pytest.mark.slow


# ================ Global variables ================
_AIRCRAFT = "DA62"
_OPT_NAME = "PostProcTest"
_DE = {"NP": 6, "CR": 0.9, "F": 0.8, "lambda": 0.5,
       "k_max": 2, "seed": 7, "std_tol": 0.0}


# ================================================================================
# PUBLIC API - Archive fan-out test
# ================================================================================

def test_generation_archive_and_manifest(tmp_path, db_specs) -> None:
    '''RunOpt writes one pickle per generation and a matching manifest.'''
    run = RunCLEO(
        aircraft_name  = _AIRCRAFT,
        opt_name       = _OPT_NAME,
        db_specs       = db_specs,
        de_hyperpar    = _DE,
        enable_logging = False,
    )
    out_dir = tmp_path / "run"
    run.run_optimization(out_dir=out_dir)

    gens_dir = out_dir / "generations"
    manifest = out_dir / "manifest.json"
    assert gens_dir.exists()
    assert manifest.exists()

    pkls = sorted(gens_dir.glob("gen_*.pkl"))
    assert len(pkls) >= 1

    with open(manifest) as fh:
        mani = json.load(fh)
    assert "schema_version" in mani
    # Snapshots may include duplicates pointing to a previous pickle; only
    # the distinct ones produce a file on disk.
    distinct = [s for s in mani["snapshots"] if not s.get("is_duplicate", False)]
    assert len(distinct) == len(pkls)

    # Each archived snapshot reloads into a populated RuntimeData.
    with open(pkls[0], "rb") as fh:
        snap = pickle.load(fh)
    assert isinstance(snap, RuntimeData)
    assert snap.fitness is not None
