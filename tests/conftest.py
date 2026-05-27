'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Shared Test Fixtures Module.

Provides the fixtures every unit-test file leans on. The expensive ones
build a real CL3O session against the DA62 / WortmannFX63137 database
exactly as ``cl3o.main.RunCLEO`` does, so the FEA + post-processing
modules can be exercised against genuine, fully-populated StaticData and
RuntimeData containers instead of synthetic stand-ins.

Fixture overview
----------------
    db_specs        : resolved DatabaseSpec list for the DA62 reference.
    runner          : session-scoped RunCLEO with one evaluated design.
    static          : StaticData (wing/airfoil/laminate/ply/loads/fem_setup).
    runtime         : a solved RuntimeData (sections .. fitness populated).
    de_history      : HistoryData from a tiny end-to-end DE run.
    airfoil_arrays  : raw adimensional WortmannFX63137 coordinate arrays.

The heavy fixtures are session-scoped (built once) and the tests that
consume them are tagged ``@pytest.mark.slow`` so ``pytest -m "not slow"``
stays fast in development.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend for the plot smoke tests

import numpy as np
import pytest

# ================ Default Database Paths ================
from cl3o.paths import (
    WINGS_DIR     as _DFLT_WNG_DIR,
    AIRFOILS_DIR  as _DFLT_AFL_DIR,
    MATERIALS_DIR as _DFLT_MAT_DIR,
    OPPOINTS_DIR  as _DFLT_OPP_DIR,
    LOADS_DIR     as _DFLT_LDS_DIR,
    OUTPUTS_DIR   as _DFLT_OUT_DIR,
)

# ================ Module imports ================
from cl3o.main import (
    RunCLEO, DatabaseSpec, StaticData, MainHelpers, _resolve_db_specs,
)
from cl3o.geometry.wing      import WingData
from cl3o.geometry.airfoil   import AirfoilData
from cl3o.materials.laminate import LaminateData
from cl3o.utils.oppoints     import OppData
from cl3o.fea.loads.load_mapper import ExLoadsData, InLoadsData


# ================ Global variables ================
_AIRCRAFT = "DA62"
_AFL_NAME = "WortmannFX63137"
_OPT_NAME = "PytestSession"

# Purposefully tiny DE configuration so the suite runs in seconds.
_TEST_DE_HYPERPAR = {
    "NP": 6, "CR": 0.9, "F": 0.8, "lambda": 0.5,
    "k_max": 2, "seed": 1234, "std_tol": 0.0,
}


# ================================================================================
# PRIVATE API - DA62 database assembly (mirrors cl3o.main bottom block)
# ================================================================================

def _discover_materials() -> list[str]:
    '''Glob the curated laminate catalogue, skipping legacy MAT{int}.'''
    return sorted(
        f.stem.removesuffix("_LaminateData")
        for f in _DFLT_MAT_DIR.glob("MAT_*_LaminateData.json")
    )


def _build_db_specs() -> list[DatabaseSpec]:
    '''Replicate the DA62 spec list assembled at the bottom of main.py.'''
    specs: list[DatabaseSpec] = [
        DatabaseSpec(WingData,    _DFLT_WNG_DIR, _AIRCRAFT.lower()),
        DatabaseSpec(AirfoilData, _DFLT_AFL_DIR, _AFL_NAME.lower()),
    ]
    for mat_name in _discover_materials():
        specs.append(DatabaseSpec(LaminateData, _DFLT_MAT_DIR, mat_name))
    specs.append(DatabaseSpec(OppData,     _DFLT_OPP_DIR, _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(ExLoadsData, _DFLT_LDS_DIR, _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(InLoadsData, _DFLT_LDS_DIR, _AIRCRAFT.lower()))
    return _resolve_db_specs(specs)


# ================================================================================
# PUBLIC API - Session fixtures
# ================================================================================

@pytest.fixture(scope="session")
def db_specs() -> list[DatabaseSpec]:
    '''Resolved DatabaseSpec list for the DA62 reference design.'''
    specs = _build_db_specs()
    MainHelpers.verify_missing_database(specs)
    return specs


@pytest.fixture(scope="session")
def runner(db_specs) -> RunCLEO:
    '''
    Fully-built RunCLEO with one design evaluated at the centre of the
    DE bounds, so runner._builder.rt is a solved RuntimeData.
    '''
    run = RunCLEO(
        aircraft_name  = _AIRCRAFT,
        opt_name       = _OPT_NAME,
        db_specs       = db_specs,
        de_hyperpar    = _TEST_DE_HYPERPAR,
        enable_logging = False,
    )
    setup = run.static.opt_setup
    X = 0.5 * (setup.data.lo + setup.data.hi)
    run.evaluator(X)
    return run


@pytest.fixture(scope="session")
def static(runner) -> StaticData:
    '''The populated StaticData container.'''
    return runner.static


@pytest.fixture(scope="session")
def runtime(runner):
    '''A solved RuntimeData (sections .. fitness populated).'''
    return runner._builder.rt


@pytest.fixture(scope="session")
def de_history(db_specs):
    '''HistoryData from a tiny end-to-end DE run on a scratch out_dir.'''
    run = RunCLEO(
        aircraft_name  = _AIRCRAFT,
        opt_name       = _OPT_NAME + "Hist",
        db_specs       = db_specs,
        de_hyperpar    = _TEST_DE_HYPERPAR,
        enable_logging = False,
    )
    out_dir = _DFLT_OUT_DIR / f"{_AIRCRAFT.lower()}_{(_OPT_NAME + 'Hist').lower()}"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    run.run_optimization(out_dir=out_dir)
    history = run.static.opt_result.history
    yield history
    shutil.rmtree(out_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def opt_data(runner):
    '''OptData stashed on the session runner.'''
    return runner.static.opt_setup.data


# ================================================================================
# PUBLIC API - Lightweight fixtures (no DB load)
# ================================================================================

@pytest.fixture(scope="session")
def airfoil_arrays() -> dict[str, np.ndarray]:
    '''Raw adimensional WortmannFX63137 coordinate arrays from JSON.'''
    path = _DFLT_AFL_DIR / f"{_AFL_NAME.lower()}_AirfoilData.json"
    with open(path) as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=float) for k, v in raw.items()}
