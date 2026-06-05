'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Runtime-Pipeline Validation Module.

Exercises every step of the call chain that ``src/main.py`` triggers in a
normal CL3O session, asserting that each artifact ends up in the right
container, with the right shape, and the right type.

Scenarios
---------
1. Database loading       - replicates main.RunCLEO._import_database and
                            checks that StaticData is fully populated
                            with all expected sub-objects.
2. DE bounds              - asserts SetupOpt._build_de_bounds dimensions
                            (D = 11 * n_cpts + 3), strict ordering
                            (hi > lo) and that layup indices stay within
                            the laminate_db cardinality.
3. Evaluator pipeline     - drives BuildEvaluator.eval_ once on a random
                            design vector at the centre of bounds and
                            asserts every RuntimeData slot
                            (sections .. fitness) was written.
4. Snapshot consistency   - re-evaluates the same X and verifies that
                            the fitness scalar is reproducible and that
                            BuildEvaluator.snapshot_best returns the
                            same RuntimeData object identity.
5. Design-vector guards   - exercises FobjectiveHelper.validate_design_vector
                            for the three documented failure modes.
6. DE outer loop          - runs RunOpt with a tiny (NP, k_max)
                            configuration and asserts that HistoryData
                            arrays are well-shaped, finite, and the
                            best fitness is monotonically non-increasing.
7. PostProcessing fan-out - exercises PostProcessing.run end-to-end and
                            checks that every expected artifact (archive
                            JSON + each plot file) was created on disk.

The script is module-level executable; on success prints a summary
table with the status of each scenario. Any assertion failure halts
the run with an explicit error.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import os
import sys
import json
import shutil
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

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

# Constants
from cl3o.Constants import DE_HYPERPAR, OPT_LIMS

# Main entry-points (re-uses everything the CL3O CLI driver builds)
from cl3o.main import (
    RunCLEO,
    DatabaseSpec, StaticData,
    MainHelpers, _resolve_db_specs,
)

# Geometry / materials / loads (typed dataclasses)
from cl3o.geometry.wing      import WingData, LerpWingData
from cl3o.geometry.airfoil   import AirfoilData
from cl3o.materials.laminate import LaminateData, PlyData
from cl3o.utils.oppoints     import OppData
from cl3o.utils.database_utils import discover_laminates
from cl3o.fea.loads.load_mapper import ExLoadsData, InLoadsData
from cl3o.fea.pre.fem_setup  import FemPreprocessData

# Optimization
from cl3o.optimization.de_opt    import (
    SetupOpt, RunOpt, OptData, HistoryData, OptVars,
)
from cl3o.optimization.fobjective import (
    BuildEvaluator, RuntimeData, FitnessData, FobjectiveHelper,
)


# ================ Global variables ================
_AIRCRAFT    = "DA62"
_AFL_NAME    = "WortmannFX63137"

_TEST_OPT_NAME = "RuntimePipelineTest"

# A purposefully tiny DE configuration so the validation runs in seconds.
_TEST_DE_HYPERPAR: dict = {
    "NP"      : 6,
    "CR"      : 0.9,
    "F"       : 0.8,
    "lambda"  : 0.5,
    "k_max"   : 2,
    "seed"    : 1234,
    "std_tol" : 0.0,
}


# ================================================================================
# Internal helpers - assertion utilities
# ================================================================================

class _Reporter:
    '''Minimal pass/fail aggregator that fails loud on the first issue.'''

    def __init__(self) -> None:
        self.records : list[tuple[str, str]] = []

    def expect(self, label: str, cond: bool, detail: str = "") -> None:
        if not cond:
            raise AssertionError(
                f"[CL3O validate_runtime_pipeline] FAIL: {label}\n"
                f"| detail : {detail}"
            )
        self.records.append((label, detail))

    def section(self, name: str) -> None:
        bar = "=" * 72
        print(f"\n{bar}\n  {name}\n{bar}")

    def summary(self) -> None:
        bar = "-" * 72
        print(f"\n{bar}\n  PIPELINE VALIDATION SUMMARY ({len(self.records)} checks)\n{bar}")
        for label, detail in self.records:
            tag = f"  [OK] {label}"
            print(tag if not detail else f"{tag}  ::  {detail}")


def _discover_materials() -> list[str]:
    '''Discover the curated laminate catalogue exactly as main.py does:
    glob MAT_*_LaminateData.json (skips legacy MAT{int} test laminates).'''
    return discover_laminates(_DFLT_MAT_DIR)


def _build_db_specs() -> list[DatabaseSpec]:
    '''Replicate the DA62 spec list assembled at the bottom of main.py.'''
    specs : list[DatabaseSpec] = [
        DatabaseSpec(WingData, _DFLT_WNG_DIR, _AIRCRAFT.lower()),
        DatabaseSpec(AirfoilData, _DFLT_AFL_DIR, _AFL_NAME.lower()),
    ]
    for mat_name in _discover_materials():
        specs.append(DatabaseSpec(LaminateData, _DFLT_MAT_DIR, mat_name))
    specs.append(DatabaseSpec(OppData, _DFLT_OPP_DIR, _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(ExLoadsData, _DFLT_LDS_DIR, _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(InLoadsData, _DFLT_LDS_DIR, _AIRCRAFT.lower()))
    return _resolve_db_specs(specs)


def _midpoint_design(setup: SetupOpt) -> np.ndarray:
    '''Return the centre of the DE bounds: useful as a deterministic probe.'''
    return 0.5 * (setup.data.lo + setup.data.hi)


def _cleanup_dir(path: Path) -> None:
    '''Wipe a directory; only used on test-scoped output paths.'''
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _make_runner(
    db_specs : list,
    opt_name : str = _TEST_OPT_NAME,
    **overrides,
) -> RunCLEO:
    '''Single source of truth for RunCLEO construction in this script.'''
    return RunCLEO(
        aircraft_name  = _AIRCRAFT,
        opt_name       = opt_name,
        db_specs       = db_specs,
        de_hyperpar    = _TEST_DE_HYPERPAR,
        runner_options = {"enable_logging": False},
        **overrides,
    )


# ================================================================================
# PUBLIC API - Scenario 1: Database loading
# ================================================================================

def scenario_database(rep: _Reporter) -> RunCLEO:
    '''
    Drive RunCLEO.__init__ exactly as main.py does, then sanity-check
    StaticData. Returns the fully-built RunCLEO instance, reused by all
    downstream scenarios so we only pay the IO cost once.
    '''
    rep.section("Scenario 1 - Database loading + StaticData population")

    db_specs = _build_db_specs()
    MainHelpers.verify_missing_database(db_specs)
    n_mats = len(_discover_materials())
    rep.expect(
        f"DatabaseSpec list resolves to N={5 + n_mats} entries (5 + {n_mats} laminates)",
        len(db_specs) == 5 + n_mats,
        f"got {len(db_specs)}",
    )

    runner = _make_runner(db_specs)

    st = runner.static
    rep.expect("static is StaticData",       isinstance(st, StaticData))
    rep.expect("wing_db is WingData",        isinstance(st.wing_db, WingData))
    rep.expect("lerp_wing_db is LerpWingData", isinstance(st.lerp_wing_db, LerpWingData))
    rep.expect("airfoil_db is dict",         isinstance(st.airfoil_db, dict))
    rep.expect(
        "airfoil_db contains target profile",
        _AFL_NAME.lower() in st.airfoil_db,
        f"keys={list(st.airfoil_db.keys())}",
    )

    rep.expect("laminate_db is dict",        isinstance(st.laminate_db, dict))
    rep.expect(
        "laminate_db has sequential MAT* keys",
        all(f"MAT{k}" in st.laminate_db for k in range(1, len(st.laminate_db) + 1)),
        f"keys={sorted(st.laminate_db.keys())}",
    )
    rep.expect(
        "every laminate is a LaminateData",
        all(isinstance(v, LaminateData) for v in st.laminate_db.values()),
    )

    rep.expect("ply_db populated", isinstance(st.ply_db, dict) and len(st.ply_db) > 0)
    rep.expect(
        "ply_db values are PlyData",
        all(isinstance(v, PlyData) for v in st.ply_db.values()),
    )

    rep.expect("opp_db is OppData",          isinstance(st.opp_db, OppData))
    rep.expect("exloads_db is ExLoadsData",  isinstance(st.exloads_db, ExLoadsData))
    rep.expect("inloads_db is InLoadsData",  isinstance(st.inloads_db, InLoadsData))
    rep.expect("fem_setup is FemPreprocessData", isinstance(st.fem_setup, FemPreprocessData))

    # FemPreprocessData shape integrity
    fem = st.fem_setup
    n   = int(fem.n)
    nc  = int(fem.loads["nc"])
    rep.expect("fem_setup.n > 1",  n > 1)
    rep.expect("fem_setup.m == n-1", fem.m == n - 1)
    rep.expect("fem_setup.dof == 6n", fem.dof == 6 * n)
    rep.expect(
        "fem_setup.loads['F'] shape == (n, 6, nc)",
        np.asarray(fem.loads["F"]).shape == (n, 6, nc),
    )
    rep.expect(
        "fem_setup.loads['F_flat'] shape == (6n, nc)",
        np.asarray(fem.loads["F_flat"]).shape == (6 * n, nc),
    )

    return runner


# ================================================================================
# PUBLIC API - Scenario 2: DE bounds and SetupOpt cache
# ================================================================================

def scenario_setup_opt(rep: _Reporter, runner: RunCLEO) -> None:
    '''Inspect the SetupOpt stashed at static.opt_setup by RunCLEO.__init__.'''
    rep.section("Scenario 2 - DE bounds and SetupOpt cache")

    setup = runner.static.opt_setup
    rep.expect("opt_setup is SetupOpt", isinstance(setup, SetupOpt))

    opt = setup.data
    rep.expect("OptData stored", isinstance(opt, OptData))
    n_cpts = int(runner.static.wing_db.n_cpts)
    expected_D = 11 * n_cpts + 3
    rep.expect(
        "DE dimension matches OptVars layout",
        int(opt.D) == expected_D,
        f"D={opt.D}, expected {expected_D}",
    )

    rep.expect("bounds_lo / hi same length", opt.lo.size == opt.hi.size == opt.D)
    rep.expect(
        "hi strictly greater than lo",
        bool(np.all(opt.hi > opt.lo)),
        f"min(hi-lo) = {float(np.min(opt.hi - opt.lo)):.3e}",
    )
    rep.expect(
        "X0 shape (NP, D)",
        opt.X0.shape == (opt.NP, opt.D),
        f"got {opt.X0.shape}",
    )
    rep.expect(
        "X0 within bounds",
        bool(np.all(opt.X0 >= opt.lo) and np.all(opt.X0 <= opt.hi)),
    )

    # Layup tail of the bounds must address a valid MAT* index. The layup
    # index ranges are split per group (skin/web/flange); the tail floor is
    # the smallest lower bound across those groups.
    n_mats = len(runner.static.laminate_db)
    layup_lo_floor = min(
        OPT_LIMS["layup_skin"][0],
        OPT_LIMS["layup_web"][0],
        OPT_LIMS["layup_flange"][0],
    )
    tail_lo = opt.lo[-8 * n_cpts:]
    tail_hi = opt.hi[-8 * n_cpts:]
    rep.expect(
        "layup tail upper bound <= n_mats",
        float(np.max(tail_hi)) <= n_mats,
        f"max(tail_hi)={float(np.max(tail_hi))}, n_mats={n_mats}",
    )
    rep.expect(
        "layup tail lower bound >= min layup index",
        float(np.min(tail_lo)) >= float(layup_lo_floor),
        f"min(tail_lo)={float(np.min(tail_lo))}, floor={layup_lo_floor}",
    )

    # _build_de_bounds is a pure static method; exercise it standalone too.
    lo2, hi2 = SetupOpt._build_de_bounds(n_cpts=n_cpts, n_mats=n_mats)
    rep.expect("_build_de_bounds returns matching D", lo2.size == hi2.size == expected_D)


# ================================================================================
# PUBLIC API - Scenario 3: Evaluator pipeline (RuntimeData fan-out)
# ================================================================================

def scenario_evaluator(rep: _Reporter, runner: RunCLEO) -> None:
    '''
    Run BuildEvaluator.eval_ on the centre of the bounds and check that
    every RuntimeData slot has been written.
    '''
    rep.section("Scenario 3 - Evaluator pipeline + RuntimeData fan-out")

    setup = runner.static.opt_setup
    builder = runner._builder
    rep.expect("BuildEvaluator stored on runner", isinstance(builder, BuildEvaluator))
    rep.expect("evaluator is callable",            callable(runner.evaluator))
    rep.expect("rt is RuntimeData (pre-eval)",     isinstance(builder.rt, RuntimeData))

    X = _midpoint_design(setup)
    f = runner.evaluator(X)
    rep.expect(
        "fitness is finite float",
        np.isfinite(f) and isinstance(f, float),
        f"f={f!r}",
    )

    rt = builder.rt
    rep.expect("rt.sections populated", rt.sections is not None)
    rep.expect("rt.mesh populated",     rt.mesh     is not None)
    rep.expect("rt.fea_rts populated",  rt.fea_rts  is not None)
    rep.expect("rt.stress populated",   rt.stress   is not None)
    rep.expect("rt.tsw populated",      rt.tsw      is not None)
    rep.expect("rt.displ populated",    rt.displ    is not None)
    rep.expect("rt.penalty populated",  rt.penalty  is not None)
    rep.expect("rt.score populated",    rt.score    is not None)
    rep.expect("rt.fitness is FitnessData", isinstance(rt.fitness, FitnessData))
    rep.expect(
        "rt.fitness.total == eval_ return",
        np.isclose(rt.fitness.total, f),
        f"rt={rt.fitness.total}, ret={f}",
    )

    # FEA arrays linked to the right number of DOFs.
    n_dof = int(runner.static.fem_setup.dof)
    K = np.asarray(rt.mesh.K, dtype=float)
    rep.expect(
        "global K is square (n_dof, n_dof)",
        K.ndim == 2 and K.shape[0] == K.shape[1] == n_dof,
        f"K.shape={K.shape}, n_dof={n_dof}",
    )

    dm = np.asarray(rt.fea_rts.dmatrix, dtype=float)
    rep.expect(
        "dmatrix shape is (6, n, nc)",
        dm.ndim == 3 and dm.shape[0] == 6,
        f"dmatrix.shape={dm.shape}",
    )

    # Fitness arithmetic should match TotalScore definition.
    from cl3o.Constants import WEIGHTING_FACTOR
    z = float(rt.score.total) * float(WEIGHTING_FACTOR) + float(rt.penalty.total)
    rep.expect(
        "z(X) = w_m * m + P(X) (thesis Eq. 3.68)",
        np.isclose(z, rt.fitness.total, rtol=1e-9, atol=1e-9),
        f"reconstructed={z}, stored={rt.fitness.total}",
    )
    rep.expect(
        "is_feasible mirrors penalty feasibility",
        bool(rt.fitness.is_feasible) == bool(rt.penalty.is_feasible),
    )


# ================================================================================
# PUBLIC API - Scenario 4: snapshot consistency
# ================================================================================

def scenario_snapshot(rep: _Reporter, runner: RunCLEO) -> None:
    '''Re-evaluating the same X must yield the same fitness, bit-for-bit.'''
    rep.section("Scenario 4 - evaluator determinism + best_rt tracking")

    setup   = runner.static.opt_setup
    builder = runner._builder
    X = _midpoint_design(setup)

    f1 = float(runner.evaluator(X))
    rt1_id = id(builder.rt)
    f2 = float(runner.evaluator(X))
    rt2_id = id(builder.rt)

    rep.expect("rt identity preserved across calls", rt1_id == rt2_id)
    rep.expect(
        "fitness deterministic on identical X",
        np.isclose(f1, f2, rtol=0.0, atol=0.0),
        f"f1={f1}, f2={f2}",
    )
    rep.expect(
        "best_rt captured (RuntimeData)",
        isinstance(builder.best_rt, RuntimeData) and builder.best_rt.fitness is not None,
    )


# ================================================================================
# PUBLIC API - Scenario 5: design-vector validation
# ================================================================================

def scenario_design_vector_guards(rep: _Reporter, runner: RunCLEO) -> None:
    '''
    FobjectiveHelper.validate_design_vector must reject:
        a) non-ndarray inputs,
        b) wrong-size vectors,
        c) NaN / inf entries.
    '''
    rep.section("Scenario 5 - FobjectiveHelper.validate_design_vector")

    expected = 11 * int(runner.static.wing_db.n_cpts) + 3
    good = _midpoint_design(runner.static.opt_setup)

    rep.expect("baseline X passes", good.size == expected)
    FobjectiveHelper.validate_design_vector(good, expected)  # no-throw

    # a) wrong type
    try:
        FobjectiveHelper.validate_design_vector([0.0] * expected, expected)
        raised = False
    except ValueError:
        raised = True
    rep.expect("rejects non-ndarray input", raised)

    # b) wrong size
    try:
        FobjectiveHelper.validate_design_vector(good[:-1], expected)
        raised = False
    except ValueError:
        raised = True
    rep.expect("rejects wrong-size vector", raised)

    # c) non-finite entry
    bad = good.copy()
    bad[0] = np.nan
    try:
        FobjectiveHelper.validate_design_vector(bad, expected)
        raised = False
    except ValueError:
        raised = True
    rep.expect("rejects NaN entry", raised)


# ================================================================================
# PUBLIC API - Scenario 6: DE outer loop
# ================================================================================

def scenario_de_loop(rep: _Reporter, runner: RunCLEO) -> None:
    '''
    Drive RunOpt with the tiny test config and verify HistoryData layout
    and running-best monotonicity.
    '''
    rep.section("Scenario 6 - RunOpt outer loop + HistoryData")

    out_dir = _DFLT_OUT_DIR / f"{_AIRCRAFT.lower()}_{_TEST_OPT_NAME.lower()}"
    _cleanup_dir(out_dir)

    runner.run_optimization(out_dir=out_dir)
    hist = runner.static.opt_result.history

    rep.expect("history is HistoryData", isinstance(hist, HistoryData))

    k_max = int(_TEST_DE_HYPERPAR["k_max"])
    rep.expect(
        "history.ng <= k_max",
        int(hist.ng) <= k_max,
        f"ng={hist.ng}, k_max={k_max}",
    )
    rep.expect("history.D matches OptData", int(hist.D) == int(runner.static.opt_setup.data.D))

    n_records = hist.best_f.size
    rep.expect("best_X (ng+1, D)", hist.best_X.shape == (n_records, hist.D))
    rep.expect("mean_f same length as best_f", hist.mean_f.size == n_records)
    rep.expect("std_f  same length as best_f", hist.std_f.size  == n_records)

    rep.expect("best_f all finite", bool(np.all(np.isfinite(hist.best_f))))
    rep.expect("std_f all >= 0",    bool(np.all(hist.std_f  >= -1e-12)))

    # Per-gen running best must be non-increasing.
    rbest = np.minimum.accumulate(hist.best_f)
    rep.expect(
        "running best monotonically non-increasing",
        bool(np.all(np.diff(rbest) <= 1e-9)),
    )

    rep.expect(
        "archive directory created",
        (out_dir / "generations").exists(),
    )
    rep.expect(
        "archive manifest.json written",
        (out_dir / "manifest.json").exists(),
    )

    # The manifest is JSON-readable and schema-tagged.
    with open(out_dir / "manifest.json") as fh:
        mani = json.load(fh)
    rep.expect("manifest exposes schema_version", "schema_version" in mani)
    distinct = [s for s in mani["snapshots"] if not s.get("is_duplicate", False)]
    rep.expect(
        "distinct snapshot count matches written pickles",
        len(distinct) == len(list((out_dir / "generations").glob("gen_*.pkl"))),
    )



# ================================================================================
# Module-level usage
# ================================================================================

def main() -> int:
    rep = _Reporter()
    try:
        runner = scenario_database    (rep)
        scenario_setup_opt            (rep, runner)
        scenario_evaluator            (rep, runner)
        scenario_snapshot             (rep, runner)
        scenario_design_vector_guards (rep, runner)
        scenario_de_loop              (rep, runner)
    except AssertionError as err:
        print(err)
        traceback.print_exc()
        return 1
    except Exception as err:  # pragma: no cover - surfaces wiring errors
        print(f"[CL3O validate_runtime_pipeline] UNEXPECTED ERROR: {err!r}")
        traceback.print_exc()
        return 2

    rep.summary()
    print("\n[CL3O validate_runtime_pipeline] all scenarios passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
