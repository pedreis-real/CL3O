'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Penalty-Constants Grid Sweep Script.

Tunes the logistic penalty constants (L, psi_1, psi_2) of fpenalty.py
over the intervals listed in Tabela 9 of the thesis (p. 89):

    L      in [2*df, 6*df]     penalidade maxima
    psi_1  in [0.05, 0.15]     fracao P(v_1)/L no patamar v_1 = 0.05
    psi_2  in [0.80, 0.95]     fracao P(v_2)/L no patamar v_2 = 0.20

For each grid combination the script runs a reduced-budget DE with a
fixed baseline (NP, CR, F, lambda) and records the best fitness,
population mean/std, and convergence generation. The emitted JSON +
Markdown artifacts feed Section 4.3 of the monograph.

Usage:
    python scripts/tune_penalty.py              # quick sweep (2x2x2 = 8)
    python scripts/tune_penalty.py --full       # 3x3x3 = 27 combinations
    python scripts/tune_penalty.py --k_max 100  # override DE budget

@ CWSS Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ================ Default Database Paths ================
_OUT_DIR = _ROOT / "artifacts"
_OUT_DIR.mkdir(exist_ok=True)

# ================ Module imports ================
from main import (
    RunCL3O, DatabaseSpec, build_default_evaluator, _build_de_bounds,
    _DFLT_AFL_DIR, _DFLT_MAT_DIR, _DFLT_LDS_DIR, _DFLT_OPP_DIR, _DFLT_WNG_DIR,
)
from geometry.wing         import WingData
from geometry.airfoil      import AirfoilData
from materials.laminate    import LaminateData
from fem.loads.load_mapper import ExLoadsData
from utils.oppoints        import OppData

# ================ Global variables ================
# Baseline DE hyper-parameters for the sweep. Kept fixed so the only
# lever moving across combinations is the penalty triplet.
_BASELINE_NP     = 16
_BASELINE_CR     = 0.90
_BASELINE_F      = 0.80
_BASELINE_LAMBDA = 0.50
_BASELINE_SEED   = 42

# Sweep grids (per Tabela 9 intervals).
_GRID_QUICK = {
    "L"    : [2_000.0, 6_000.0],
    "psi1" : [0.05, 0.15],
    "psi2" : [0.80, 0.95],
}
_GRID_FULL = {
    "L"    : [2_000.0, 4_000.0, 6_000.0],
    "psi1" : [0.05, 0.10, 0.15],
    "psi2" : [0.80, 0.875, 0.95],
}


# ================================================================================
# PRIVATE API - Pipeline bootstrap
# ================================================================================

def _build_db_specs(aircraft: str, airfoil: str, n_mat: int) -> list:
    '''Mirror the __main__ block of src/main.py.'''
    db = [DatabaseSpec(AirfoilData, _DFLT_AFL_DIR, airfoil.lower())]
    for k in range(1, n_mat + 1):
        db.append(DatabaseSpec(LaminateData, _DFLT_MAT_DIR, f"MAT{k}"))
    db.append(DatabaseSpec(ExLoadsData, _DFLT_LDS_DIR, aircraft.lower()))
    db.append(DatabaseSpec(OppData,     _DFLT_OPP_DIR, aircraft.lower()))
    db.append(DatabaseSpec(WingData,    _DFLT_WNG_DIR, aircraft.lower()))
    return db


def _iter_combos(grid: dict) -> list[tuple[float, float, float]]:
    '''Cartesian product of (L, psi1, psi2) grid values.'''
    return [
        (L, p1, p2)
        for L  in grid["L"]
        for p1 in grid["psi1"]
        for p2 in grid["psi2"]
    ]


def _convergence_gen(best_f: np.ndarray, rel_tol: float = 1.0e-3) -> int:
    '''
    Generation at which best_f stops improving by more than rel_tol
    (relative to the final value). Returns n_gen if never converged.
    '''
    final = float(best_f[-1])
    if not np.isfinite(final) or abs(final) < 1.0e-12:
        return int(best_f.size)
    threshold = abs(final) * rel_tol
    for k in range(best_f.size - 1, 0, -1):
        if abs(best_f[k] - final) > threshold:
            return int(k + 1)
    return 0


# ================================================================================
# PUBLIC API - Sweep driver
# ================================================================================

def run_sweep(
    aircraft : str    = "DA62",
    airfoil  : str    = "WortmannFX63137",
    n_mat    : int    = 5,
    k_max    : int    = 60,
    grid     : dict   = None,
    out_stem : str    = "penalty_sweep",
) -> Path:
    '''Run the sweep and dump JSON + Markdown artifacts. Returns JSON path.'''
    grid    = grid or _GRID_QUICK
    combos  = _iter_combos(grid)
    runner  = RunCL3O(
        aircraft_name  = aircraft,
        opt_name       = "PenSweep",
        db_specs       = _build_db_specs(aircraft, airfoil, n_mat),
        enable_logging = False,
    )
    lo, hi  = _build_de_bounds(runner.static)
    results = []

    t_start = time.time()
    for idx, (L, p1, p2) in enumerate(combos, start=1):
        evaluator = build_default_evaluator(
            static         = runner.static,
            runtime        = runner.runtime,
            penalty_kwargs = {
                "L"              : float(L),
                "psi1"           : float(p1),
                "psi2"           : float(p2),
                "enable_logging" : False,
            },
        )
        t0 = time.time()
        history = runner.run_optimization(
            bounds_lo      = lo,
            bounds_hi      = hi,
            evaluator      = evaluator,
            NP             = _BASELINE_NP,
            CR             = _BASELINE_CR,
            F              = _BASELINE_F,
            lam            = _BASELINE_LAMBDA,
            k_max          = k_max,
            seed           = _BASELINE_SEED,
            tol            = 1.0e-6,
            stall_patience = 30,
        )
        dt = time.time() - t0

        k_conv = _convergence_gen(history.best_f)
        rec = {
            "combo_id"  : idx,
            "L"         : float(L),
            "psi1"      : float(p1),
            "psi2"      : float(p2),
            "k_max"     : int(k_max),
            "n_gen"     : int(history.n_gen),
            "best_f"    : float(history.best_f[-1]),
            "mean_f"    : float(history.mean_f[-1]),
            "std_f"     : float(history.std_f[-1]),
            "k_convergence": int(k_conv),
            "best_X"    : history.best_X[-1].tolist(),
            "runtime_s" : round(dt, 2),
        }
        results.append(rec)
        print(
            f"[{idx:>2}/{len(combos)}] "
            f"L={L:7.0f} psi1={p1:.3f} psi2={p2:.3f}  "
            f"best_f={rec['best_f']:.2e}  "
            f"k_conv={k_conv:>3}  "
            f"dt={dt:6.1f}s"
        )

    total_dt = time.time() - t_start

    # -------- Pick the winner: lowest final best_f, tiebreak k_conv --------
    best_rec = min(
        results,
        key = lambda r: (r["best_f"], r["k_convergence"]),
    )

    out_json = _OUT_DIR / f"{out_stem}.json"
    out_md   = _OUT_DIR / f"{out_stem}.md"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump({
            "grid"       : grid,
            "baseline"   : {
                "NP": _BASELINE_NP, "CR": _BASELINE_CR,
                "F" : _BASELINE_F,  "lambda": _BASELINE_LAMBDA,
                "seed": _BASELINE_SEED, "k_max": k_max,
            },
            "total_runtime_s": round(total_dt, 2),
            "best_combo"     : best_rec,
            "results"        : results,
        }, f, indent=2)

    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# Penalty Sweep Results\n\n")
        f.write(f"- Baseline: NP={_BASELINE_NP}, CR={_BASELINE_CR}, "
                f"F={_BASELINE_F}, lambda={_BASELINE_LAMBDA}, "
                f"k_max={k_max}\n")
        f.write(f"- Total runtime: {total_dt:.1f} s\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| # | L | psi1 | psi2 | best_f | k_conv | dt [s] |\n")
        f.write(f"|---|---|------|------|--------|--------|--------|\n")
        for r in results:
            flag = " **WINNER**" if r["combo_id"] == best_rec["combo_id"] else ""
            f.write(
                f"| {r['combo_id']} | {r['L']:.0f} | {r['psi1']:.3f} "
                f"| {r['psi2']:.3f} | {r['best_f']:.3e} "
                f"| {r['k_convergence']} | {r['runtime_s']:.1f} |{flag}\n"
            )
        f.write(f"\n## Winner\n\n")
        f.write(f"- L     = {best_rec['L']}\n")
        f.write(f"- psi1  = {best_rec['psi1']}\n")
        f.write(f"- psi2  = {best_rec['psi2']}\n")
        f.write(f"- best_f = {best_rec['best_f']:.4e}\n")

    print(f"\n[{out_json}] written.")
    print(f"[{out_md}] written.")
    print(f"Winner: L={best_rec['L']}, psi1={best_rec['psi1']}, "
          f"psi2={best_rec['psi2']}, best_f={best_rec['best_f']:.3e}")
    return out_json


# ================================================================================
# CLI entrypoint
# ================================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full",  action="store_true",
                   help="Use the 3x3x3 full grid (default: quick 2x2x2).")
    p.add_argument("--k_max", type=int, default=60,
                   help="DE iterations per combo (default: 60).")
    p.add_argument("--out",   default="penalty_sweep",
                   help="Artifact file stem under artifacts/.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_sweep(
        grid     = _GRID_FULL if args.full else _GRID_QUICK,
        k_max    = args.k_max,
        out_stem = args.out,
    )
