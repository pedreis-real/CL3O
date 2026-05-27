'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Final DE Optimization Run.

Executes the full DE run on the DA62 left wing with tuned parameters
from artifacts/penalty_sweep.json (Tabela 9 winner) and
artifacts/de_lhs.json (Tabela 10 winner). Produces the optimum
individual and all plots needed for Section 4.6 of the monograph.

Usage:
    python scripts/run_final_opt.py              # tuned params, k_max=400
    python scripts/run_final_opt.py --k_max 200  # quicker run
    python scripts/run_final_opt.py --np 40 --cr 0.9 --f 0.8 --lam 0.5

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
_OUT_DIR      = _ROOT / "artifacts"
_OUT_DIR.mkdir(exist_ok=True)
_PENALTY_JSON = _OUT_DIR / "penalty_sweep.json"
_DE_JSON      = _OUT_DIR / "de_lhs.json"
_POST_DIR     = _ROOT / "out" / "da62_final"

# ================ Module imports ================
from main import (
    RunCL3O, DatabaseSpec, build_default_evaluator, _build_de_bounds,
    PostProcessing,
    _DFLT_AFL_DIR, _DFLT_MAT_DIR, _DFLT_LDS_DIR, _DFLT_OPP_DIR, _DFLT_WNG_DIR,
)
from geometry.wing         import WingData
from geometry.airfoil      import AirfoilData
from materials.laminate    import LaminateData
from fem.loads.load_mapper import ExLoadsData
from utils.oppoints        import OppData

# ================ Global variables ================
_DFLT_PENALTY = {"L": 4_000.0, "psi1": 0.10, "psi2": 0.90}
_DFLT_DE      = {"NP": 40, "CR": 0.90, "F": 0.80, "lam": 0.50}


# ================================================================================
# PRIVATE API - Parameter loading
# ================================================================================

def _build_db_specs(aircraft: str, airfoil: str, n_mat: int) -> list:
    db = [DatabaseSpec(AirfoilData, _DFLT_AFL_DIR, airfoil.lower())]
    for k in range(1, n_mat + 1):
        db.append(DatabaseSpec(LaminateData, _DFLT_MAT_DIR, f"MAT{k}"))
    db.append(DatabaseSpec(ExLoadsData, _DFLT_LDS_DIR, aircraft.lower()))
    db.append(DatabaseSpec(OppData,     _DFLT_OPP_DIR, aircraft.lower()))
    db.append(DatabaseSpec(WingData,    _DFLT_WNG_DIR, aircraft.lower()))
    return db


def _load_penalty(override: dict | None = None) -> dict:
    if override:
        return override
    if not _PENALTY_JSON.is_file():
        print(f"[final_opt] No {_PENALTY_JSON.name}; using defaults.")
        return dict(_DFLT_PENALTY)
    with _PENALTY_JSON.open(encoding="utf-8") as f:
        best = json.load(f).get("best_combo", {})
    out = {
        "L"    : float(best.get("L",    _DFLT_PENALTY["L"])),
        "psi1" : float(best.get("psi1", _DFLT_PENALTY["psi1"])),
        "psi2" : float(best.get("psi2", _DFLT_PENALTY["psi2"])),
    }
    print(f"[final_opt] Penalty loaded: {out}")
    return out


def _load_de_params(override: dict | None = None) -> dict:
    if override:
        return override
    if not _DE_JSON.is_file():
        print(f"[final_opt] No {_DE_JSON.name}; using defaults.")
        return dict(_DFLT_DE)
    with _DE_JSON.open(encoding="utf-8") as f:
        best = json.load(f).get("best_sample", {})
    out = {
        "NP" : int(best.get("NP",     _DFLT_DE["NP"])),
        "CR" : float(best.get("CR",   _DFLT_DE["CR"])),
        "F"  : float(best.get("F",    _DFLT_DE["F"])),
        "lam": float(best.get("lambda", _DFLT_DE["lam"])),
    }
    print(f"[final_opt] DE params loaded: {out}")
    return out


# ================================================================================
# PUBLIC API - Final run driver
# ================================================================================

def run_final(
    aircraft : str  = "DA62",
    airfoil  : str  = "WortmannFX63137",
    n_mat    : int  = 5,
    k_max    : int  = 400,
    seed     : int  = 42,
    stall_p  : int  = 50,
    penalty  : dict | None = None,
    de_params: dict | None = None,
) -> None:
    '''Run the full DE optimisation and export artifacts for Section 4.6.'''
    pen_kwargs = _load_penalty(penalty)
    pen_kwargs["enable_logging"] = False
    de = _load_de_params(de_params)

    runner = RunCL3O(
        aircraft_name  = aircraft,
        opt_name       = "FinalOpt",
        db_specs       = _build_db_specs(aircraft, airfoil, n_mat),
        enable_logging = True,
    )
    lo, hi = _build_de_bounds(runner.static)
    evaluator = build_default_evaluator(
        static         = runner.static,
        runtime        = runner.runtime,
        penalty_kwargs = pen_kwargs,
    )

    t0 = time.time()
    history = runner.run_optimization(
        bounds_lo      = lo,
        bounds_hi      = hi,
        evaluator      = evaluator,
        NP             = de["NP"],
        CR             = de["CR"],
        F              = de["F"],
        lam            = de["lam"],
        k_max          = k_max,
        seed           = seed,
        tol            = 1.0e-6,
        stall_patience = stall_p,
    )
    dt = time.time() - t0
    print(f"[final_opt] n_gen={history.n_gen}, best_f={history.best_f[-1]:.4e}, "
          f"dt={dt:.1f} s")

    best_X = history.best_X[-1].copy()

    # Re-evaluate once to refresh runtime artefacts (section, stress, tsw).
    _ = evaluator(best_X)

    # Archive the optimum.
    opt_json = _OUT_DIR / "final_opt.json"
    with opt_json.open("w", encoding="utf-8") as f:
        json.dump({
            "penalty_kwargs" : pen_kwargs,
            "de_params"      : de,
            "k_max"          : k_max,
            "seed"           : seed,
            "n_gen"          : int(history.n_gen),
            "best_f"         : float(history.best_f[-1]),
            "best_X"         : best_X.tolist(),
            "best_f_history" : history.best_f.tolist(),
            "mean_f_history" : history.mean_f.tolist(),
            "std_f_history"  : history.std_f.tolist(),
            "runtime_s"      : round(dt, 2),
        }, f, indent=2)
    print(f"[{opt_json}] written.")

    # Full post-processing (plots, section/wing/load figs).
    _POST_DIR.mkdir(parents=True, exist_ok=True)
    PostProcessing(enable_logging=True).run(runner=runner, out_dir=_POST_DIR)
    print(f"[final_opt] Plots under {_POST_DIR}")


# ================================================================================
# CLI entrypoint
# ================================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k_max", type=int, default=400)
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--np",    dest="np_",  type=int, default=None,
                   help="Override NP (else: tuned from de_lhs.json).")
    p.add_argument("--cr",    type=float, default=None)
    p.add_argument("--f",     type=float, default=None)
    p.add_argument("--lam",   type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    de_override = None
    if any(v is not None for v in (args.np_, args.cr, args.f, args.lam)):
        de_override = {
            "NP"  : args.np_ or _DFLT_DE["NP"],
            "CR"  : args.cr  if args.cr  is not None else _DFLT_DE["CR"],
            "F"   : args.f   if args.f   is not None else _DFLT_DE["F"],
            "lam" : args.lam if args.lam is not None else _DFLT_DE["lam"],
        }
    run_final(k_max=args.k_max, seed=args.seed, de_params=de_override)
