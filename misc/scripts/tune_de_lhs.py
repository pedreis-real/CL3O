'''
================================================================================
CWSS - Composite Wing Structural Sizing.
DE Hyperparameter LHS Tuning Script.

Runs a Latin-Hypercube sample over the DE control parameters listed in
Tabela 10 of the thesis (p. 90):

    NP      in [10, 100]          step 10     (tamanho da populacao)
    CR      in [0.1, 0.9]         step 0.1    (probabilidade de recombinacao)
    F       in [0.4, 1.0]         step 0.1    (fator de variacao diferencial)
    lambda  in [0.0, 1.0]         step 0.1    (fator de aprimoramento ganancioso)
    k_max   fixo em 400

The penalty triplet (L, psi_1, psi_2) is pinned to the winner recorded
by scripts/tune_penalty.py in artifacts/penalty_sweep.json. Falls back
to defaults (_DFLT_L, _DFLT_PSI1, _DFLT_PSI2 from fpenalty.py) if the
artifact is absent.

For each LHS sample the script runs a reduced-budget DE (k_max defaults
to 100 for tractable run time; pass --k_max 400 to match the thesis)
and records convergence metrics. Output artifacts feed Section 4.4 of
the monograph.

Usage:
    python scripts/tune_de_lhs.py                 # 30 samples, k_max=100
    python scripts/tune_de_lhs.py --n 30 --k_max 400
    python scripts/tune_de_lhs.py --seed 7

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
from scipy.stats import qmc

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ================ Default Database Paths ================
_OUT_DIR        = _ROOT / "artifacts"
_OUT_DIR.mkdir(exist_ok=True)
_PENALTY_JSON   = _OUT_DIR / "penalty_sweep.json"

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
# LHS bounds per Tabela 10.
_LHS_LO = np.array([10.0, 0.10, 0.40, 0.00])  # NP, CR, F, lambda
_LHS_HI = np.array([100.0, 0.90, 1.00, 1.00])

# Fallback penalty triplet (midpoints of Tabela 9).
_DFLT_PENALTY = {"L": 4_000.0, "psi1": 0.10, "psi2": 0.90}


# ================================================================================
# PRIVATE API - Pipeline bootstrap
# ================================================================================

def _build_db_specs(aircraft: str, airfoil: str, n_mat: int) -> list:
    db = [DatabaseSpec(AirfoilData, _DFLT_AFL_DIR, airfoil.lower())]
    for k in range(1, n_mat + 1):
        db.append(DatabaseSpec(LaminateData, _DFLT_MAT_DIR, f"MAT{k}"))
    db.append(DatabaseSpec(ExLoadsData, _DFLT_LDS_DIR, aircraft.lower()))
    db.append(DatabaseSpec(OppData,     _DFLT_OPP_DIR, aircraft.lower()))
    db.append(DatabaseSpec(WingData,    _DFLT_WNG_DIR, aircraft.lower()))
    return db


def _load_penalty_winner() -> dict:
    '''
    Pull (L, psi1, psi2) from the penalty-sweep artifact if present,
    otherwise fall back to midpoint defaults.
    '''
    if not _PENALTY_JSON.is_file():
        print(f"[tune_de_lhs] No penalty sweep artifact at {_PENALTY_JSON}; "
              f"using defaults {_DFLT_PENALTY}.")
        return dict(_DFLT_PENALTY)
    with _PENALTY_JSON.open(encoding="utf-8") as f:
        payload = json.load(f)
    best = payload.get("best_combo", {})
    out  = {
        "L"    : float(best.get("L",    _DFLT_PENALTY["L"])),
        "psi1" : float(best.get("psi1", _DFLT_PENALTY["psi1"])),
        "psi2" : float(best.get("psi2", _DFLT_PENALTY["psi2"])),
    }
    print(f"[tune_de_lhs] Penalty winner loaded: {out}")
    return out


def _build_lhs_samples(n: int, seed: int) -> np.ndarray:
    '''
    Latin-Hypercube sample in the unit hypercube, scaled to the Tab. 10
    intervals, with NP rounded to integer (step 10 floor to align with
    the thesis grid).
    '''
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    unit    = sampler.random(n=n)
    scaled  = qmc.scale(unit, _LHS_LO, _LHS_HI)
    # NP to nearest multiple of 10 per Tab. 10 "passo 10".
    scaled[:, 0] = np.round(scaled[:, 0] / 10.0) * 10.0
    scaled[:, 0] = np.clip(scaled[:, 0], _LHS_LO[0], _LHS_HI[0])
    return scaled


def _convergence_gen(best_f: np.ndarray, rel_tol: float = 1.0e-3) -> int:
    '''Generation at which best_f locks onto the final value within rel_tol.'''
    final = float(best_f[-1])
    if not np.isfinite(final) or abs(final) < 1.0e-12:
        return int(best_f.size)
    threshold = abs(final) * rel_tol
    for k in range(best_f.size - 1, 0, -1):
        if abs(best_f[k] - final) > threshold:
            return int(k + 1)
    return 0


# ================================================================================
# PUBLIC API - LHS sweep driver
# ================================================================================

def run_lhs(
    aircraft : str  = "DA62",
    airfoil  : str  = "WortmannFX63137",
    n_mat    : int  = 5,
    n_samples: int  = 30,
    k_max    : int  = 100,
    seed     : int  = 42,
    out_stem : str  = "de_lhs",
) -> Path:
    '''Run the LHS sweep and dump JSON + Markdown artifacts.'''
    penalty_kwargs = _load_penalty_winner()
    penalty_kwargs["enable_logging"] = False

    samples = _build_lhs_samples(n_samples, seed)

    runner  = RunCL3O(
        aircraft_name  = aircraft,
        opt_name       = "DELhs",
        db_specs       = _build_db_specs(aircraft, airfoil, n_mat),
        enable_logging = False,
    )
    lo, hi  = _build_de_bounds(runner.static)
    results = []

    t_start = time.time()
    for idx, (NP_f, CR, F, lam) in enumerate(samples, start=1):
        NP = int(NP_f)
        evaluator = build_default_evaluator(
            static         = runner.static,
            runtime        = runner.runtime,
            penalty_kwargs = penalty_kwargs,
        )
        t0 = time.time()
        history = runner.run_optimization(
            bounds_lo      = lo,
            bounds_hi      = hi,
            evaluator      = evaluator,
            NP             = NP,
            CR             = float(CR),
            F              = float(F),
            lam            = float(lam),
            k_max          = k_max,
            seed           = seed,
            tol            = 1.0e-6,
            stall_patience = 30,
        )
        dt = time.time() - t0

        k_conv = _convergence_gen(history.best_f)
        rec = {
            "sample_id"     : idx,
            "NP"            : NP,
            "CR"            : round(float(CR),  4),
            "F"             : round(float(F),   4),
            "lambda"        : round(float(lam), 4),
            "k_max"         : int(k_max),
            "n_gen"         : int(history.n_gen),
            "best_f"        : float(history.best_f[-1]),
            "mean_f"        : float(history.mean_f[-1]),
            "std_f"         : float(history.std_f[-1]),
            "k_convergence" : int(k_conv),
            "n_evaluations" : int(history.n_gen * NP),
            "runtime_s"     : round(dt, 2),
        }
        results.append(rec)
        print(
            f"[{idx:>2}/{n_samples}] "
            f"NP={NP:>3} CR={CR:.2f} F={F:.2f} lam={lam:.2f}  "
            f"best_f={rec['best_f']:.3e} "
            f"k_conv={k_conv:>3} "
            f"evals={rec['n_evaluations']:>5} "
            f"dt={dt:5.1f}s"
        )

    total_dt = time.time() - t_start

    # Winner: fewest evaluations to convergence, tiebreaker best_f.
    best_rec = min(
        results,
        key = lambda r: (r["n_evaluations"] if r["k_convergence"] < r["n_gen"]
                         else r["n_gen"] * r["NP"] + 1_000_000,
                         r["best_f"]),
    )

    out_json = _OUT_DIR / f"{out_stem}.json"
    out_md   = _OUT_DIR / f"{out_stem}.md"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump({
            "lhs_bounds"        : {"lo": _LHS_LO.tolist(),
                                   "hi": _LHS_HI.tolist()},
            "penalty_kwargs"    : penalty_kwargs,
            "k_max"             : k_max,
            "seed"              : seed,
            "total_runtime_s"   : round(total_dt, 2),
            "best_sample"       : best_rec,
            "samples"           : results,
        }, f, indent=2)

    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# DE LHS Tuning Results\n\n")
        f.write(f"- Penalty (pinned): L={penalty_kwargs['L']}, "
                f"psi1={penalty_kwargs['psi1']}, "
                f"psi2={penalty_kwargs['psi2']}\n")
        f.write(f"- k_max = {k_max}, n_samples = {n_samples}, seed = {seed}\n")
        f.write(f"- Total runtime: {total_dt:.1f} s\n\n")
        f.write(f"## Samples (sorted by n_evaluations ascending)\n\n")
        f.write(f"| # | NP | CR | F | lambda | best_f | k_conv | evals | dt [s] |\n")
        f.write(f"|---|----|----|---|--------|--------|--------|-------|--------|\n")
        for r in sorted(results, key=lambda x: x["n_evaluations"]):
            flag = " **WINNER**" if r["sample_id"] == best_rec["sample_id"] else ""
            f.write(
                f"| {r['sample_id']} | {r['NP']} | {r['CR']:.2f} "
                f"| {r['F']:.2f} | {r['lambda']:.2f} "
                f"| {r['best_f']:.3e} | {r['k_convergence']} "
                f"| {r['n_evaluations']} | {r['runtime_s']:.1f} |{flag}\n"
            )
        f.write(f"\n## Winner\n\n")
        f.write(f"- NP     = {best_rec['NP']}\n")
        f.write(f"- CR     = {best_rec['CR']}\n")
        f.write(f"- F      = {best_rec['F']}\n")
        f.write(f"- lambda = {best_rec['lambda']}\n")
        f.write(f"- best_f = {best_rec['best_f']:.4e}\n")
        f.write(f"- k_convergence = {best_rec['k_convergence']}\n")

    print(f"\n[{out_json}] written.")
    print(f"[{out_md}] written.")
    print(f"Winner: NP={best_rec['NP']}, CR={best_rec['CR']}, "
          f"F={best_rec['F']}, lambda={best_rec['lambda']}, "
          f"best_f={best_rec['best_f']:.3e}")
    return out_json


# ================================================================================
# CLI entrypoint
# ================================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",     type=int, default=30,
                   help="Number of LHS samples (default: 30).")
    p.add_argument("--k_max", type=int, default=100,
                   help="DE iterations per sample (default: 100).")
    p.add_argument("--seed",  type=int, default=42,
                   help="LHS and DE random seed (default: 42).")
    p.add_argument("--out",   default="de_lhs",
                   help="Artifact file stem under artifacts/.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_lhs(
        n_samples = args.n,
        k_max     = args.k_max,
        seed      = args.seed,
        out_stem  = args.out,
    )
