'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Per-Variable Sensitivity Sweep Script.

Produces one-at-a-time (OAT) sensitivity curves for each of the three
variable groups reported in Section 4.5 of the monograph:

    4.5.1 Posicionamento das longarinas  (xw1, xw2)
    4.5.2 Flanges                        (bf1, bf2, bf3, bf4)
    4.5.3 Revestimento                   (ls1, ls2, lw1, lw2,
                                          lf1, lf2, lf3, lf4)

Each variable is swept across its DE bound interval while the remaining
design vector is held at a seed configuration. The seed is loaded from
the final optimum of artifacts/final_opt.json if present; otherwise the
midpoint of each DE bound is used.

Outputs per subsection:
    artifacts/sensitivity_<subsection>.json        raw data
    artifacts/sensitivity_<subsection>.png         one line per variable

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
import matplotlib.pyplot as plt

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ================ Default Database Paths ================
_OUT_DIR     = _ROOT / "artifacts"
_OUT_DIR.mkdir(exist_ok=True)
_PENALTY_JSON  = _OUT_DIR / "penalty_sweep.json"
_OPTIMUM_JSON  = _OUT_DIR / "final_opt.json"

# ================ Module imports ================
from main import (
    RunCL3O, DatabaseSpec, build_default_evaluator, _build_de_bounds,
    _CONTINUOUS_VARS, _DISCRETE_VARS,
    _DFLT_AFL_DIR, _DFLT_MAT_DIR, _DFLT_LDS_DIR, _DFLT_OPP_DIR, _DFLT_WNG_DIR,
)
from geometry.wing         import WingData
from geometry.airfoil      import AirfoilData
from materials.laminate    import LaminateData
from fem.loads.load_mapper import ExLoadsData
from utils.oppoints        import OppData

# ================ Global variables ================
_VAR_GROUPS = {
    "4.5.1_longarinas" : ["xw1", "xw2"],
    "4.5.2_flanges"    : ["bf1", "bf2", "bf3", "bf4"],
    "4.5.3_revestimento": ["ls1", "ls2", "lw1", "lw2",
                           "lf1", "lf2", "lf3", "lf4"],
}
_N_SAMPLES_DFLT = 15
_DFLT_PENALTY   = {"L": 4_000.0, "psi1": 0.10, "psi2": 0.90}


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


def _load_penalty() -> dict:
    if not _PENALTY_JSON.is_file():
        return dict(_DFLT_PENALTY)
    with _PENALTY_JSON.open(encoding="utf-8") as f:
        best = json.load(f).get("best_combo", {})
    return {
        "L"    : float(best.get("L",    _DFLT_PENALTY["L"])),
        "psi1" : float(best.get("psi1", _DFLT_PENALTY["psi1"])),
        "psi2" : float(best.get("psi2", _DFLT_PENALTY["psi2"])),
    }


def _load_seed(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    '''Use the final-opt vector if present, else the midpoint of each bound.'''
    if _OPTIMUM_JSON.is_file():
        with _OPTIMUM_JSON.open(encoding="utf-8") as f:
            x = np.asarray(json.load(f).get("best_X", []), dtype=float)
        if x.size == lo.size:
            print(f"[sensitivity] Seed loaded from {_OPTIMUM_JSON.name}.")
            return x
    print(f"[sensitivity] No optimum artifact; using bound midpoints as seed.")
    return 0.5 * (lo + hi)


def _var_slices(n_cpts: int) -> dict[str, slice]:
    '''
    Map each design variable name to the contiguous slice of X it
    occupies. Layout matches _decode_design_vector in src/main.py:
        [ xw1 ... bf4 ] [ ls1 ... lf4 ]
    '''
    out    = {}
    cursor = 0
    for name in list(_CONTINUOUS_VARS) + list(_DISCRETE_VARS):
        out[name] = slice(cursor, cursor + n_cpts)
        cursor  += n_cpts
    return out


# ================================================================================
# PUBLIC API - Sensitivity sweep driver
# ================================================================================

def run_sensitivity(
    aircraft : str  = "DA62",
    airfoil  : str  = "WortmannFX63137",
    n_mat    : int  = 5,
    n_points : int  = _N_SAMPLES_DFLT,
) -> None:
    '''Drive the three sensitivity sweeps and emit per-subsection artifacts.'''
    penalty_kwargs = _load_penalty()
    penalty_kwargs["enable_logging"] = False

    runner  = RunCL3O(
        aircraft_name  = aircraft,
        opt_name       = "Sensitivity",
        db_specs       = _build_db_specs(aircraft, airfoil, n_mat),
        enable_logging = False,
    )
    lo, hi  = _build_de_bounds(runner.static)
    n_cpts  = int(runner.static.wing_db.n_cpts)
    seed    = _load_seed(lo, hi)
    slices  = _var_slices(n_cpts)

    evaluator = build_default_evaluator(
        static         = runner.static,
        runtime        = runner.runtime,
        penalty_kwargs = penalty_kwargs,
    )

    seed_score = float(evaluator(seed))
    print(f"[sensitivity] Seed score: {seed_score:.4e}")

    for group_tag, var_names in _VAR_GROUPS.items():
        t0 = time.time()
        group_data = {"group": group_tag, "seed_score": seed_score,
                      "n_points": n_points, "variables": {}}

        fig, ax = plt.subplots(figsize=(8, 5))

        for name in var_names:
            sl        = slices[name]
            v_lo, v_hi = float(lo[sl.start]), float(hi[sl.start])
            sweep_vals = np.linspace(v_lo, v_hi, n_points)
            scores     = np.zeros(n_points)

            for j, v in enumerate(sweep_vals):
                X            = seed.copy()
                X[sl]        = v
                scores[j]    = float(evaluator(X))
                print(
                    f"[{group_tag}] {name}={v:.4f} -> score={scores[j]:.3e}"
                )

            group_data["variables"][name] = {
                "sweep_values" : sweep_vals.tolist(),
                "scores"       : scores.tolist(),
                "bound_lo"     : v_lo,
                "bound_hi"     : v_hi,
            }

            ax.plot(sweep_vals, scores, marker="o", label=name)

        ax.axhline(
            seed_score, color="gray", linestyle="--",
            linewidth=0.8, label="seed",
        )
        ax.set_xlabel("Valor da variavel")
        ax.set_ylabel("TotalScore")
        ax.set_title(f"Sensibilidade - {group_tag}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        png_path  = _OUT_DIR / f"sensitivity_{group_tag}.png"
        json_path = _OUT_DIR / f"sensitivity_{group_tag}.json"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(group_data, f, indent=2)

        dt = time.time() - t0
        print(f"[{group_tag}] {len(var_names)} variable(s), {n_points} points/var, "
              f"dt={dt:.1f}s -> {png_path.name}, {json_path.name}")


# ================================================================================
# CLI entrypoint
# ================================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=int, default=_N_SAMPLES_DFLT,
                   help="Samples per variable (default: 15).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_sensitivity(n_points=args.points)
