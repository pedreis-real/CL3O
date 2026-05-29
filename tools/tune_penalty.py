'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Penalty Tuning Tool.

Parameter sweep for the logistic penalty function P(v). Runs in two phases:

Phase 1 (always): plots P(v) curve families and (k, v0) heat-maps for a
(psi1 x psi2) grid at fixed Pcap, v1, v2, nv_test.

Phase 2 (default, skip with --no-lhc): evaluates _N_LHC random Latin-
Hypercube design vectors through the full CL3O pipeline, collects their
(FailureData, DisplacementData) responses, then re-applies the penalty
formula across the same parameter grid without re-running FEA. Outputs a
heat-map of feasible-candidate fraction and mean penalty per (psi1, psi2)
cell.

Usage:
    python -m tools.tune_penalty           # both phases
    python -m tools.tune_penalty --no-lhc  # analytical only

Outputs written to tools/output/tune_penalty/:
    Pv_curves.png           P(v) families for varying psi1 / psi2
    kv0_heatmap.png         derived k and v0 over the (psi1, psi2) grid
    feasibility_heatmap.png frac feasible and mean penalty from LHC candidates
    sweep_results.csv       tabular version of the heatmap data

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ================ Pathing ================
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ================ Module imports ================

# Constants
from cl3o.Constants import PENALTY_VARS, DE_HYPERPAR

# Utilities
from cl3o.paths import DATA_DIR
from cl3o.utils.oppoints import OppData

# Geometry
from cl3o.geometry.wing    import WingData
from cl3o.geometry.airfoil import AirfoilData

# Materials
from cl3o.materials.laminate import LaminateData

# FEA
from cl3o.fea.loads.load_mapper import ExLoadsData, InLoadsData

# Optimization
from cl3o.optimization.fpenalty import Penalty

# Main
from cl3o.main import RunCLEO, _resolve_db_specs, DatabaseSpec, MainHelpers


# ================================================================================
# Configuration
# ================================================================================

_AIRCRAFT   = "DA62"
_N_LHC      = 40        # random candidates evaluated in phase 2
_GRID_N     = 6         # grid points per axis in (psi1, psi2) sweep
_OUT_DIR    = Path(__file__).resolve().parent / "output" / "tune_penalty"

_PSI1_RANGE = (0.05, 0.40)
_PSI2_RANGE = (0.60, 0.95)
_PCAP       = PENALTY_VARS["Pcap"]
_V1         = PENALTY_VARS["v1"]
_V2         = PENALTY_VARS["v2"]
_NV         = PENALTY_VARS["nv_test"]


# ================================================================================
# Database helpers
# ================================================================================

def _build_specs() -> list:
    '''Return the resolved DatabaseSpec list for the DA62 configuration.'''
    mat_dir = DATA_DIR / "materials"
    materials = sorted(
        f.stem.removesuffix("_LaminateData")
        for f in mat_dir.glob("MAT_*_LaminateData.json")
    )
    specs: list = []
    specs.append(DatabaseSpec(WingData,    DATA_DIR / "wings",    _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(AirfoilData, DATA_DIR / "airfoils", "wortmannfx63137"))
    for mat in materials:
        specs.append(DatabaseSpec(LaminateData, mat_dir, mat))
    specs.append(DatabaseSpec(OppData,     DATA_DIR / "oppoints", _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(ExLoadsData, DATA_DIR / "loads",    _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(InLoadsData, DATA_DIR / "loads",    _AIRCRAFT.lower()))
    return _resolve_db_specs(specs)


def _build_runner() -> RunCLEO:
    '''Load all static databases and build the CL3O evaluator.'''
    db_specs = _build_specs()
    MainHelpers.verify_missing_database(db_specs)
    return RunCLEO(
        aircraft_name    = _AIRCRAFT,
        opt_name         = "tune_penalty",
        db_specs         = db_specs,
        pipeline_logging = False,
        enable_logging   = False,
        de_hyperpar      = {**DE_HYPERPAR, "NP": 4, "k_max": 1},
    )


# ================================================================================
# Math primitives (mirror fpenalty.py; no import side-effects)
# ================================================================================

def _logit(p: float) -> float:
    '''Inverse sigmoid: ln(p / (1 - p)).'''
    return math.log(p / (1.0 - p))


def _sigmoid_vec(z: np.ndarray) -> np.ndarray:
    '''Vectorised logistic function, numerically stable.'''
    z_clip = np.clip(z, -500.0, 500.0)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z_clip)),
        np.exp(z_clip) / (1.0 + np.exp(z_clip)),
    )


def _derive_k_v0(psi1: float, psi2: float) -> tuple[float, float]:
    '''
    Solve the 2x2 linear system for (k, v0) given (psi1, psi2).

    Args:
        psi1: P(v1) / Pcap fraction at violation v1.
        psi2: P(v2) / Pcap fraction at violation v2.

    Returns:
        Tuple (k, v0) — logistic slope and inflection violation.
    '''
    vv1 = _V1 * _NV
    vv2 = _V2 * _NV
    l1, l2 = _logit(psi1), _logit(psi2)
    denom   = l2 - l1
    return denom / (vv2 - vv1), (vv1 * l2 - vv2 * l1) / denom


def _penalty_vec(v: np.ndarray, k: float, v0: float) -> np.ndarray:
    '''
    Evaluate the normalised logistic penalty over an array of violation values.

    Args:
        v:  Violation severity array.
        k:  Logistic slope.
        v0: Inflection violation.

    Returns:
        P(v) array clipped to [0, Pcap].
    '''
    g_shift = float(_sigmoid_vec(np.array([-k * v0]))[0])
    denom   = 1.0 - g_shift
    if abs(denom) < 1e-15:
        return np.full_like(v, _PCAP)
    return np.clip(
        (_sigmoid_vec(k * (v - v0)) - g_shift) / denom * _PCAP,
        0.0, _PCAP,
    )


# ================================================================================
# Phase 1a - P(v) curve families
# ================================================================================

def plot_Pv_curves(out_dir: Path) -> None:
    '''
    Plot two P(v) families: one varying psi1 (psi2 fixed), one varying psi2.

    Args:
        out_dir: Directory where Pv_curves.png is written.
    '''
    psi1_vals = np.linspace(*_PSI1_RANGE, _GRID_N)
    psi2_vals = np.linspace(*_PSI2_RANGE, _GRID_N)
    v_arr     = np.linspace(0.0, _V2 * _NV * 3.5, 500)
    colors    = plt.cm.viridis(np.linspace(0.1, 0.9, _GRID_N))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    psi2_mid = float(np.median(psi2_vals))
    ax = axes[0]
    for psi1, col in zip(psi1_vals, colors):
        k, v0 = _derive_k_v0(psi1, psi2_mid)
        ax.plot(v_arr, _penalty_vec(v_arr, k, v0), color=col,
                label=f"psi1={psi1:.2f}")
    ax.axvline(_V1 * _NV, ls='--', c='gray', lw=0.8, label="v1")
    ax.axvline(_V2 * _NV, ls=':',  c='gray', lw=0.8, label="v2")
    ax.set_xlabel("v  (violation severity)")
    ax.set_ylabel("P(v)")
    ax.set_title(f"Varying psi1  (psi2 = {psi2_mid:.2f} fixed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    psi1_mid = float(np.median(psi1_vals))
    ax = axes[1]
    for psi2, col in zip(psi2_vals, colors):
        k, v0 = _derive_k_v0(psi1_mid, psi2)
        ax.plot(v_arr, _penalty_vec(v_arr, k, v0), color=col,
                label=f"psi2={psi2:.2f}")
    ax.axvline(_V1 * _NV, ls='--', c='gray', lw=0.8)
    ax.axvline(_V2 * _NV, ls=':',  c='gray', lw=0.8)
    ax.set_xlabel("v  (violation severity)")
    ax.set_ylabel("P(v)")
    ax.set_title(f"Varying psi2  (psi1 = {psi1_mid:.2f} fixed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Logistic penalty P(v)  --  Pcap={_PCAP:.0f},  nv_test={_NV}")
    fig.tight_layout()
    p = out_dir / "Pv_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


# ================================================================================
# Phase 1b - k and v0 heat-maps
# ================================================================================

def plot_kv0_heatmap(out_dir: Path) -> None:
    '''
    Plot k and v0 as heat-maps over the (psi1, psi2) parameter space.

    Args:
        out_dir: Directory where kv0_heatmap.png is written.
    '''
    n    = _GRID_N * 2
    psi1 = np.linspace(*_PSI1_RANGE, n)
    psi2 = np.linspace(*_PSI2_RANGE, n)
    K    = np.full((n, n), np.nan)
    V0   = np.full((n, n), np.nan)

    for i, p1 in enumerate(psi1):
        for j, p2 in enumerate(psi2):
            if p2 <= p1:
                continue
            try:
                K[i, j], V0[i, j] = _derive_k_v0(p1, p2)
            except (ValueError, ZeroDivisionError):
                pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, cmap in zip(
        axes, [K, V0],
        ["Logistic slope  k", "Inflection point  v0"],
        ["plasma", "cividis"],
    ):
        im = ax.imshow(
            data, origin="lower", aspect="auto", cmap=cmap,
            extent=[psi2[0], psi2[-1], psi1[0], psi1[-1]],
        )
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("psi2")
        ax.set_ylabel("psi1")
        ax.set_title(title)
        ax.scatter(
            [PENALTY_VARS["psi2"]], [PENALTY_VARS["psi1"]],
            marker="*", c="white", s=160, zorder=5, label="default",
        )
        ax.legend(fontsize=8)

    fig.suptitle(f"Derived penalty constants  (v1={_V1}, v2={_V2}, nv={_NV})")
    fig.tight_layout()
    p = out_dir / "kv0_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


# ================================================================================
# Phase 2 - LHC population re-scoring
# ================================================================================

def _lhc_population(
    NP  : int,
    lo  : np.ndarray,
    hi  : np.ndarray,
    rng : np.random.Generator,
) -> np.ndarray:
    '''
    Generate a stratified Latin-Hypercube population of shape (NP, D).

    Args:
        NP:  Number of candidates.
        lo:  Lower bound vector of length D.
        hi:  Upper bound vector of length D.
        rng: NumPy random generator.

    Returns:
        (NP, D) array of sampled design vectors.
    '''
    D   = lo.size
    cut = np.linspace(0.0, 1.0, NP + 1)
    u   = rng.uniform(size=(NP, D))
    pts = cut[:-1, None] + u * (cut[1:, None] - cut[:-1, None])
    for j in range(D):
        rng.shuffle(pts[:, j])
    return lo + pts * (hi - lo)


def run_lhc_rescore(runner: RunCLEO, out_dir: Path) -> None:
    '''
    Evaluate LHC candidates once, then re-score across the (psi1, psi2) grid.

    Args:
        runner:  Initialised RunCLEO instance with evaluator and bounds ready.
        out_dir: Directory for CSV and PNG outputs.
    '''
    lo  = runner.static.opt_setup.data.lo
    hi  = runner.static.opt_setup.data.hi
    rng = np.random.default_rng(0)
    X   = _lhc_population(_N_LHC, lo, hi, rng)

    print(f"  Evaluating {_N_LHC} LHC candidates through the full pipeline ...")
    tsw_store: list  = []
    disp_store: list = []
    for idx, xi in enumerate(X):
        try:
            runner.evaluator(xi)
            tsw_store.append(runner.runtime.tsw)
            disp_store.append(runner.runtime.displ)
        except Exception as exc:
            print(f"    [warn] candidate {idx}: {exc}")
            tsw_store.append(None)
            disp_store.append(None)

    psi1_vals = np.linspace(*_PSI1_RANGE, _GRID_N)
    psi2_vals = np.linspace(*_PSI2_RANGE, _GRID_N)
    feas_frac = np.full((_GRID_N, _GRID_N), np.nan)
    mean_pen  = np.full((_GRID_N, _GRID_N), np.nan)
    rows: list[dict] = []

    print(f"  Re-scoring across {_GRID_N}x{_GRID_N} (psi1, psi2) grid ...")
    for i, psi1 in enumerate(psi1_vals):
        for j, psi2 in enumerate(psi2_vals):
            if psi2 <= psi1:
                continue
            penalties: list[float] = []
            n_feas = 0
            for tsw, disp in zip(tsw_store, disp_store):
                if tsw is None:
                    continue
                try:
                    pd = Penalty(
                        data    = (tsw, disp),
                        Pcap    = _PCAP,
                        v1      = _V1,
                        v2      = _V2,
                        nv_test = _NV,
                        psi1    = float(psi1),
                        psi2    = float(psi2),
                        enable_logging = False,
                    ).data
                    penalties.append(pd.total)
                    n_feas += int(pd.is_feasible)
                except Exception:
                    pass
            if penalties:
                feas_frac[i, j] = n_feas / len(penalties)
                mean_pen[i, j]  = float(np.mean(penalties))
            rows.append({
                "psi1":         round(float(psi1), 4),
                "psi2":         round(float(psi2), 4),
                "feas_frac":    round(float(feas_frac[i, j]), 4)
                                if not np.isnan(feas_frac[i, j]) else "nan",
                "mean_penalty": round(float(mean_pen[i, j]),  4)
                                if not np.isnan(mean_pen[i, j])  else "nan",
            })

    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["psi1", "psi2", "feas_frac", "mean_penalty"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved -> {csv_path}")

    ext = [psi2_vals[0], psi2_vals[-1], psi1_vals[0], psi1_vals[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, cmap in zip(
        axes, [feas_frac, mean_pen],
        ["Feasible fraction", f"Mean penalty  (Pcap={_PCAP:.0f})"],
        ["YlGn", "YlOrRd"],
    ):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, extent=ext)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("psi2")
        ax.set_ylabel("psi1")
        ax.set_title(title)
        ax.scatter(
            [PENALTY_VARS["psi2"]], [PENALTY_VARS["psi1"]],
            marker="*", c="white", s=160, zorder=5, label="default",
        )
        ax.legend(fontsize=8)

    n_valid = sum(1 for t in tsw_store if t is not None)
    fig.suptitle(f"LHC re-score  --  {n_valid} valid candidates")
    fig.tight_layout()
    p = out_dir / "feasibility_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


# ================================================================================
# Entry point
# ================================================================================

def main() -> None:
    '''Parse arguments and run both phases of the penalty tuning tool.'''
    parser = argparse.ArgumentParser(
        description="Penalty-function parameter sweep for CL3O.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-lhc", action="store_true",
        help="Skip LHC re-scoring (analytical plots only).",
    )
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== tune_penalty  Phase 1 -- Analytical ===")
    plot_Pv_curves(_OUT_DIR)
    plot_kv0_heatmap(_OUT_DIR)

    if not args.no_lhc:
        print("\n=== tune_penalty  Phase 2 -- LHC re-scoring ===")
        runner = _build_runner()
        run_lhc_rescore(runner, _OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
