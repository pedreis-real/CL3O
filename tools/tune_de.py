'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
DE Hyper-parameter Tuning Tool.

Latin Hypercube Sampling sweep of Differential Evolution hyper-parameters.

Generates _N_SAMPLES distinct (NP, CR, F, lambda) combinations via
stratified LHS, runs a short DE for each (k_max = _K_MAX_TUNE), and
records:

    best_f_final    penalised fitness of the best individual at last gen
    feasible_f      best penalty-free fitness found (inf if none found)
    n_gens          generations actually executed (may stop early)
    gen_to_half     first generation where best_f <= 0.5 * best_f[0]
    converged       True when std-collapse or stall fired before k_max

All samples share the same evaluator (one database load) and the same DE
bounds. Each sample gets its own RNG seed (BASE_SEED + sample_index).

Usage:
    python -m tools.tune_de
    python -m tools.tune_de --samples 15 --kmax 80

Outputs written to tools/output/tune_de/:
    lhs_samples.csv         the sampled hyper-parameter table
    results.csv             per-sample convergence metrics
    convergence_all.png     best-f curves overlaid for all samples
    parallel_coords.png     parallel-coordinates of (params -> best_f_final)

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ================ Pathing ================
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ================ Module imports ================

# Constants
from cl3o.Constants import DE_HYPERPAR

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
from cl3o.optimization.de_opt import SetupOpt, RunOpt

# Main
from cl3o.main import RunCLEO, _resolve_db_specs, DatabaseSpec, MainHelpers


# ================================================================================
# Configuration
# ================================================================================

_AIRCRAFT    = "DA62"
_N_SAMPLES   = 10      # LHS sample size (override with --samples)
_K_MAX_TUNE  = 60      # short DE budget per sample (override with --kmax)
_BASE_SEED   = 100     # seed offset; sample i uses _BASE_SEED + i
_OUT_DIR     = Path(__file__).resolve().parent / "output" / "tune_de"

# Ranges for each hyper-parameter
_PARAM_RANGES: dict[str, tuple] = {
    "NP":     (10,  80),    # population size (cast to int)
    "CR":     (0.5, 1.0),   # crossover probability
    "F":      (0.3, 1.2),   # differential weight
    "lambda": (0.0, 1.0),   # best-attraction weight
}


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


def _build_runner(k_max: int) -> RunCLEO:
    '''
    Load all static databases and build the CL3O evaluator.

    Args:
        k_max: DE generation budget stored in the base opt_setup (not used
               for the sweep runs; only sets up the evaluator and bounds).

    Returns:
        Initialised RunCLEO instance.
    '''
    db_specs = _build_specs()
    MainHelpers.verify_missing_database(db_specs)
    return RunCLEO(
        aircraft_name    = _AIRCRAFT,
        opt_name         = "tune_de",
        db_specs         = db_specs,
        pipeline_logging = False,
        enable_logging   = False,
        de_hyperpar      = {**DE_HYPERPAR, "NP": 4, "k_max": k_max},
    )


# ================================================================================
# LHS sampler
# ================================================================================

def _lhs_samples(
    n      : int,
    ranges : dict[str, tuple],
    rng    : np.random.Generator,
) -> list[dict]:
    '''
    Return n stratified Latin-Hypercube samples over the given parameter ranges.

    Args:
        n:      Number of samples to generate.
        ranges: Dict mapping parameter name to (lo, hi) tuple.
        rng:    NumPy random generator.

    Returns:
        List of dicts, each containing one value per parameter key.
    '''
    keys   = list(ranges.keys())
    k      = len(keys)
    cut    = np.linspace(0.0, 1.0, n + 1)
    u      = rng.uniform(size=(n, k))
    pts    = cut[:-1, None] + u * (cut[1:, None] - cut[:-1, None])
    for j in range(k):
        rng.shuffle(pts[:, j])

    samples: list[dict] = []
    for row in pts:
        s: dict = {}
        for key, frac in zip(keys, row):
            lo, hi = ranges[key]
            val = lo + frac * (hi - lo)
            s[key] = int(round(val)) if key == "NP" else float(val)
        s["NP"] = max(4, int(s["NP"]))
        samples.append(s)
    return samples


# ================================================================================
# Per-sample run helper
# ================================================================================

def _run_sample(
    idx    : int,
    params : dict,
    runner : RunCLEO,
    k_max  : int,
) -> dict:
    '''
    Execute one DE run with the given hyper-parameters and record metrics.

    Args:
        idx:    Sample index (used to derive the RNG seed).
        params: Dict with keys NP, CR, F, lambda.
        runner: Shared RunCLEO instance (evaluator and bounds are reused).
        k_max:  Maximum generations for this run.

    Returns:
        Dict of convergence metrics; includes "_best_f_hist" (list, not in CSV).
    '''
    lo = runner.static.opt_setup.data.lo
    hi = runner.static.opt_setup.data.hi

    hypar = {
        "NP":             params["NP"],
        "CR":             params["CR"],
        "F":              params["F"],
        "lambda":         params["lambda"],
        "k_max":          k_max,
        "seed":           _BASE_SEED + idx,
        "std_tol":        DE_HYPERPAR["std_tol"],
        "stall_patience": DE_HYPERPAR["stall_patience"],
    }

    setup = SetupOpt(
        evaluator      = runner.evaluator,
        de_hyperpar    = hypar,
        bounds_lo      = lo,
        bounds_hi      = hi,
        enable_logging = False,
    )

    def _is_feasible(X: np.ndarray) -> bool:
        runner.evaluator(X)
        return bool(runner.runtime.fitness.is_feasible)

    run  = RunOpt(
        setup          = setup,
        feasible_check = _is_feasible,
        enable_logging = False,
    )
    hist = run.history
    bf   = hist.best_f
    f0   = float(bf[0]) if len(bf) > 0 else float("inf")
    half = f0 * 0.5

    gen_half = next(
        (int(g) for g, fv in enumerate(bf) if fv <= half),
        int(hist.ng),
    )

    return {
        "sample_idx":   idx,
        "NP":           params["NP"],
        "CR":           round(params["CR"],     4),
        "F":            round(params["F"],      4),
        "lambda":       round(params["lambda"], 4),
        "k_max_budget": k_max,
        "n_gens":       int(hist.ng),
        "converged":    bool(hist.ng < k_max),
        "best_f_final": round(float(bf[hist.ng]), 4) if len(bf) > 0 else float("nan"),
        "feasible_f":   round(float(hist.feasible_f), 4),
        "gen_to_half":  gen_half,
        "_best_f_hist": bf.tolist(),
    }


# ================================================================================
# Plots
# ================================================================================

def _plot_convergence(
    results     : list[dict],
    params_list : list[dict],
    k_max       : int,
    out_dir     : Path,
) -> None:
    '''
    Plot best-f convergence curves overlaid for all LHS samples.

    Args:
        results:     List of per-sample result dicts.
        params_list: Corresponding hyper-parameter dicts.
        k_max:       Generation budget (for title).
        out_dir:     Output directory.
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = cm.tab10(np.linspace(0, 1, len(results)))

    for res, par, col in zip(results, params_list, colors):
        hist  = res["_best_f_hist"]
        gens  = np.arange(len(hist))
        label = (
            f"#{res['sample_idx']}  NP={par['NP']}  "
            f"CR={par['CR']:.2f}  F={par['F']:.2f}  "
            f"lam={par['lambda']:.2f}"
        )
        ax.semilogy(gens, hist, color=col, lw=1.0, label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness  z(X)  [log scale]")
    ax.set_title(f"DE convergence -- {len(results)} LHS samples  (k_max={k_max})")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    p = out_dir / "convergence_all.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


def _plot_parallel_coords(results: list[dict], out_dir: Path) -> None:
    '''
    Plot parallel coordinates over (NP, CR, F, lambda, best_f_final).

    Args:
        results: List of per-sample result dicts.
        out_dir: Output directory.
    '''
    keys = ["NP", "CR", "F", "lambda", "best_f_final"]
    data = np.array([[r[k] for k in keys] for r in results], dtype=float)

    ndata       = np.zeros_like(data)
    ranges_ax : list[tuple] = []
    for j in range(data.shape[1]):
        lo_j, hi_j = data[:, j].min(), data[:, j].max()
        rng_j = hi_j - lo_j if hi_j > lo_j else 1.0
        ndata[:, j] = (data[:, j] - lo_j) / rng_j
        ranges_ax.append((lo_j, hi_j))

    norm   = plt.Normalize(data[:, -1].min(), data[:, -1].max())
    cmap   = plt.cm.RdYlGn_r
    colors = cmap(norm(data[:, -1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    n_axes  = len(keys)
    for i, row in enumerate(ndata):
        ax.plot(range(n_axes), row, color=colors[i], lw=1.2, alpha=0.8)
        ax.scatter(range(n_axes), row, color=colors[i], s=20, zorder=3)

    ax.set_xticks(range(n_axes))
    ax.set_xticklabels(keys)
    ax.set_yticks([])
    ax.set_ylim(-0.05, 1.05)

    for j, (lo_j, hi_j) in enumerate(ranges_ax):
        ax.text(j, -0.08, f"{lo_j:.3g}", ha="center", va="top",    fontsize=7)
        ax.text(j,  1.08, f"{hi_j:.3g}", ha="center", va="bottom", fontsize=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="best_f_final", shrink=0.6)
    ax.set_title("Parallel coordinates -- LHS hyper-parameter sweep")
    ax.grid(axis="x", alpha=0.4)

    p = out_dir / "parallel_coords.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


# ================================================================================
# Entry point
# ================================================================================

def main() -> None:
    '''Parse arguments and run the LHS DE hyper-parameter sweep.'''
    parser = argparse.ArgumentParser(
        description="LHS sweep of DE hyper-parameters for CL3O.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=_N_SAMPLES,
                        help=f"Number of LHS samples (default {_N_SAMPLES}).")
    parser.add_argument("--kmax",    type=int, default=_K_MAX_TUNE,
                        help=f"DE generation budget per sample (default {_K_MAX_TUNE}).")
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng         = np.random.default_rng(0)
    params_list = _lhs_samples(args.samples, _PARAM_RANGES, rng)

    csv_lhs = _OUT_DIR / "lhs_samples.csv"
    with open(csv_lhs, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_PARAM_RANGES.keys()))
        writer.writeheader()
        writer.writerows(params_list)
    print(f"LHS samples saved -> {csv_lhs}")
    print(f"\n{'Idx':>4}  {'NP':>4}  {'CR':>6}  {'F':>6}  {'lam':>6}")
    print("-" * 34)
    for i, p in enumerate(params_list):
        print(f"{i:>4}  {p['NP']:>4}  {p['CR']:>6.3f}  {p['F']:>6.3f}  "
              f"{p['lambda']:>6.3f}")

    print(f"\nLoading CL3O databases ...")
    runner = _build_runner(k_max=args.kmax)

    results: list[dict] = []
    for idx, params in enumerate(params_list):
        print(
            f"\n[{idx+1}/{args.samples}] NP={params['NP']}  CR={params['CR']:.3f}"
            f"  F={params['F']:.3f}  lam={params['lambda']:.3f}  ...",
            end=" ", flush=True,
        )
        try:
            res = _run_sample(idx, params, runner, args.kmax)
            results.append(res)
            feas_tag = "feasible" if res["feasible_f"] < 1e10 else "--"
            stop_tag = " (early stop)" if res["converged"] else ""
            print(f"best_f={res['best_f_final']:.4f}  gens={res['n_gens']}"
                  f"  {feas_tag}{stop_tag}")
        except Exception as exc:
            print(f"ERROR: {exc}")

    if not results:
        print("No successful runs. Exiting.")
        return

    csv_cols = [k for k in results[0].keys() if not k.startswith("_")]
    csv_res  = _OUT_DIR / "results.csv"
    with open(csv_res, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_cols})
    print(f"\nResults saved -> {csv_res}")

    _plot_convergence(results, params_list, args.kmax, _OUT_DIR)
    _plot_parallel_coords(results, _OUT_DIR)

    print(f"\n{'#':>3}  {'NP':>4}  {'CR':>5}  {'F':>5}  {'lam':>5}  "
          f"{'gens':>5}  {'best_f':>10}  {'feas_f':>10}  {'conv':>5}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x["best_f_final"]):
        feas = f"{r['feasible_f']:.4f}" if r["feasible_f"] < 1e10 else "   --"
        print(
            f"{r['sample_idx']:>3}  {r['NP']:>4}  {r['CR']:>5.3f}  "
            f"{r['F']:>5.3f}  {r['lambda']:>5.3f}  "
            f"{r['n_gens']:>5}  {r['best_f_final']:>10.4f}  "
            f"{feas:>10}  {'Y' if r['converged'] else 'N':>5}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
