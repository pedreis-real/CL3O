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

Each sample runs in an isolated child process (recycled every --batch
samples) so peak RAM tracks a single sample rather than the whole sweep,
and results.csv is flushed after every sample so a crash mid-sweep keeps
all prior results.

Usage:
    python -m tools.tune_de
    python -m tools.tune_de --samples 15 --kmax 80
    python -m tools.tune_de --batch 4          # recycle child every 4 samples

Outputs written to tools/output/tune_de/:
    lhs_samples.csv         the sampled hyper-parameter table
    results.csv             per-sample convergence metrics (written live)
    convergence_all.png     best-f curves overlaid for all samples
    parallel_coords.png     parallel-coordinates of (params -> best_f_final)

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import argparse
import csv
import gc
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from cl3o.paths import DATA_DIR, OUTPUTS_DIR as _CLEO_OUT_DIR
from cl3o.utils.oppoints import OppData
from cl3o.utils.database_utils import discover_laminates

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
_N_SAMPLES   = 20      # LHS sample size (override with --samples)
_K_MAX_TUNE  = 200
_BASE_SEED   = 67      # seed offset; sample i uses _BASE_SEED + i
_SWEEP_NAME  = "tune-de-6"
_OUT_DIR     = Path(__file__).resolve().parent / "output" / _SWEEP_NAME

# Samples run in isolated child processes so the OS fully reclaims each
# sample's RAM (caches, numpy buffers, fragmented arenas) when the child
# exits. _BATCH controls how many samples a child handles before it is
# recycled: 1 = a fresh interpreter per sample (max isolation, reloads the
# database each time); higher amortizes the database load at the cost of
# letting that many samples' peak RAM coexist. The in-process geometry/beam
# caches are also bounded LRUs (Constants.GEOM/BEAM_CACHE_MAXSIZE), so RAM is
# capped even within a single long sample.
_BATCH = 1     # samples per child process (override with --batch)

# Ranges for each hyper-parameter
_PARAM_RANGES: dict[str, tuple] = {
    "NP":     (16,  80),    # population size (cast to int)
    "CR":     (0.5, 1.0),   # crossover probability
    "F":      (0.3, 1.5),   # differential weight
    "lambda": (0.0, 1.0),   # best-attraction weight
}


# ================================================================================
# Database helpers
# ================================================================================

def _build_specs() -> list:
    '''Return the resolved DatabaseSpec list for the DA62 configuration.'''
    mat_dir = DATA_DIR / "materials"
    materials = discover_laminates(mat_dir)
    specs: list = []
    # specs.append(DatabaseSpec(WingData,    DATA_DIR / "wings",    _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(WingData,    DATA_DIR / "wings",    f"{_AIRCRAFT}_simplified".lower()))
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
        aircraft_name  = _AIRCRAFT,
        opt_name       = "tune_de",
        db_specs       = db_specs,
        de_hyperpar    = DE_HYPERPAR,
        runner_options = {"pipeline_logging": False, "enable_logging": False,
                          "live_plot": False},
    )


# ================================================================================
# Cache management
# ================================================================================

def _reset_pipeline_caches(runner: RunCLEO) -> None:
    '''
    Clear the shared geometry/beam memoization caches on the runner.

    Both caches live on the persistent runner and are reused across every
    sample of the sweep, so they accumulate for the whole run unless cleared.
    They MUST be cleared together: beam_cache keys are id() values of the
    GeomData objects kept alive by geom_cache, so clearing geom_cache alone
    would let those ids be reused by new objects and produce false cache hits.

    Args:
        runner: Shared RunCLEO instance owning the static caches.
    '''
    runner.static.geom_cache.clear()
    runner.static.fem_setup.beam_cache.clear()


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
    idx     : int,
    params  : dict,
    runner  : RunCLEO,
    k_max   : int,
    out_dir : Path | None = None,
) -> dict:
    '''
    Execute one DE run with the given hyper-parameters and record metrics.

    When `out_dir` is provided the function also:
      - pickles the last-generation RuntimeData to
        ``out_dir/gen_<n_gens:04d>.pkl``;
      - writes ``out_dir/rate.csv`` with per-generation convergence rate.

    Args:
        idx:     Sample index (used to derive the RNG seed).
        params:  Dict with keys NP, CR, F, lambda.
        runner:  Shared RunCLEO instance (evaluator and bounds are reused).
        k_max:   Maximum generations for this run.
        out_dir: Optional per-sample output directory.

    Returns:
        Dict of convergence metrics; includes "_best_f_hist" (list, not in CSV).
    '''
    lo = runner.static.opt_setup.data.lo
    hi = runner.static.opt_setup.data.hi

    # Start every sample from empty caches so independent runs never
    # accumulate each other's geometry/beam entries.
    _reset_pipeline_caches(runner)
    t_start = time.perf_counter()

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

    # The geometry/beam caches are bounded LRUs (Constants.*_CACHE_MAXSIZE),
    # so they self-evict the least-recently-used entries and no longer need a
    # manual per-generation ceiling check.
    run  = RunOpt(
        setup          = setup,
        feasible_check = _is_feasible,
        on_generation  = False,
        enable_logging = True,
        # out_dir = out_dir,
    )
    hist = run.history
    bf   = hist.best_f

    # Extract all needed scalars/lists before freeing the DE objects.
    n_gens       = int(hist.ng)
    converged    = bool(hist.ng < k_max)
    best_f_final = round(float(bf[n_gens]), 4) if len(bf) > 0 else float("nan")
    feasible_f   = round(float(hist.feasible_f), 4)
    bf_list      = bf.tolist()
    best_X_last  = hist.best_X[n_gens].copy()

    f0       = float(bf[0]) if len(bf) > 0 else float("inf")
    gen_half = next(
        (int(g) for g, fv in enumerate(bf) if fv <= f0 * 0.5),
        n_gens,
    )

    elapsed_s = round(time.perf_counter() - t_start, 2)

    # Persist last-generation snapshot and rate CSV when requested.
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        runner.evaluator(best_X_last)   # ensure runtime reflects the best design
        pkl_path = out_dir / f"gen_{n_gens:04d}.pkl"
        with open(pkl_path, "wb") as _fh:
            pickle.dump(runner.runtime, _fh, protocol=pickle.HIGHEST_PROTOCOL)
        _save_rate_csv(out_dir, bf_list, n_gens, converged)

    # Release the DE population / history arrays now that all metrics are captured.
    del bf, hist, run, setup

    return {
        "sample_idx":   idx,
        "NP":           params["NP"],
        "CR":           round(params["CR"],     4),
        "F":            round(params["F"],      4),
        "lambda":       round(params["lambda"], 4),
        "k_max_budget": k_max,
        "n_gens":       n_gens,
        "converged":    converged,
        "best_f_final": best_f_final,
        "feasible_f":   feasible_f,
        "gen_to_half":  gen_half,
        "elapsed_s":    elapsed_s,
        "_best_f_hist": bf_list,
    }


# ================================================================================
# Subprocess worker and live result persistence
# ================================================================================

# CSV column order for results.csv (excludes the leading "_best_f_hist" key,
# which is only used in-memory for the convergence plot).
_RESULT_COLS = [
    "sample_idx", "NP", "CR", "F", "lambda", "k_max_budget",
    "n_gens", "converged", "best_f_final", "feasible_f",
    "gen_to_half", "elapsed_s",
]

# Per-child lazily-built runner. Each child process builds the runner (one
# database load) on its first task and reuses it for every subsequent task it
# handles, until the pool recycles the child (max_tasks_per_child = --batch),
# at which point the OS reclaims all of its memory.
_WORKER_RUNNER: RunCLEO | None = None


def _worker(payload: tuple) -> dict:
    '''
    Child-process entry point: build/reuse a runner and run one sample.

    Args:
        payload: Tuple (sample_idx, params_dict, k_max, sweep_name).

    Returns:
        The per-sample result dict from _run_sample (picklable; the
        "_best_f_hist" list survives the process boundary for plotting).
    '''
    global _WORKER_RUNNER
    idx, params, k_max, sweep_name = payload
    if _WORKER_RUNNER is None:
        _WORKER_RUNNER = _build_runner(k_max=k_max)
    sample_dir = (
        _CLEO_OUT_DIR / f"{_AIRCRAFT.lower()}_{sweep_name}_LHS-{idx}"
    )
    res = _run_sample(idx, params, _WORKER_RUNNER, k_max, out_dir=sample_dir)
    # Help the child release this sample's transient allocations before the
    # next task it may handle within the same (un-recycled) process.
    _reset_pipeline_caches(_WORKER_RUNNER)
    gc.collect()
    return res


def _append_result_row(csv_path: Path, res: dict) -> None:
    '''
    Append one result row to results.csv, writing the header if the file is new.

    Flushed immediately so a crash mid-sweep keeps every completed sample.

    Args:
        csv_path: Destination results.csv path.
        res:      One per-sample result dict.
    '''
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULT_COLS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: res[k] for k in _RESULT_COLS})


def _save_rate_csv(
    out_dir   : Path,
    bf_list   : list,
    n_gens    : int,
    converged : bool,
) -> None:
    '''
    Write per-generation rate-of-convergence CSV for one LHS sample.

    Columns:
        k       generation index (0-based)
        best_f  best penalised fitness at generation k
        rate    relative decrease vs previous generation:
                (best_f[k-1] - best_f[k]) / |best_f[k-1]|  (0.0 at k=0)
        conv    Y on the generation that triggered early stop, N otherwise

    Args:
        out_dir  : Destination directory (per-sample).
        bf_list  : best_f history list of length n_gens + 1.
        n_gens   : Number of generations actually executed.
        converged: True when the run stopped before k_max.
    '''
    csv_path = out_dir / "rate.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "best_f", "rate", "conv"])
        writer.writeheader()
        for k in range(n_gens + 1):
            bf = float(bf_list[k])
            if k == 0:
                rate = 0.0
            else:
                prev = float(bf_list[k - 1])
                rate = (prev - bf) / max(abs(prev), 1e-12)
            conv = "Y" if (k == n_gens and converged) else "N"
            writer.writerow({
                "k":      k,
                "best_f": round(bf,   6),
                "rate":   round(rate, 6),
                "conv":   conv,
            })


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
    parser.add_argument("--batch",   type=int, default=_BATCH,
                        help=f"Samples per child process before it is recycled "
                             f"(default {_BATCH}; 1 = fresh interpreter per sample).")
    parser.add_argument("--name",    type=str, default=_SWEEP_NAME,
                        help=f"Sweep name used for output subdirectory names "
                             f"(default '{_SWEEP_NAME}').")
    parser.add_argument("--seed",    type=int, default=_BASE_SEED,
                        help="RNG seed for LHS sampling (default 0).")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "output" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng         = np.random.default_rng(args.seed)
    params_list = _lhs_samples(args.samples, _PARAM_RANGES, rng)

    csv_lhs = out_dir / "lhs_samples.csv"
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

    # Start each sweep from a clean results.csv; rows are appended live as
    # each sample completes (see _append_result_row).
    csv_res = out_dir / "results.csv"
    if csv_res.exists():
        csv_res.unlink()

    print(f"\nRunning {args.samples} samples in isolated child processes "
          f"(recycle every {args.batch}); databases load once per child ...")

    # max_workers=1 keeps a single heavy FEA job in flight at a time (peak RAM
    # = one sample). max_tasks_per_child=--batch recycles the child so the OS
    # reclaims its memory. "spawn" is the cross-platform-safe start method.
    ctx     = mp.get_context("spawn")
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers         = 1,
        max_tasks_per_child = max(1, args.batch),
        mp_context          = ctx,
    ) as ex:
        fut_to_idx = {
            ex.submit(_worker, (idx, params, args.kmax, args.name)): idx
            for idx, params in enumerate(params_list)
        }
        for fut in as_completed(fut_to_idx):
            idx    = fut_to_idx[fut]
            params = params_list[idx]
            try:
                res = fut.result()
            except Exception as exc:
                print(f"[{idx+1}/{args.samples}] NP={params['NP']} "
                      f"CR={params['CR']:.3f} F={params['F']:.3f} "
                      f"lam={params['lambda']:.3f}  ERROR: {exc}")
                continue
            results.append(res)
            _append_result_row(csv_res, res)   # live save after each sample
            feas_tag = "feasible" if res["feasible_f"] < 1e10 else "--"
            stop_tag = " (early stop)" if res["converged"] else ""
            print(f"[{idx+1}/{args.samples}] NP={params['NP']} "
                  f"CR={params['CR']:.3f} F={params['F']:.3f} "
                  f"lam={params['lambda']:.3f}  best_f={res['best_f_final']:.4f} "
                  f"gens={res['n_gens']} {feas_tag}{stop_tag} "
                  f"{res['elapsed_s']:.1f}s  -> saved")

    if not results:
        print("No successful runs. Exiting.")
        return

    # Restore sample order (as_completed yields in completion order) so plots
    # line up with params_list regardless of pool scheduling.
    results.sort(key=lambda r: r["sample_idx"])
    plot_params = [params_list[r["sample_idx"]] for r in results]
    print(f"\nResults saved -> {csv_res}")

    _plot_convergence(results, plot_params, args.kmax, out_dir)
    _plot_parallel_coords(results, out_dir)

    print(f"\n{'#':>3}  {'NP':>4}  {'CR':>5}  {'F':>5}  {'lam':>5}  "
          f"{'gens':>5}  {'best_f':>10}  {'feas_f':>10}  {'conv':>5}  {'time(s)':>8}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x["best_f_final"]):
        feas = f"{r['feasible_f']:.4f}" if r["feasible_f"] < 1e10 else "   --"
        print(
            f"{r['sample_idx']:>3}  {r['NP']:>4}  {r['CR']:>5.3f}  "
            f"{r['F']:>5.3f}  {r['lambda']:>5.3f}  "
            f"{r['n_gens']:>5}  {r['best_f_final']:>10.4f}  "
            f"{feas:>10}  {'Y' if r['converged'] else 'N':>5}  {r['elapsed_s']:>8.1f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
