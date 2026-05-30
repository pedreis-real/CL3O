'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Structural Group Sensitivity Analysis Tool.

ANOVA-based sensitivity analysis for the CL3O design vector. Perturbs each
structural group independently around a reference design (centre of bounds),
evaluates the full CL3O fitness for each perturbation, and applies a one-way
ANOVA to identify which group has the highest influence on z(X) = m(X) + P(X).

Structural groups and their design-vector slices (n = wing.n_cpts):
    Longarinas   xw1[0:n]  + xw2[n:2n]             spar chord positions
    Flanges      bf_roots[2n:2n+4] + tpr[2n+4:3n+3] flange width and taper
    Revestimento ls1[3n+3:4n+3] + ls2[4n+3:5n+3]   skin layup indices
    Almas        lw1[5n+3:6n+3] + lw2[6n+3:7n+3]   web layup indices
    Mesas        lf1..lf4 [7n+3:11n+3]              flange layup indices

For each group, _N_PERT Latin-Hypercube perturbations are generated within
the group bounds while all other groups are held at X_ref. The fitness
values from each group form one ANOVA treatment level.

ANOVA metrics reported per group:
    SS_within    residual variance within the group (intra-group sensitivity)
    eta_sq       effect size  SS_between / SS_total
    F_stat       one-way ANOVA F-statistic (scipy.stats.f_oneway)

Usage:
    python -m tools.sensitivity_analysis
    python -m tools.sensitivity_analysis --npert 30

Outputs written to tools/output/sensitivity/:
    anova_results.csv       ANOVA table with per-group statistics
    sensitivity_bar.png     eta^2 and coefficient-of-variation bar chart
    boxplots.png            fitness distributions per group
    convergence_ref.png     fitness components at X_ref for reference

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
import matplotlib.patches as mpatches
from scipy.stats import f_oneway

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

# Main
from cl3o.main import RunCLEO, _resolve_db_specs, DatabaseSpec, MainHelpers


# ================================================================================
# Configuration
# ================================================================================

_AIRCRAFT = "DA62"
_N_PERT   = 20       # perturbations per structural group (override with --npert)
_OUT_DIR  = Path(__file__).resolve().parent / "output" / "sensitivity"

_GROUP_NAMES  = ["Longarinas", "Flanges", "Revestimento", "Almas", "Mesas"]
_GROUP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


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
        aircraft_name  = _AIRCRAFT,
        opt_name       = "sensitivity",
        db_specs       = db_specs,
        de_hyperpar    = {**DE_HYPERPAR, "NP": 4, "k_max": 1},
        runner_options = {"pipeline_logging": False, "enable_logging": False,
                          "live_plot": False},
    )


# ================================================================================
# Group slice builder
# ================================================================================

def _get_groups(n: int) -> dict[str, tuple[slice, bool]]:
    '''
    Return the design-vector slice and discrete flag for each structural group.

    The layout mirrors fobjective._decode_design_vector (total = 11*n + 3):
        xw1     [0,    n)     xw2     [n,    2n)
        bf_roots[2n,   2n+4)  tpr     [2n+4, 3n+3)
        ls1     [3n+3, 4n+3)  ls2     [4n+3, 5n+3)
        lw1     [5n+3, 6n+3)  lw2     [6n+3, 7n+3)
        lf1..lf4[7n+3, 11n+3)

    Args:
        n: Number of spanwise control points (wing_db.n_cpts).

    Returns:
        Dict mapping group name to (slice, is_discrete). is_discrete=True
        for layup-index variables; the LHC sampler will round them.
    '''
    return {
        "Longarinas":   (slice(0,        2*n),    False),
        "Flanges":      (slice(2*n,      3*n+3),  False),
        "Revestimento": (slice(3*n+3,    5*n+3),  True),
        "Almas":        (slice(5*n+3,    7*n+3),  True),
        "Mesas":        (slice(7*n+3,    11*n+3), True),
    }


# ================================================================================
# LHC perturbation sampler
# ================================================================================

def _lhc_group(
    n_pert   : int,
    lo_g     : np.ndarray,
    hi_g     : np.ndarray,
    discrete : bool,
    rng      : np.random.Generator,
) -> np.ndarray:
    '''
    Generate stratified LHC samples for a single group's sub-space.

    Args:
        n_pert:   Number of perturbations to generate.
        lo_g:     Lower bound sub-vector for this group.
        hi_g:     Upper bound sub-vector for this group.
        discrete: When True, samples are rounded to the nearest integer.
        rng:      NumPy random generator.

    Returns:
        (n_pert, len(lo_g)) array of sampled group values.
    '''
    k   = len(lo_g)
    cut = np.linspace(0.0, 1.0, n_pert + 1)
    u   = rng.uniform(size=(n_pert, k))
    pts = cut[:-1, None] + u * (cut[1:, None] - cut[:-1, None])
    for j in range(k):
        rng.shuffle(pts[:, j])
    samples = lo_g + pts * (hi_g - lo_g)
    if discrete:
        samples = np.round(samples).astype(float)
    return samples


# ================================================================================
# Evaluator wrapper
# ================================================================================

def _eval_safe(runner: RunCLEO, X: np.ndarray) -> float | None:
    '''
    Evaluate the full CL3O pipeline, returning None on any exception.

    Args:
        runner: RunCLEO instance with a ready evaluator.
        X:      Flat design vector.

    Returns:
        Scalar fitness z(X), or None if the evaluation raised an exception.
    '''
    try:
        return float(runner.evaluator(X))
    except Exception:
        return None


# ================================================================================
# Reference design
# ================================================================================

def _make_reference(lo: np.ndarray, hi: np.ndarray, n: int) -> np.ndarray:
    '''
    Build the reference design X_ref as the midpoint of the DE bounds.

    Discrete (layup) variables are rounded to the nearest integer.

    Args:
        lo: Lower bound vector.
        hi: Upper bound vector.
        n:  Number of control points (used to locate the layup block start).

    Returns:
        Reference design vector of shape (D,).
    '''
    X_ref = (lo + hi) / 2.0
    layup_start = 3 * n + 3
    X_ref[layup_start:] = np.round(X_ref[layup_start:])
    return X_ref


# ================================================================================
# Reference breakdown plot
# ================================================================================

def _plot_reference(runner: RunCLEO, X_ref: np.ndarray, out_dir: Path) -> None:
    '''
    Evaluate X_ref and save a bar chart of mass, penalty, and total fitness.

    Args:
        runner:  RunCLEO instance.
        X_ref:   Reference design vector.
        out_dir: Output directory.
    '''
    f_ref = _eval_safe(runner, X_ref)
    if f_ref is None:
        print("  [warn] reference design evaluation failed -- skipping ref plot.")
        return

    rt   = runner.runtime
    mass = float(rt.score.total)
    pen  = float(rt.penalty.total)
    feas = rt.penalty.is_feasible

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Mass  m(X)", "Penalty  P(X)", "Total  z(X)"],
        [mass, pen, f_ref],
        color=["#2ca02c", "#d62728", "#1f77b4"],
    )
    for bar, val in zip(bars, [mass, pen, f_ref]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(f_ref, 1.0),
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Value  [kg or --]")
    ax.set_title(f"Reference design  X_ref\nfeasible={feas}  |  z(X_ref)={f_ref:.4f}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "convergence_ref.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


# ================================================================================
# ANOVA plots
# ================================================================================

def _plot_sensitivity_bar(anova_rows: list[dict], out_dir: Path) -> None:
    '''
    Bar chart comparing eta^2 (left axis) and CV = std/mean (right axis).

    Args:
        anova_rows: Filtered list of ANOVA result dicts (n_valid > 0).
        out_dir:    Output directory.
    '''
    groups   = [r["group"]  for r in anova_rows]
    eta_sq   = [r["eta_sq"] for r in anova_rows]
    cv       = [
        r["std_f"] / r["mean_f"] if r["mean_f"] > 0 else 0.0
        for r in anova_rows
    ]

    x = np.arange(len(groups))
    w = 0.35
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.bar(x - w/2, eta_sq, width=w, color=_GROUP_COLORS,
            alpha=0.85, label="eta^2 (ANOVA effect size)")
    ax2 = ax1.twinx()
    ax2.bar(x + w/2, cv, width=w, color=_GROUP_COLORS,
            alpha=0.45, hatch="//", label="CV  std/mean")

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=15, ha="right")
    ax1.set_ylabel("eta^2  (proportion of total variance)", color="#1f77b4")
    ax2.set_ylabel("Coefficient of variation  std/mean",    color="gray")
    ax1.set_ylim(0, max(eta_sq) * 1.25 + 1e-6)
    ax2.set_ylim(0, max(cv)     * 1.25 + 1e-6)

    h1 = mpatches.Patch(color="gray", alpha=0.85,          label="eta^2 -- left axis")
    h2 = mpatches.Patch(color="gray", alpha=0.45, hatch="//", label="CV -- right axis")
    ax1.legend(handles=[h1, h2], fontsize=8, loc="upper right")

    ax1.set_title("Structural group sensitivity  --  one-way ANOVA")
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "sensitivity_bar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


def _plot_boxplots(
    group_fitness : dict[str, list[float]],
    f_ref         : float | None,
    out_dir       : Path,
) -> None:
    '''
    Box plots of fitness distributions for each structural group.

    Args:
        group_fitness: Dict mapping group name to list of fitness values.
        f_ref:         Reference design fitness (drawn as a dashed line).
        out_dir:       Output directory.
    '''
    groups = list(group_fitness.keys())
    data   = [group_fitness[g] for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=1.5))
    for patch, col in zip(bp["boxes"], _GROUP_COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    if f_ref is not None:
        ax.axhline(f_ref, ls="--", c="black", lw=1.2,
                   label=f"X_ref  ({f_ref:.3f})")
        ax.legend(fontsize=8)

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("Fitness  z(X)")
    ax.set_title(
        f"Fitness distribution per structural group  ({len(data[0])} pert. each)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "boxplots.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


# ================================================================================
# Entry point
# ================================================================================

def main() -> None:
    '''Parse arguments and run the structural group sensitivity analysis.'''
    parser = argparse.ArgumentParser(
        description="ANOVA-based structural group sensitivity for CL3O.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--npert", type=int, default=_N_PERT,
        help=f"Perturbations per group (default {_N_PERT}).",
    )
    args   = parser.parse_args()
    n_pert = int(args.npert)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CL3O databases ...")
    runner = _build_runner()

    lo = runner.static.opt_setup.data.lo
    hi = runner.static.opt_setup.data.hi
    n  = int(runner.static.wing_db.n_cpts)
    D  = int(lo.size)
    print(f"  wing n_cpts = {n}  |  D = {D}")

    groups = _get_groups(n)
    X_ref  = _make_reference(lo, hi, n)
    rng    = np.random.default_rng(42)

    print("\nEvaluating reference design X_ref ...")
    f_ref = _eval_safe(runner, X_ref)
    if f_ref is not None:
        rt = runner.runtime
        print(
            f"  z(X_ref) = {f_ref:.4f}  |  "
            f"mass={rt.score.total:.3f} kg  |  "
            f"penalty={rt.penalty.total:.3f}  |  "
            f"feasible={rt.penalty.is_feasible}"
        )
    _plot_reference(runner, X_ref, _OUT_DIR)

    group_fitness: dict[str, list[float]] = {}
    total_evals   = 0

    for gname, (slc, discrete) in groups.items():
        lo_g     = lo[slc]
        hi_g     = hi[slc]
        perturb  = _lhc_group(n_pert, lo_g, hi_g, discrete, rng)
        n_vars   = int(perturb.shape[1])
        var_type = "discrete" if discrete else "continuous"

        print(
            f"\n[{gname}]  {n_vars} vars  ({var_type})"
            f"  --  evaluating {n_pert} perturbations ..."
        )

        fitness_vals: list[float] = []
        n_fail = 0
        for sample_g in perturb:
            X = X_ref.copy()
            X[slc] = sample_g
            fval   = _eval_safe(runner, X)
            if fval is not None:
                fitness_vals.append(fval)
            else:
                n_fail += 1

        if n_fail:
            print(f"  [warn] {n_fail}/{n_pert} evaluations failed -- excluded from ANOVA.")

        if not fitness_vals:
            print(f"  [warn] all evaluations failed for {gname} -- skipping.")
            group_fitness[gname] = []
            continue

        group_fitness[gname] = fitness_vals
        total_evals          += len(fitness_vals)
        arr = np.array(fitness_vals)
        print(
            f"  n_valid={len(arr)}  mean={arr.mean():.4f}  "
            f"std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}"
        )

    # One-way ANOVA across all groups with at least 2 observations
    valid_groups = {g: v for g, v in group_fitness.items() if len(v) >= 2}
    all_fitness  = np.concatenate(list(valid_groups.values()))
    grand_mean   = float(all_fitness.mean())
    SS_total     = float(np.sum((all_fitness - grand_mean) ** 2))

    if len(valid_groups) >= 2:
        f_stat, p_val = f_oneway(*valid_groups.values())
    else:
        f_stat, p_val = float("nan"), float("nan")

    anova_rows: list[dict] = []
    for gname in _GROUP_NAMES:
        vals = group_fitness.get(gname, [])
        if not vals:
            anova_rows.append({
                "group": gname, "n_valid": 0,
                "mean_f": "nan", "std_f": "nan",
                "min_f":  "nan", "max_f": "nan",
                "SS_within": "nan", "eta_sq": "nan",
            })
            continue
        arr   = np.array(vals)
        gm    = float(arr.mean())
        SS_w  = float(np.sum((arr - gm) ** 2))
        SS_b  = float(len(arr) * (gm - grand_mean) ** 2)
        eta2  = SS_b / SS_total if SS_total > 0 else 0.0
        anova_rows.append({
            "group":     gname,
            "n_valid":   int(len(arr)),
            "mean_f":    round(gm,               4),
            "std_f":     round(float(arr.std()),  4),
            "min_f":     round(float(arr.min()),  4),
            "max_f":     round(float(arr.max()),  4),
            "SS_within": round(SS_w,              4),
            "eta_sq":    round(eta2,              6),
        })

    print(f"\n{'Group':<15} {'n':>4}  {'mean':>8}  {'std':>8}  "
          f"{'SS_within':>10}  {'eta^2':>8}")
    print("-" * 60)
    for r in anova_rows:
        print(
            f"{r['group']:<15} {r['n_valid']:>4}  {str(r['mean_f']):>8}  "
            f"{str(r['std_f']):>8}  {str(r['SS_within']):>10}  "
            f"{str(r['eta_sq']):>8}"
        )
    print(f"\nOne-way ANOVA: F = {f_stat:.4f},  p = {p_val:.4g}"
          f"  (total evals = {total_evals})")

    fieldnames = ["group", "n_valid", "mean_f", "std_f", "min_f",
                  "max_f", "SS_within", "eta_sq"]
    footer = [{
        "group": "ANOVA", "n_valid": total_evals,
        "mean_f": round(grand_mean, 4), "std_f": "",
        "min_f": "", "max_f": "",
        "SS_within": round(SS_total, 4),
        "eta_sq": f"F={f_stat:.4f} p={p_val:.4g}",
    }]

    csv_path = _OUT_DIR / "anova_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(anova_rows)
        writer.writerows(footer)
    print(f"\nANOVA table saved -> {csv_path}")

    _plot_sensitivity_bar(
        [r for r in anova_rows if r["n_valid"] > 0 and r["eta_sq"] != "nan"],
        _OUT_DIR,
    )
    _plot_boxplots(
        {g: v for g, v in group_fitness.items() if v},
        f_ref,
        _OUT_DIR,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
