'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Laminate Coverage Diagnostic Module.

Post-optimization diagnostic that inspects how the curated laminate
catalogue is exercised by the DE search. Decodes the converged design
vector and the per-generation best trajectory into the eight role genes
(ls1, ls2, lw1, lw2, lf1..lf4), then reports:

    1. Selection frequency per (role x laminate)
    2. Dead options    - laminates never (or barely) selected
    3. Dominator entries - laminates that monopolise a role
    4. Mass / stiffness summary of selected laminates per role

The output is a markdown report suitable for inclusion in the thesis
appendix. Intended workflow:

    runner = RunCLEO(...).run(...)
    coverage_report(runner.static.opt_result, runner.static.laminate_db,
                    n_cpts=int(runner.static.wing_db.n_cpts))

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ================ Default Database Paths ================
from cl3o.paths import MATERIALS_DIR as _DFLT_MAT_DIR

# ================ Module imports ================

# Utilities
from cl3o.utils import io_utils as io

# Materials
from cl3o.materials.laminate import LaminateData

# Optimization
from cl3o.optimization.de_opt import HistoryData


# ========================================================================
# Module-level constants
# ========================================================================

ROLE_NAMES : tuple[str, ...] = (
    "ls1", "ls2",                       # skins
    "lw1", "lw2",                       # webs
    "lf1", "lf2", "lf3", "lf4",         # boom flanges
)
N_ROLES : int = len(ROLE_NAMES)

_DEAD_THRESHOLD      : float = 0.05    # < 5% selection across all roles
_DOMINATOR_THRESHOLD : float = 0.80    # > 80% selection within a role


# ========================================================================
# Internal helpers - design-vector decoding
# ========================================================================

class CoverageHelper:

    @staticmethod
    def role_block(X: np.ndarray, role_idx: int, n_cpts: int) -> np.ndarray:
        '''
        Extract the (n_cpts,) integer laminate-index block for one role
        gene from a flat design vector X.

        Layout matches fobjective._decode_design_vector:
            offset = 2*n_cpts + 4 + (n_cpts - 1) + role_idx * n_cpts
                   = 3*n_cpts + 3 + role_idx * n_cpts
        '''
        start = 3 * n_cpts + 3 + role_idx * n_cpts
        end   = start + n_cpts
        return np.asarray(X[start:end], dtype=int)

    @staticmethod
    def n_cpts_from_D(D: int) -> int:
        '''Recover n_cpts from the flat design dimension (D = 11 n + 3).'''
        if (D - 3) % 11 != 0:
            raise ValueError(
                f"[CL3O] Design dimension D={D} is not of the form 11*n_cpts+3."
            )
        return (D - 3) // 11

    @staticmethod
    def collect_selections(
        X_samples: np.ndarray,
        n_cpts   : int,
    ) -> np.ndarray:
        '''
        Stack laminate-index samples into a (n_samples, N_ROLES, n_cpts)
        integer tensor. X_samples is (n_samples, D) or (D,).
        '''
        Xs = np.atleast_2d(X_samples)
        out = np.empty((Xs.shape[0], N_ROLES, n_cpts), dtype=int)
        for s in range(Xs.shape[0]):
            for r in range(N_ROLES):
                out[s, r, :] = CoverageHelper.role_block(Xs[s], r, n_cpts)
        return out


# ========================================================================
# Internal helpers - laminate database I/O
# ========================================================================

def _load_catalogue_from_disk(
    mat_dir : Path = _DFLT_MAT_DIR,
    glob    : str  = "MAT_*_LaminateData.json",
) -> dict[str, LaminateData]:
    '''
    Load every laminate in `mat_dir` matching `glob`, sorted by filename.
    The returned dict is keyed by dense indices "MAT0".."MAT{n-1}" so it
    matches the runtime re-keying done in main._import_database.
    '''
    files = sorted(mat_dir.glob(glob))
    lams: dict[str, LaminateData] = {}
    for k, fp in enumerate(files):
        lams[f"MAT{k}"] = io.read_json(filepath=fp, dcls=LaminateData)
    return lams


def _display_names(
    laminate_db : dict[str, LaminateData],
    mat_dir     : Path = _DFLT_MAT_DIR,
    glob        : str  = "MAT_*_LaminateData.json",
) -> list[str]:
    '''
    Recover the on-disk laminate names (e.g. MAT_CFRP_UD8) in the same
    order as the runtime `MAT{k}` keys, by re-globbing the directory.
    Falls back to the runtime keys when the count does not match.
    '''
    files = sorted(mat_dir.glob(glob))
    if len(files) != len(laminate_db):
        return list(laminate_db.keys())
    return [fp.stem.removesuffix("_LaminateData") for fp in files]


# ========================================================================
# Public API - Coverage report
# ========================================================================

def coverage_report(
    history     : HistoryData,
    laminate_db : dict[str, LaminateData],
    n_cpts      : Optional[int]       = None,
    lam_names   : Optional[list[str]] = None,
    trail_frac  : float               = 0.25,
) -> str:
    '''
    Build a markdown report on how the DE search exercised the laminate
    catalogue.

    Args:
        history     : Completed HistoryData (best_X + feasible_X populated).
        laminate_db : Runtime dict {'MAT0': LaminateData, ...} - usually
            taken from `static_data.laminate_db`.
        n_cpts      : Number of wing control points. If None, recovered
            from history.D via the 11*n+3 relation.
        lam_names   : Human-readable names ordered to match laminate_db
            keys. If None, recovered by globbing _DFLT_MAT_DIR.
        trail_frac  : Fraction of the trailing best_X trajectory used as a
            "converged neighbourhood" sample (default last 25%).

    Returns:
        Markdown-formatted report as a single string.
    '''
    if n_cpts is None:
        n_cpts = CoverageHelper.n_cpts_from_D(int(history.D))

    n_lams = len(laminate_db)
    if lam_names is None or len(lam_names) != n_lams:
        lam_names = _display_names(laminate_db)

    # -------- 1. Decode feasible_X (the converged design) --------
    feas_X    = np.asarray(history.feasible_X, dtype=float)
    feas_sel  = CoverageHelper.collect_selections(feas_X, n_cpts)[0]  # (R, n_cpts)

    # -------- 2. Decode trailing best_X (convergence neighbourhood) --------
    best_X = np.asarray(history.best_X, dtype=float)
    if best_X.ndim == 2 and best_X.shape[0] >= 2:
        k_trail = max(1, int(np.floor(trail_frac * best_X.shape[0])))
        trail   = CoverageHelper.collect_selections(best_X[-k_trail:], n_cpts)
    else:
        trail = feas_sel[np.newaxis, ...]

    # -------- 3. Per-role frequency table over the trailing window --------
    # freq[r, m] = fraction of (gen, cpt) samples picking laminate m for role r
    n_samples = trail.shape[0] * n_cpts
    freq = np.zeros((N_ROLES, n_lams), dtype=float)
    for r in range(N_ROLES):
        counts = np.bincount(trail[:, r, :].ravel(), minlength=n_lams)
        freq[r, :] = counts / float(n_samples)

    overall_freq = freq.mean(axis=0)        # mean across roles

    # -------- 4. Dead options and dominators --------
    dead = [
        (m, lam_names[m]) for m in range(n_lams)
        if overall_freq[m] < _DEAD_THRESHOLD
    ]
    dominators = [
        (r, m, lam_names[m], freq[r, m])
        for r in range(N_ROLES)
        for m in range(n_lams)
        if freq[r, m] > _DOMINATOR_THRESHOLD
    ]

    # -------- 5. Per-role mass / stiffness of converged selections --------
    summary_lines = []
    for r, role in enumerate(ROLE_NAMES):
        for c in range(n_cpts):
            m = int(feas_sel[r, c])
            lam = laminate_db[f"MAT{m}"]
            summary_lines.append(
                f"| {role} | cpt {c} | {lam_names[m]} | "
                f"{lam.thick:.3f} | {lam.rho * 1e9:.3f} | "
                f"{lam.E1 / 1000:.1f} | {lam.G12 / 1000:.1f} |"
            )

    # -------- 6. Assemble markdown --------
    md: list[str] = []
    md.append("# Laminate Coverage Diagnostic\n")
    md.append(f"- design dimension D        : {int(history.D)}")
    md.append(f"- n control points          : {n_cpts}")
    md.append(f"- generations executed      : {int(history.ng)}")
    md.append(f"- laminates in catalogue    : {n_lams}")
    md.append(f"- trailing window for stats : "
              f"last {int(trail.shape[0])} / {int(best_X.shape[0])} gens")
    md.append(f"- converged feasible f      : {float(history.feasible_f):.4f}")
    md.append("")

    md.append("## Selection frequency per role (% of cpts in trailing window)\n")
    header = "| Laminate | " + " | ".join(ROLE_NAMES) + " | overall |"
    sep    = "|---" * (N_ROLES + 2) + "|"
    md.append(header)
    md.append(sep)
    for m in range(n_lams):
        row_vals = " | ".join(f"{100*freq[r, m]:5.1f}" for r in range(N_ROLES))
        md.append(
            f"| {lam_names[m]} | {row_vals} | {100*overall_freq[m]:5.1f} |"
        )
    md.append("")

    md.append(f"## Dead options (overall freq < {100*_DEAD_THRESHOLD:.0f}%)\n")
    if dead:
        for m, name in dead:
            md.append(f"- `{name}` (MAT{m}) - {100*overall_freq[m]:.1f}%")
    else:
        md.append("_None - every catalogue entry is used somewhere._")
    md.append("")

    md.append(
        f"## Dominators (single laminate > {100*_DOMINATOR_THRESHOLD:.0f}% "
        f"in a role)\n"
    )
    if dominators:
        for r, m, name, f_rm in dominators:
            md.append(
                f"- role `{ROLE_NAMES[r]}` -> `{name}` (MAT{m}) "
                f"at {100*f_rm:.1f}%"
            )
    else:
        md.append("_None - every role exhibits a meaningful trade._")
    md.append("")

    md.append("## Converged feasible design - per-section selections\n")
    md.append(
        "| role | cpt | laminate | t [mm] | rho [t/mm^3 * 1e-9] "
        "| E1 [GPa] | G12 [GPa] |"
    )
    md.append("|---|---|---|---|---|---|---|")
    md.extend(summary_lines)
    md.append("")

    return "\n".join(md)


# ========================================================================
# Public API - Standalone CLI
# ========================================================================

def run_from_disk(
    history_path : str | Path,
    mat_dir      : Path = _DFLT_MAT_DIR,
    out_path     : Optional[str | Path] = None,
) -> str:
    '''
    Load a HistoryData snapshot from JSON or pickle, then run the report.

    Args:
        history_path: Path to a HistoryData JSON (preferred) or pickle.
        mat_dir     : Directory holding the curated MAT_*_LaminateData.json.
        out_path    : Optional .md path to write the report to.

    Returns:
        Markdown report string.
    '''
    history_path = Path(history_path)
    if history_path.suffix == ".json":
        history = io.read_json(filepath=history_path, dcls=HistoryData)
    else:
        import pickle
        with open(history_path, "rb") as f:
            history = pickle.load(f)

    laminate_db = _load_catalogue_from_disk(mat_dir=mat_dir)
    report = coverage_report(history, laminate_db)

    if out_path is not None:
        Path(out_path).write_text(report, encoding="ascii")
    return report


# ============================================================================ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run laminate coverage diagnostic on a saved HistoryData."
    )
    parser.add_argument(
        "history_path",
        help="Path to HistoryData JSON or pickle.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional output .md filepath.",
    )
    args = parser.parse_args()

    print(run_from_disk(args.history_path, out_path=args.out))
