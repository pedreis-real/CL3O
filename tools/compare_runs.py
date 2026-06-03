'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
DE-Run Comparison Tool.

Compares any archived RuntimeData field across two or more DE runs. For each
selected run under outputs/<run>/generations/ it resolves the snapshot to load
(by default the last generation, dedup resolved through the run manifest),
unpickles the RuntimeData, and projects it through one of the cl3o UI "views"
(cl3o.ui.backend.extract). Fields inside a view are then addressed by dotted
path and laid side by side as a console/CSV table and, where numeric, a chart.

Views (the data families you can compare) and what they expose:
    info     scalar summary  - fitness.*, tsw.*, displacement.*, mass.*, optvars
    forces   spanwise beam diagrams - span, local.{N,S1,S2,T,M1,M2},
                                       global.{N,S1,S2,T,M1,M2}
    section  one cross-section (pick --station) - props.*, fluxes.*
    mesh     beam mesh - nodes, displacement, max_disp, deformed
    stress   per-panel shear - tau, q, min, max, n_*

The forces view uses unified component names so the section-aligned (local)
and wing (global) frames line up field-for-field:
    N   axial            S1  shear-1 (local Y / global X)   M1  moment-1
    T   torsion          S2  shear-2 (local Z / global Z)   M2  moment-2

A field path is "<key>.<subkey>...". Point it at:
    - a scalar leaf (e.g. info.mass.total)     -> table + bar chart across runs
    - a 1-D array leaf (e.g. forces.local.Sz)  -> overlaid line plot + stats
    - a whole sub-dict  (e.g. forces.local)    -> expands to every numeric leaf

Discover what a view offers with --list (no field needed).

Usage:
    # what can I compare in the 'forces' view?
    python -m tools.compare_runs --view forces --list

    # compare a scalar across every archived run
    python -m tools.compare_runs --view info --field mass.total fitness.total

    # compare spanwise shear/torsion for a subset of runs
    python -m tools.compare_runs da62_test-1 da62_test-2 \
        --view forces --field local.Sz local.T

    # compare a cross-section property at a given station
    python -m tools.compare_runs --view section --station 0 --field props.I_XX

    # sweep every comparable field in a view at once
    python -m tools.compare_runs --view forces --all

    # peek an earlier generation instead of the last one
    python -m tools.compare_runs da62_test-1 --view info --field mass.total --gen 3

Outputs written to tools/output/compare_runs/:
    <view>__<fields>.csv    per-run / per-field table
    <view>__<fields>.png    bar chart (scalars) or overlaid series (arrays)

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import argparse
import csv
import fnmatch
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ================ Pathing ================
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ================ Module imports ================

# Utilities
from cl3o.paths import OUTPUTS_DIR

# Results
from cl3o.ui.backend import extract

# ================ Module-level constants ================
_OUT_DIR = Path(__file__).resolve().parents[1] / "tools" / "output" / "compare_runs"
_VIEWS   = ("info", "forces", "section", "mesh", "stress")
# Per-view natural x-axis key used when overlaying 1-D array fields.
_XAXIS   = {"forces": "span", "mesh": None, "stress": None,
            "section": None, "info": None}
# Unify forces component names so local/global frames compare field-for-field.
#   N axial | S1,S2 shears | T torsion | M1,M2 moments
_FORCES_LOCAL_MAP  = {"N": "N", "Sy": "S1", "Sz": "S2",
                      "T": "T", "My": "M1", "Mz": "M2"}
_FORCES_GLOBAL_MAP = {"N": "N", "SX": "S1", "SZ": "S2",
                      "T": "T", "MX": "M1", "MZ": "M2"}


# ================================================================================
# Helpers - run / snapshot discovery
# ================================================================================

def _list_runs(patterns: list[str]) -> list[Path]:
    '''
    Resolve run directories under OUTPUTS_DIR matching the given patterns.

    A run qualifies only if it owns a generations/ sub-directory holding at
    least one gen_*.pkl. With no patterns, every qualifying run is returned.

    Args:
        patterns: Run names or fnmatch globs (matched against the dir name).

    Returns:
        List of run directory Paths, sorted by name.
    '''
    candidates = [
        d for d in OUTPUTS_DIR.iterdir()
        if d.is_dir() and (d / "generations").is_dir()
        and any((d / "generations").glob("gen_*.pkl"))
    ]
    if patterns:
        candidates = [
            d for d in candidates
            if any(fnmatch.fnmatch(d.name, p) or d.name == p for p in patterns)
        ]
    return sorted(candidates, key=lambda d: d.name)


def _resolve_snapshot(run_dir: Path, gen: int | None) -> tuple[Path, dict]:
    '''
    Resolve which gen_*.pkl to load for a run.

    The manifest's snapshot list maps every generation index to its archived
    pickle (resolving dedup), so we honour it when present: `gen=None` -> last
    snapshot, otherwise the snapshot with matching `k`. Without a manifest we
    fall back to the highest-numbered gen_*.pkl on disk.

    Args:
        run_dir: Run directory (owns manifest.json and generations/).
        gen:     Generation index to load, or None for the last one.

    Returns:
        Tuple (pickle_path, snapshot_meta) where snapshot_meta is the manifest
        record for that generation (empty dict if no manifest).
    '''
    gens_dir = run_dir / "generations"
    manifest = run_dir / "manifest.json"

    if manifest.is_file():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        snaps = m.get("snapshots") or []
        if snaps:
            if gen is None:
                rec = snaps[-1]
            else:
                rec = next((s for s in snaps if int(s.get("k", -1)) == gen), None)
                if rec is None:
                    raise SystemExit(
                        f"[CL3O] Generation {gen} not found in manifest.\n"
                        f"| Run  : {run_dir.name}\n"
                        f"| Have : {[int(s.get('k', -1)) for s in snaps]}"
                    )
            return gens_dir / rec["file"], rec

    files = sorted(gens_dir.glob("gen_*.pkl"))
    if not files:
        raise SystemExit(f"[CL3O] No gen_*.pkl under {gens_dir}.")
    if gen is not None:
        target = gens_dir / f"gen_{gen:04d}.pkl"
        if not target.is_file():
            raise SystemExit(f"[CL3O] Missing {target.name} in {run_dir.name}.")
        return target, {}
    return files[-1], {}


def _load_runtime(pkl_path: Path) -> object:
    '''Unpickle a RuntimeData snapshot.'''
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _extract_view(rt: object, view: str, args: argparse.Namespace) -> dict:
    '''Project a RuntimeData through the requested cl3o UI extractor.'''
    if view == "info":
        return extract.info(rt)
    if view == "forces":
        fr = extract.forces(rt, lc=args.lc)
        fr["local"]  = {_FORCES_LOCAL_MAP[k]: v
                        for k, v in fr["local"].items()}
        fr["global"] = {_FORCES_GLOBAL_MAP[k]: v
                        for k, v in fr["global"].items()}
        return fr
    if view == "mesh":
        return extract.mesh(rt, lc=args.lc, deformed=True)
    if view == "stress":
        return extract.stress(rt, lc=args.lc)
    if view == "section":
        return extract.section(rt, args.station)
    raise SystemExit(f"[CL3O] Unknown view '{view}'. Choose from {_VIEWS}.")


# ================================================================================
# Helpers - field navigation
# ================================================================================

def _navigate(view_dict: dict, path: str) -> object:
    '''
    Resolve a dotted field path inside a view dict.

    Args:
        view_dict: Output of an extract.* view function.
        path:      Dotted key path, e.g. "mass.total" or "local.Sz".

    Returns:
        The node addressed by the path (scalar, list/array, or sub-dict).

    Raises:
        KeyError: when a path segment is missing (with the valid keys shown).
    '''
    node = view_dict
    walked: list[str] = []
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            avail = list(node.keys()) if isinstance(node, dict) else "(leaf)"
            raise KeyError(
                f"[CL3O] Field '{path}' not found at '{'.'.join(walked) or '<root>'}'.\n"
                f"| Available here : {avail}"
            )
        node = node[key]
        walked.append(key)
    return node


def _is_scalar(v: object) -> bool:
    '''True for a plain numeric/bool scalar (None counts as a missing scalar).'''
    return v is None or isinstance(v, (int, float, bool, np.integer, np.floating))


def _as_1d(v: object) -> np.ndarray | None:
    '''Coerce a list/array leaf to a finite-capable 1-D float array, else None.'''
    try:
        arr = np.asarray(
            [np.nan if x is None else x for x in v]
            if isinstance(v, (list, tuple)) else v,
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    return arr.ravel() if arr.ndim == 1 else None


def _flatten_leaves(node: object, prefix: str = "") -> list[tuple[str, object]]:
    '''
    Flatten a view (sub-)dict into (dotted_path, leaf) pairs.

    Recurses through nested dicts; stops at scalars and list/array leaves.
    Used by --list and by sub-dict field expansion.
    '''
    out: list[tuple[str, object]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(_flatten_leaves(v, f"{prefix}.{k}" if prefix else k))
    else:
        out.append((prefix, node))
    return out


def _leaf_kind(v: object) -> str:
    '''Classify a leaf for display: scalar / array(n) / 2d(shape) / other.'''
    if _is_scalar(v):
        return "scalar"
    a = _as_1d(v)
    if a is not None:
        return f"array({a.size})"
    arr = np.asarray(v, dtype=object)
    if arr.ndim >= 2:
        return f"nd{arr.shape}"
    return "other"


# ================================================================================
# Helpers - reporting
# ================================================================================

def _safe_name(fields: list[str]) -> str:
    '''Filesystem-safe stem from the requested field list.'''
    joined = "_".join(fields) if fields else "all"
    return re.sub(r"[^0-9A-Za-z._-]+", "-", joined)[:80]


def _series_stats(arr: np.ndarray) -> dict:
    '''Summary scalars for a 1-D series leaf.'''
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {k: np.nan for k in ("first", "peak_abs", "last", "min", "max", "mean")}
    return {
        "first":    float(arr[0]),
        "peak_abs": float(finite[np.argmax(np.abs(finite))]),
        "last":     float(arr[-1]),
        "min":      float(finite.min()),
        "max":      float(finite.max()),
        "mean":     float(finite.mean()),
    }


# ================================================================================
# Main
# ================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare any archived RuntimeData field across DE runs."
    )
    parser.add_argument(
        "runs", nargs="*",
        help="Run names or globs under outputs/ (default: all archived runs).",
    )
    parser.add_argument(
        "--view", choices=_VIEWS, default="info",
        help="Data family to project the RuntimeData through (default: info).",
    )
    parser.add_argument(
        "--field", nargs="*", default=[],
        help="Dotted field path(s) within the view (sub-dicts expand to leaves).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Compare every comparable (scalar / 1-D) field in the view.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List every comparable field in the view (uses the first run) and exit.",
    )
    parser.add_argument(
        "--gen", type=int, default=None,
        help="Generation index to compare (default: last generation per run).",
    )
    parser.add_argument(
        "--lc", type=int, default=0,
        help="Load case index, for views that depend on it (default: 0).",
    )
    parser.add_argument(
        "--station", type=int, default=0,
        help="Cross-section station index for the 'section' view (default: 0).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip chart generation (table/CSV only).",
    )
    args = parser.parse_args()

    runs = _list_runs(args.runs)
    if not runs:
        raise SystemExit(
            f"[CL3O] No matching runs with archived generations under {OUTPUTS_DIR}."
        )

    # ---- Load + extract each run -------------------------------------------
    views: dict[str, dict] = {}   # run name -> view dict
    meta:  dict[str, dict] = {}   # run name -> snapshot meta
    for run_dir in runs:
        pkl, m = _resolve_snapshot(run_dir, args.gen)
        rt = _load_runtime(pkl)
        views[run_dir.name] = _extract_view(rt, args.view, args)
        meta[run_dir.name]  = {"file": pkl.name, **m}
        print(
            f"[CL3O] {run_dir.name:<28} {pkl.name:<14} "
            f"best_f={m.get('best_f')!s:<10} feasible={m.get('is_feasible')}"
        )

    first = views[next(iter(views))]

    # ---- --list mode: enumerate fields and exit ----------------------------
    if args.list or (not args.field and not args.all):
        print(f"\n{'='*70}\nView '{args.view}' - comparable fields"
              f"{' (station %d)' % args.station if args.view == 'section' else ''}")
        print("=" * 70)
        for path, leaf in _flatten_leaves(first):
            print(f"  {path:<40} {_leaf_kind(leaf)}")
        print("\n[CL3O] Re-run with --field <path> [...] or --all to compare.")
        return

    # ---- Resolve the field list to compare ---------------------------------
    # --all sweeps every scalar / 1-D leaf in the view; otherwise expand each
    # requested path (a sub-dict path fans out to its numeric leaves).
    fields: list[str] = []
    if args.all:
        for path, leaf in _flatten_leaves(first):
            if _is_scalar(leaf) or _as_1d(leaf) is not None:
                fields.append(path)
    else:
        for fpath in args.field:
            node = _navigate(first, fpath)
            if isinstance(node, dict):
                fields.extend(f"{fpath}.{sub}" for sub, _ in _flatten_leaves(node))
            else:
                fields.append(fpath)

    # Split fields by leaf kind (decided on the first run).
    scalar_fields = [f for f in fields if _is_scalar(_navigate(first, f))]
    array_fields  = [f for f in fields
                     if not _is_scalar(_navigate(first, f))
                     and _as_1d(_navigate(first, f)) is not None]
    skipped = [f for f in fields if f not in scalar_fields and f not in array_fields]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.view}__{_safe_name(args.field)}"

    # ---- Scalar comparison: table + bar chart ------------------------------
    if scalar_fields:
        print(f"\n{'='*70}\nScalars - view={args.view}, lc={args.lc}\n{'='*70}")
        rows = [["run", "file", *scalar_fields]]
        table: dict[str, list] = {}
        for name in views:
            vals = []
            for f in scalar_fields:
                v = _navigate(views[name], f)
                vals.append(float(v) if _is_scalar(v) and v is not None else np.nan)
            table[name] = vals
            rows.append([name, meta[name]["file"], *vals])

        col_w = max(len(n) for n in views)
        head = "  " + " " * col_w + "".join(f"{f:>16}" for f in scalar_fields)
        print(head)
        for name, vals in table.items():
            line = "  " + name.ljust(col_w)
            line += "".join(f"{v:>16.5g}" for v in vals)
            print(line)

        csv_path = _OUT_DIR / f"{stem}__scalars.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        print(f"\n[CL3O] Wrote scalar table -> {csv_path}")

        if not args.no_plot:
            names = list(views)
            x = np.arange(len(names))
            nf = len(scalar_fields)
            width = 0.8 / nf
            fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names)), 4.5))
            for j, f in enumerate(scalar_fields):
                ax.bar(x + (j - (nf - 1) / 2) * width,
                       [table[n][j] for n in names], width, label=f)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_title(f"{args.view} scalars (lc={args.lc})")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            png = _OUT_DIR / f"{stem}__scalars.png"
            fig.savefig(png, dpi=150)
            print(f"[CL3O] Wrote scalar bar chart -> {png}")

    # ---- Array comparison: overlaid series + per-series stats --------------
    if array_fields:
        xkey = _XAXIS.get(args.view)
        print(f"\n{'='*70}\nSeries - view={args.view}, lc={args.lc} "
              f"(x = {xkey or 'index'})\n{'='*70}")

        stat_rows = [["run", "field", "first", "peak_abs", "last",
                      "min", "max", "mean"]]
        for f in array_fields:
            print(f"\n  {f}")
            print(f"    {'run':<28}{'first':>13}{'peak|.|':>13}{'last':>13}")
            for name in views:
                arr = _as_1d(_navigate(views[name], f))
                st = _series_stats(arr)
                stat_rows.append([name, f, st["first"], st["peak_abs"],
                                  st["last"], st["min"], st["max"], st["mean"]])
                print(f"    {name:<28}{st['first']:>13.4g}"
                      f"{st['peak_abs']:>13.4g}{st['last']:>13.4g}")

        csv_path = _OUT_DIR / f"{stem}__series.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(stat_rows)
        print(f"\n[CL3O] Wrote series stats -> {csv_path}")

        if not args.no_plot:
            n = len(array_fields)
            ncol = min(3, n)
            nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow),
                                     squeeze=False)
            for idx, f in enumerate(array_fields):
                ax = axes[idx // ncol][idx % ncol]
                for name in views:
                    vd = views[name]
                    y = _as_1d(_navigate(vd, f))
                    x = _as_1d(vd.get(xkey)) if xkey else None
                    if x is None or x.size != y.size:
                        x = np.arange(y.size)
                    order = np.argsort(x)
                    ax.plot(x[order], y[order], marker=".", ms=3, lw=1.2,
                            label=name)
                ax.axhline(0.0, color="0.7", lw=0.7)
                ax.set_title(f)
                ax.set_xlabel(xkey or "index")
                ax.grid(True, alpha=0.3)
            for idx in range(n, nrow * ncol):
                axes[idx // ncol][idx % ncol].axis("off")
            h, l = axes[0][0].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=min(len(views), 4),
                       frameon=False)
            fig.suptitle(f"{args.view} series (lc={args.lc})", fontsize=13)
            fig.tight_layout(rect=(0, 0.05, 1, 0.97))
            png = _OUT_DIR / f"{stem}__series.png"
            fig.savefig(png, dpi=150)
            print(f"[CL3O] Wrote series plot -> {png}")

    # ---- Report non-comparable leaves --------------------------------------
    if skipped:
        print(f"\n[CL3O] Skipped non-scalar / non-1-D fields: {skipped}")


if __name__ == "__main__":
    main()
