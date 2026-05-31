# Design: `src/cl3o/utils/stats.py` — Post-hoc Results Visualization

**Date:** 2026-05-31
**Status:** Approved for planning (pending user review of this spec)

## 1. Purpose

A standalone, **plots-only** post-hoc analysis module for the CL3O package. It
*reads* artifacts produced by other parts of the system and renders
publication-style figures for the thesis (TCC). It produces no numeric reports
or CSVs and writes nothing back into the producing tools.

It consumes three result sources:

1. **Latin-Hypercube DE sweep** — `tools/output/<sweep>/results.csv` plus the
   per-sample `outputs/<aircraft>_<sweep>_LHS-<k>/rate.csv` convergence traces.
2. **Sensitivity analysis (ANOVA)** — the *redefined* `anova_results.csv`
   (per-group rows) plus its sibling `anova_summary.csv` (F/p/grand stats).
3. **Best design** — the last `gen_*.pkl` `RuntimeData` snapshot of a run.

## 2. Scope & Non-goals

- **In scope:** loading the artifacts with pandas, computing the few statistics
  the plots need (Spearman correlations via `scipy.stats.spearmanr`), and saving
  matplotlib/seaborn figures to `tools/output/stats/`.
- **Out of scope / non-goals:**
  - No edits to `tools/tune_de.py` or `tools/sensitivity_analysis.py` (they stay
    as the producers). The redefinition of `anova_results.csv` into variant A is
    a *separate* change the user will make to `sensitivity_analysis.py`; this
    module only consumes the new schema.
  - No numeric report files, no CSV output, no DataFrame return contracts beyond
    what the plotting methods need internally.
  - No re-running of the DE or the FEA pipeline. It reads finished artifacts only.

## 3. ANOVA schema this module targets (variant A)

The module reads the redefined format (the producer change is the user's
responsibility, done before this module is run):

`anova_results.csv` — one row per structural group, fully rectangular:

```
group, n_valid, mean_f, std_f, min_f, max_f, SS_within, SS_between, eta_sq
```

`anova_summary.csv` — single summary row:

```
grand_mean, SS_total, df_between, df_within, F_stat, p_value
```

The loader tolerates the summary file being absent (F/p annotations are then
omitted from the plot rather than raising).

## 4. Module architecture

Single file, following the project's **Data / Helper / Main** 3-layer
convention (banner docstring, package-qualified imports, ASCII-only,
`io.setup_logger`).

### 4.1 `StatsData` (`@dataclass`)

Resolved configuration: input paths, output dir, and style flags. Built by a
small factory/`__post_init__` from `(aircraft, sweep)` so the `__main__` block
stays terse, mirroring the other modules.

| Property        | Description                                                  |
|-----------------|--------------------------------------------------------------|
| `aircraft`      | Aircraft id (e.g. `"da62"`), lower-cased                     |
| `sweep`         | Sweep name (e.g. `"tune-de-3"`)                              |
| `lhs_results`   | Path to `tools/output/<sweep>/results.csv`                   |
| `rate_glob`     | Glob for `outputs/<aircraft>_<sweep>_LHS-*/rate.csv`         |
| `anova_results` | Path to `anova_results.csv`                                  |
| `anova_summary` | Path to sibling `anova_summary.csv`                          |
| `run_name`      | Run folder under `outputs/` whose last `.pkl` is the design  |
| `out_dir`       | `tools/output/stats/` (created if missing)                   |
| `dpi`, `style`  | Figure DPI and seaborn theme/context flags                   |

Path roots: `outputs/` via `cl3o.paths.OUTPUTS_DIR`; the `tools/output/` root is
**not** a package path, so it is resolved as a `StatsData` field defaulting to
`<repo-root>/tools/output` (repo root taken from `cl3o.paths.ROOT_DIR`).

### 4.2 `StatsHelper` (static methods only)

Pure utilities, no state:

- `load_lhs_results(path) -> pd.DataFrame` — read `results.csv`.
- `load_rate_curves(glob) -> dict[int, pd.DataFrame]` — read every `rate.csv`,
  keyed by LHS sample index parsed from the folder name.
- `load_anova(results_path, summary_path) -> tuple[pd.DataFrame, dict | None]`
  — per-group table + parsed summary dict (or `None` if the summary is missing).
- `load_last_pkl(run_dir) -> RuntimeData` — reuse the `_PKL_SUBDIRS`
  search-order logic from `sensitivity_analysis.py` (`generations/`,
  `opt_files/`, root) to find and unpickle the highest-numbered `gen_*.pkl`.
- `spearman_matrix(df, params, outcomes) -> pd.DataFrame` — Spearman ρ block via
  `scipy.stats.spearmanr`.
- `apply_style(data)` — set the seaborn theme/context once.

### 4.3 `RunStats` (orchestrator / Main)

Owns a `StatsData` and a logger. Public API:

- `plot_lhs()` — the four LHS figures (§5.1).
- `plot_sensitivity()` — the two ANOVA figures (§5.2).
- `plot_best_design()` — the four best-design figures (§5.3).
- `run_all()` — calls all three, skipping any source whose artifacts are
  missing (logs a warning, does not raise), so a partial dataset still produces
  what it can.

`__main__` block hardcodes `aircraft="da62"`, `sweep="tune-de-3"`,
`run_name=...` (editable, like every other module's `__main__`), builds
`StatsData`, and calls `run_all()`.

## 5. Figures

Each figure is saved as `tools/output/stats/<name>.png` (and the existing
analysis tools' DPI/`bbox_inches="tight"` conventions are reused).

### 5.1 LHS sweep (`results.csv` + `rate.csv`)

- **`lhs_corr_heatmap.png`** — Spearman ρ heatmap, hyper-params
  `{NP, CR, F, lambda}` against outcomes
  `{best_f_final, feasible_f, n_gens, gen_to_half, elapsed_s}`. Annotated cells.
- **`lhs_param_scatter.png`** — 2×2 panel, each hyper-param vs `best_f_final`,
  points colored by `converged`, with a linear trend line per panel.
- **`lhs_convergence.png`** — `best_f` vs generation `k` (log-y) overlaid for
  every `rate.csv`; best-final sample highlighted; convergence (`conv == "Y"`)
  generation marked.
- **`lhs_speed_ecdf.png`** — ECDF (or histogram) of `n_gens` across samples, the
  convergence-speed distribution.

### 5.2 Sensitivity (`anova_results.csv` + `anova_summary.csv`)

- **`anova_eta_sq.png`** — horizontal η² bar (tornado), groups ranked by effect
  size; subtitle annotated with `F_stat` and `p_value` from the summary file
  (omitted gracefully if the summary file is absent).
- **`anova_group_means.png`** — per-group `mean_f` bar with ±`std_f` error bars.
  (Box plots are intentionally not attempted: the redefined CSV carries summary
  statistics only, not the raw per-perturbation fitness values.)

### 5.3 Best design (last `.pkl` `RuntimeData`)

Access patterns mirror `cl3o/ui/backend/extract.py`:

- **`design_mass.png`** — mass breakdown bar from `rt.score` (`total` plus the
  available component arrays, e.g. `panels`, `flanges`).
- **`design_margins.png`** — Tsai-Wu margin of safety and displacement margin
  of safety, with the critical (minimum) margin highlighted (from `rt.tsw`,
  `rt.displ`).
- **`design_panel_stress.png`** — per-panel max|τ| spanwise from `rt.stress`
  (`tauA`/`tauB`), mirroring `extract.stress`.
- **`design_forces.png`** — internal force / moment diagrams along the span from
  `rt.fea_rts`, mirroring `extract.forces` (local-frame resultants vs |Y|).

## 6. Dependencies

Add **seaborn** to `pyproject.toml`. Because `stats.py` lives inside the
installed package and imports seaborn at module top-level, seaborn must be
importable wherever this module is imported. To avoid forcing it on all package
users, add it as a new optional extra:

```toml
[project.optional-dependencies]
analysis = ["seaborn>=0.13"]
```

Installed via `pip install -e ".[analysis]"`. The module's banner docstring and
`CLAUDE.md`-style usage note will document this. matplotlib/scipy/pandas are
already core dependencies.

## 7. Error handling

- Missing artifact files: each `plot_*` method checks its inputs and, when
  missing, logs a `[CL3O]`-prefixed warning and returns without raising, so
  `run_all()` degrades gracefully on partial data.
- Hard failures (a present-but-corrupt CSV/pickle) propagate as the underlying
  pandas/pickle exception — these indicate a real problem, not a missing source.
- All user-facing error strings follow the project's `[CL3O]` pipe-aligned
  convention with a recovery hint.

## 8. Testing / verification

- Plotting modules in this repo are not unit-tested (the suite dropped plot
  smoke tests; CI uses `MPLBACKEND=Agg`). Verification for this module is:
  1. Run `python -m cl3o.utils.stats` headless (`MPLBACKEND=Agg`) against the
     existing `tune-de-3` artifacts and confirm the LHS + best-design figures
     are written to `tools/output/stats/`.
  2. The ANOVA figures require the redefined variant-A CSVs; verify against a
     small hand-written `anova_results.csv` + `anova_summary.csv` fixture if the
     producer has not yet been updated.
- Typecheck is not applicable (pure Python, no tsc); a `python -c "import
  cl3o.utils.stats"` import check confirms the module loads with seaborn present.

## 9. Open assumptions

- The redefined `anova_results.csv`/`anova_summary.csv` (variant A) will exist
  by the time the sensitivity figures are generated; until then those two
  figures are skipped with a warning.
- `run_name` for the best-design figures defaults to a `tune-de-3` LHS folder
  (which contains a single last-gen `.pkl`); any `outputs/` run folder works.
