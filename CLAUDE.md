# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CL3O (Composite Lifting Surface Structural Sizing & Optimization) is an undergraduate-thesis (TCC) engineering tool that sizes and optimizes the composite structure of an aircraft lifting surface (wing). It models the wing cross-section as **three closed cells**, evaluates each candidate via **structural idealization + matrix structural analysis (Euler-Bernoulli beam FEA) + Tsai-Wu failure criteria**, and drives the design with a **Differential Evolution (DE)** optimizer. A separate FastAPI + React UI visualizes archived DE runs.

The reference design throughout the codebase is the **DA62** wing using the **WortmannFX63137** airfoil.

## Commands

The Python core is an installable package (`cl3o`) using a `src/` layout with a `pyproject.toml`. Install editable first; once installed, modules import as `cl3o.*` and run from anywhere (the old `sys.path` hacks are gone).

```bash
# Install (Python >= 3.10). Extras: [ui] backend, [desktop] launcher, [dev] pytest.
pip install -e ".[ui,dev]"

# Run the full optimization pipeline (edit the __main__ block of src/cl3o/main.py
# to change aircraft_name, opt_name, DE hyper-parameters, materials/airfoils loaded)
python -m cl3o.main

# Rebuild JSON databases under data/ — each is produced by running its module's
# __main__ block (parameters are hardcoded there, edit before running):
python -m cl3o.geometry.wing                 # data/wings/<id>_WingData.json
python -m cl3o.geometry.airfoil              # data/airfoils/<id>_AirfoilData.json
python -m cl3o.materials.composite_library   # data/materials/plies/*.json + MAT_*_LaminateData.json
python -m cl3o.fea.loads.load_mapper         # data/loads/<id>_ExLoadsData.json + InLoadsData.json
python -m cl3o.utils.oppoints                # data/oppoints/<id>_OppData.json

# Tests (pytest config in pyproject; testpaths=["tests"], `slow` marker):
pytest -m "not slow"                                   # fast unit tests
pytest                                                 # full suite (builds a real DA62 session)

# Validation scripts under tests/validation/ (function-based, standalone — they
# assert against the production modules; the runtime gate is also the CI smoke test):
python -m tests.validation.validate_runtime_pipeline   # end-to-end pipeline smoke + shape/type asserts
python -m tests.validation.validate_fea                # FEA beam pipeline dump
python -m tests.validation.validate_cross_section
python -m tests.validation.laminate_coverage           # post-run diagnostic (takes a HistoryData)
```

The two plot-based validators (`validate_fea`, `validate_cross_section`) are headless by default (`_SHOW_PLOTS = False`); flip that flag to display figures. Use `MPLBACKEND=Agg` in CI / headless shells.

### Visualization UI

```bash
# Frontend (React + Vite + Plotly + Zustand), from src/cl3o/ui/frontend/:
npm install
npm run build        # tsc && vite build -> src/cl3o/ui/frontend/dist/ (required before desktop mode)
npm run dev          # Vite dev server (proxies to the API)

# API server (serves JSON API + the built SPA on one port), from project root:
python -m uvicorn cl3o.ui.backend.app:app --reload --port 8000

# Standalone desktop app (starts/reuses the server, opens a native window):
python -m cl3o.ui.desktop
```

Dependencies are declared in `pyproject.toml`: core = `numpy, scipy, matplotlib, vtk, pandas`; `[ui]` = `fastapi, uvicorn[standard]`; `[desktop]` = `pywebview`; `[dev]` = `pytest`. `requirements.txt` is a thin pointer to `-e .[ui]`.

Per the user's global workflow: typecheck after a series of changes (frontend: `tsc`), and prefer running a single test / validation script over the whole suite.

## Architecture

### Data flow: static database → runtime pipeline → optimization

The system separates **static data** (loaded once per run, immutable during optimization) from **runtime data** (rebuilt for every DE candidate).

1. **JSON databases under `data/`** are the single source of truth for inputs. They are written once by running individual modules' `__main__` blocks (see commands above) and read back at runtime. Categories: `wings/`, `airfoils/`, `materials/` (+ `materials/plies/`), `loads/`, `oppoints/`. Laminates are discovered by glob over `MAT_*_LaminateData.json` (the underscore prefix distinguishes the curated catalogue from legacy `MAT{int}` test laminates).

2. **`src/cl3o/main.py`** is the entry point. `RunCLEO.__init__` loads every `DatabaseSpec` into a single `StaticData` container (including derived `lerp_wing_db` and FEA pre-process artifacts `fem_setup`), assembles the DE evaluator closure, resolves DE bounds, and builds `SetupOpt`. `RunCLEO.run()` then drives the DE loop with an optional live Matplotlib (convergence) + VTK (3-D wing geometry) viewer (`LivePlotter`).

3. **`src/cl3o/optimization/fobjective.py` → `BuildEvaluator`** assembles the **10-step core pipeline** into one closure `eval_(X) -> float` called once per DE candidate. This is the heart of the system — to understand the data flow, read this file. The steps and their owning modules (all under `src/cl3o/`):
   1. Validate flat design vector `X`
   2. Decode `X` → `OptVars` (`_decode_design_vector`; layout is `11 * n_cpts + 3`)
   3. Cross-section geometry — `geometry/section_builder.py` (`SectionBuilder`), backed by `geometry/geom_properties.py` and `geometry/structural_idealization.py`
   4. Global mesh + stiffness assembly — `fea/solver/mesh_builder.py` (`MeshBuilder`), `fea/elements/beam_element.py`
   5. Linear static solve `{F} = [K]{d}` — `fea/solver/static_analysis.py` (`LinearStaticSolver`)
   6. Stress recovery — `fea/post/stress_recovery.py` (`StressRecovery`)
   7. Tsai-Wu failure — `fea/post/tsw_failure.py` (`TsaiWuFailure`)
   8. Displacement margins of safety — `fea/post/displacement_ms.py` (`DisplacementMargins`)
   9. Penalty `P(X)` — `optimization/fpenalty.py` (`Penalty`)
   10. Structural mass `m(X)` (`optimization/fscore.py`) and scalar fitness `z(X) = w_m * m(X) + P(X)` (`TotalScore`)

4. **`src/cl3o/optimization/de_opt.py`** implements the DE algorithm (`SetupOpt`, `RunOpt`, `OptVars`, `HistoryData`). When `RunOpt` is given an `out_dir`, it pickles one `RuntimeData` snapshot per generation under `out_dir/generations/` and writes `manifest.json` — this is exactly what the visualization UI consumes.

5. **Outputs** land in `outputs/<aircraft>_<opt>/` (gitignored).

### Cross-section structural topology (project-wide constants)

The three-cell cross-section is described by fixed index maps in `src/cl3o/Constants.py` — these are load-bearing and referenced across geometry/FEA/post modules. Key constants: **7 structural booms** (`B1..B7`), **10 T2 sub-panels**, **7 T1 base segments**, **4 boom flanges** (`T4: F1..F4`). The `T2_TO_T1`, `BOOM_TO_T4`, `FLANGE_BOOM_IDX`, and `STRINGER_BOOM_IDX` arrays encode how panels/flanges/stringers map onto booms. **All shared constants and design limits (`OPT_LIMS`, `DE_HYPERPAR`, `PENALTY_VARS`, unit conversions) live in `Constants.py` — import from there, never redefine locally.**

### Module conventions

- **3-layer module structure**: production modules follow a consistent **Data / Helper / Main** split — `@dataclass` containers for I/O, a static-method `*Helper` class for pure utilities, and the main API class(es). Validation scripts under `tests/validation/` intentionally break this (function-based).
- **Dataclass-based I/O**: every database entry is a dataclass; `cl3o/utils/io_utils.py` provides `read_json(filepath, dcls=...)`, `write_json(...)`, and `setup_logger(self, enable_logging)`. JSON deserialization is dataclass-typed.
- **Path resolution**: imports are package-qualified (`from cl3o.geometry.wing import WingData`). All filesystem roots come from the single module `cl3o/paths.py` (`ROOT_DIR`, `DATA_DIR`, `OUTPUTS_DIR`, and the per-category dirs) — import those constants, never recompute `__file__`-relative paths. `data/` and `outputs/` stay at the repo root for an editable clone.
- **Units**: lengths in **mm**, mass density stored in **t/mm³** (mass output converted to kg via `T_TO_KG`/`DFLT_MASS_COEF`). Coordinate convention: X = chord, Y = span, Z = vertical.

### Visualization UI architecture

- **Backend** (`src/cl3o/ui/backend/`): FastAPI app (`app.py`) over a `RunRepository` that reads the pickled per-generation snapshots from `outputs/`. `extract.py` pulls plot-ready arrays from a snapshot; `surface.py` builds 3-D geometry/stress surfaces; `serialize.py` (`to_jsonable`) sanitizes numpy → list and NaN/Inf → null for browser safety. Routes are namespaced `/api/runs/{run_id}/gen/{k}/...`; the built SPA is mounted last at `/`.
- **Frontend** (`src/cl3o/ui/frontend/`): React 18 + TypeScript + Vite, Plotly for charts, Zustand for state (`src/state/store.ts`), API client in `src/api/client.ts`. Plot components live under `src/plots/`.

## Notes

- The thesis PDF referenced in older code as `docs/TCC.pdf` is not in the repo; the docstring now points to the README instead.
- `tests/` holds the live suite: `tests/*.py` unit tests (heavy ones marked `slow`) plus the standalone `tests/validation/` scripts. CI (`.github/workflows/ci.yml`) runs `pytest` on Python 3.11/3.12, then the runtime-pipeline gate, plus a frontend `npm run build`.
- The active laminate catalogue is the `MAT_*_LaminateData.json` files directly under `data/materials/` (discovered by glob); per-ply JSON lives in `data/materials/plies/`.
