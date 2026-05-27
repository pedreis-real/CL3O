# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CL3O (Composite Lifting Surface Structural Sizing & Optimization) is an undergraduate-thesis (TCC) engineering tool that sizes and optimizes the composite structure of an aircraft lifting surface (wing). It models the wing cross-section as **three closed cells**, evaluates each candidate via **structural idealization + matrix structural analysis (Euler-Bernoulli beam FEA) + Tsai-Wu failure criteria**, and drives the design with a **Differential Evolution (DE)** optimizer. A separate FastAPI + React UI visualizes archived DE runs.

The reference design throughout the codebase is the **DA62** wing using the **WortmannFX63137** airfoil.

## Commands

There is no build system, packaging (`setup.py`/`pyproject.toml`), or test framework for the Python core. Modules are run directly with the interpreter. **Always run from the project root** so the `src/`-relative path insertion in each module resolves correctly.

```bash
# Run the full optimization pipeline (edit the __main__ block of src/main.py to
# change aircraft_name, opt_name, DE hyper-parameters, materials/airfoils loaded)
python src/main.py

# Rebuild JSON databases under data/ — each is produced by running its module's
# __main__ block (parameters are hardcoded there, edit before running):
python src/geometry/wing.py                 # data/wings/<id>_WingData.json
python src/geometry/airfoil.py              # data/airfoils/<id>_AirfoilData.json
python src/materials/composite_library.py   # data/materials/plies/*.json + MAT_*_LaminateData.json
python src/fea/loads/load_mapper.py         # data/loads/<id>_ExLoadsData.json + InLoadsData.json
python src/utils/oppoints.py                # data/oppoints/<id>_OppData.json

# Validation scripts (function-based, standalone — these are the closest thing
# to a test suite; they assert against the production modules):
python src/validation/validate_runtime_pipeline.py   # end-to-end pipeline smoke + shape/type asserts
python src/validation/validate_fea.py                # FEA beam pipeline vs. reference
python src/validation/validate_cross_section.py
python src/validation/laminate_coverage.py
```

### Visualization UI

```bash
# Backend deps (the only pinned requirements; core scientific deps are unpinned):
pip install -r requirements.txt          # fastapi, uvicorn

# Frontend (React + Vite + Plotly + Zustand), from src/ui/frontend/:
npm install
npm run build        # tsc && vite build -> src/ui/frontend/dist/ (required before desktop mode)
npm run dev          # Vite dev server (proxies to the API)

# API server (serves JSON API + the built SPA on one port), from project root:
python -m uvicorn ui.backend.app:app --reload --app-dir src --port 8000

# Standalone desktop app (starts/reuses the server, opens a native window):
python src/ui/desktop.py
```

Core Python dependencies (unpinned, install manually): `numpy`, `scipy`, `matplotlib`, `vtk`, `pandas`. Optional for desktop UI: `pywebview`.

Per the user's global workflow: typecheck after a series of changes (frontend: `tsc`), and prefer running a single validation script over everything.

## Architecture

### Data flow: static database → runtime pipeline → optimization

The system separates **static data** (loaded once per run, immutable during optimization) from **runtime data** (rebuilt for every DE candidate).

1. **JSON databases under `data/`** are the single source of truth for inputs. They are written once by running individual modules' `__main__` blocks (see commands above) and read back at runtime. Categories: `wings/`, `airfoils/`, `materials/` (+ `materials/plies/`), `loads/`, `oppoints/`. Laminates are discovered by glob over `MAT_*_LaminateData.json` (the underscore prefix distinguishes the curated catalogue from legacy `MAT{int}` test laminates).

2. **`src/main.py`** is the entry point. `RunCLEO.__init__` loads every `DatabaseSpec` into a single `StaticData` container (including derived `lerp_wing_db` and FEA pre-process artifacts `fem_setup`), assembles the DE evaluator closure, resolves DE bounds, and builds `SetupOpt`. `RunCLEO.run()` then drives the DE loop with an optional live Matplotlib (convergence) + VTK (3-D wing geometry) viewer (`LivePlotter`).

3. **`src/optimization/fobjective.py` → `BuildEvaluator`** assembles the **10-step core pipeline** into one closure `eval_(X) -> float` called once per DE candidate. This is the heart of the system — to understand the data flow, read this file. The steps and their owning modules:
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

4. **`src/optimization/de_opt.py`** implements the DE algorithm (`SetupOpt`, `RunOpt`, `OptVars`, `HistoryData`). When `RunOpt` is given an `out_dir`, it pickles one `RuntimeData` snapshot per generation under `out_dir/generations/` and writes `manifest.json` — this is exactly what the visualization UI consumes.

5. **Outputs** land in `outputs/<aircraft>_<opt>/` (gitignored).

### Cross-section structural topology (project-wide constants)

The three-cell cross-section is described by fixed index maps in `src/Constants.py` — these are load-bearing and referenced across geometry/FEA/post modules. Key constants: **7 structural booms** (`B1..B7`), **10 T2 sub-panels**, **7 T1 base segments**, **4 boom flanges** (`T4: F1..F4`). The `T2_TO_T1`, `BOOM_TO_T4`, `FLANGE_BOOM_IDX`, and `STRINGER_BOOM_IDX` arrays encode how panels/flanges/stringers map onto booms. **All shared constants and design limits (`OPT_LIMS`, `DE_HYPERPAR`, `PENALTY_VARS`, unit conversions) live in `Constants.py` — import from there, never redefine locally.**

### Module conventions

- **3-layer module structure**: production modules follow a consistent **Data / Helper / Main** split — `@dataclass` containers for I/O, a static-method `*Helper` class for pure utilities, and the main API class(es). Validation scripts under `src/validation/` intentionally break this (function-based).
- **Dataclass-based I/O**: every database entry is a dataclass; `utils/io_utils.py` provides `read_json(filepath, dcls=...)`, `write_json(...)`, and `setup_logger(self, enable_logging)`. JSON deserialization is dataclass-typed.
- **Path resolution**: each runnable module inserts `src/` (or project root) onto `sys.path` at import time, then imports siblings as top-level packages (e.g. `from geometry.wing import WingData`). This is why scripts must be run from the project root.
- **Units**: lengths in **mm**, mass density stored in **t/mm³** (mass output converted to kg via `T_TO_KG`/`DFLT_MASS_COEF`). Coordinate convention: X = chord, Y = span, Z = vertical.

### Visualization UI architecture

- **Backend** (`src/ui/backend/`): FastAPI app (`app.py`) over a `RunRepository` that reads the pickled per-generation snapshots from `outputs/`. `extract.py` pulls plot-ready arrays from a snapshot; `surface.py` builds 3-D geometry/stress surfaces; `serialize.py` (`to_jsonable`) sanitizes numpy → list and NaN/Inf → null for browser safety. Routes are namespaced `/api/runs/{run_id}/gen/{k}/...`; the built SPA is mounted last at `/`.
- **Frontend** (`src/ui/frontend/`): React 18 + TypeScript + Vite, Plotly for charts, Zustand for state (`src/state/store.ts`), API client in `src/api/client.ts`. Plot components live under `src/plots/`.

## Notes

- `README.md`, `docs/`, and `Changelog.txt` are currently empty. The thesis PDF referenced in code as `../docs/TCC.pdf` is not present in the repo.
- `misc/old/tests/` holds an **outdated** pytest suite (kept for reference, not wired up). `misc/scripts/` and `misc/artifacts/` hold tuning/sensitivity sweep scripts and their JSON/PNG results.
- `data/materials/.old/` holds superseded laminate definitions; the active catalogue is the `MAT_*` files directly under `data/materials/`.
