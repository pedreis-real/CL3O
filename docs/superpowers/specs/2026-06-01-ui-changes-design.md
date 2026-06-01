# CL3O UI Changes — Design Spec
**Date:** 2026-06-01  
**Branch:** feature/stats-module  
**Scope:** Visualization app (`src/cl3o/ui/`)

---

## Overview

Seven independent changes to the CL3O visualization app, split across frontend
(React + Plotly) and backend (FastAPI). Changes are grouped by subsystem.

---

## 1. Hover Picker — All 3D Surface Plots

**Goal:** Show the numeric value of the displayed quantity when hovering over
any 3D wing surface.

**Frontend changes:**
- `src/cl3o/ui/frontend/src/plots/Plot.tsx` — extend `meshTrace` builder with
  optional `hoverValues: number[]` parameter. When provided, set `customdata`
  (one value per vertex, matching the `intensity` array) and `hovertemplate`
  with label and unit.
- Each plot component passes the correct array and label string:
  - **GeometryPlot**: laminate index string per vertex (e.g. `"ls1 = 3"`).
  - **MeshPlot** (displacement mode): displacement magnitude in mm.
  - **StressPlot** (stress mode): τ in MPa; (flux mode): q in N/mm;
    (failure mode): R dimensionless.

**No backend changes required.** All necessary data is already in the existing
API responses.

---

## 2. Snapshot Save to `outputs/snaps/`

**Goal:** The camera button in the Plotly modebar downloads the PNG to the
browser *and* saves a copy on the server under `outputs/snaps/`.

**Frontend changes:**
- `src/cl3o/ui/frontend/src/plots/Plot.tsx` — in `baseLayout` config:
  - Add `modeBarButtonsToRemove: ['toImage']`.
  - Add `modeBarButtonsToAdd` with a custom camera button.
- New hook `src/cl3o/ui/frontend/src/hooks/useSnapshotButton.ts`:
  - Accepts `(graphDivRef, view, runId, gen)`.
  - Returns a Plotly custom-button descriptor.
  - On click: calls `Plotly.toImage(div, {format:'png', scale:2})`, triggers
    browser download via temporary `<a>` element, then POSTs to `/api/snaps`.
- All plot components that use `baseLayout` pass the hook result into the
  `modeBarButtonsToAdd` array.

**Backend changes:**
- `src/cl3o/ui/backend/app.py` — new endpoint:
  ```
  POST /api/snaps
  Body: { run_id: str, gen: int, view: str, data: str }  // data = base64 PNG
  ```
  Decodes base64, writes to `outputs/snaps/<run_id>_gen<k>_<view>_<ts>.png`.
  Creates `outputs/snaps/` directory if absent. Returns `{ path: str }`.

---

## 3. Geometry — Skin Layup Color Bug

**Goal:** Fix the 3D wing surface colormap for skin panels (`ls1` / `ls2`).

**Investigation target:** `src/cl3o/ui/backend/surface.py` — `build_scene()`.
Hypothesis: the per-vertex layup index array for the skin surface uses a wrong
offset or `ls2` overwrites `ls1` in the lofting loop.

**Fix:** Correct the vertex-to-layup-index mapping so upper skin (`ls1`) and
lower skin (`ls2`) panels are colored independently and correctly.

**No type or API changes required.**

---

## 4. Geometry — Design Vector Table

**Goal:** Add a button next to "Layups" that opens a table showing
`i | variable | control point` for each entry of the flat design vector `X`.

**Frontend changes:**
- `src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx`:
  - Add a second toggle button `"X[ ]"` beside the existing "Layups" button.
  - Controls a new local boolean state `showXTable`.
  - When open, renders an overlay panel (same dark-card style as the layup panel)
    with a scrollable table.
- Label computation (pure client-side, no API call):
  ```
  n_cpts = (optvars.length - 3) / 11
  layout = [
    { name: "xw1",      len: n,   hasCp: true  },
    { name: "xw2",      len: n,   hasCp: true  },
    { name: "bf1_root", len: 1,   hasCp: false },
    { name: "bf2_root", len: 1,   hasCp: false },
    { name: "bf3_root", len: 1,   hasCp: false },
    { name: "bf4_root", len: 1,   hasCp: false },
    { name: "tpr",      len: n-1, hasCp: true  },
    { name: "ls1",      len: n,   hasCp: true  },
    { name: "ls2",      len: n,   hasCp: true  },
    { name: "lw1",      len: n,   hasCp: true  },
    { name: "lw2",      len: n,   hasCp: true  },
    { name: "lf1",      len: n,   hasCp: true  },
    { name: "lf2",      len: n,   hasCp: true  },
    { name: "lf3",      len: n,   hasCp: true  },
    { name: "lf4",      len: n,   hasCp: true  },
  ]
  ```
  Each block expands to rows: `{ i, variable: name, cp: hasCp ? k : "—" }`.

**No backend changes required.** Data arrives via existing `info.optvars`.

---

## 5. Stress — Default End = Min (B)

**Goal:** Change the initial element-end selector from `"avg"` to `"B"` (Min).

**Frontend change:** `src/cl3o/ui/frontend/src/state/store.ts` — change the
initial value of `end` from `"avg"` to `"B"`.

---

## 6. Stress — Tsai-Wu Failure Ratio R

**Goal:** Add a third mode "Failure (R)" to the Stress tab, showing the
Tsai-Wu strength ratio R lofted over the wing surface (R = 1 → failure).

**Frontend changes:**
- `src/cl3o/ui/frontend/src/state/store.ts` — extend `StressMode` union:
  `"stress" | "flux" | "tsw"`.
- `src/cl3o/ui/frontend/src/plots/StressPlot.tsx`:
  - Add "Failure (R)" button alongside "Stress (τ)" and "Flux (q)".
  - New fetch call to `/api/runs/{run_id}/gen/{k}/tsw3d?lc=&end=`.
  - Dedicated colormap `TSW_CMAP`: green (R ≥ 2) → yellow (R = 1) → red
    (R < 1). Colorbar centered at R = 1.0.
  - Reuses the same `Mesh3D` rendering path as stress mode.
- `src/cl3o/ui/frontend/src/api/client.ts` — add `tsw3d(runId, gen, lc, end)`
  function.
- `src/cl3o/ui/frontend/src/types.ts` — extend `StressScene` or add a sibling
  `TswScene` type if colorbar semantics differ enough.

**Backend changes:**
- `src/cl3o/ui/backend/surface.py` — new function `build_tsw_surface(rt, wing, lc, end)`:
  - Same lofting geometry as `build_stress_surface`.
  - `intensity` = `R_panels[:, :, end_idx, lc]` (shape m×10, one value per
    panel per element).
  - Boom rods colored by `R_booms[:, :, end_idx, lc]` (booms 0,2,4,6 =
    chord members).
  - Returns a dict compatible with `StressScene` (reuses `serialize.to_jsonable`).
- `src/cl3o/ui/backend/app.py` — new route:
  ```
  GET /api/runs/{run_id}/gen/{k}/tsw3d?lc=int&end=str
  ```

---

## 7. Sensitivity Tab — ANOVA Charts

**Goal:** Replace the placeholder in the Sensitivity sub-view of the Misc tab
with two stacked Plotly charts rendered from the project-level ANOVA results.

**Layout (option C — single combined view):**
```
┌─────────────────────────────────────────┐
│  η² por grupo estrutural                │
│  (barras horizontais, ordem crescente)  │
│  subtítulo: F = X,xxx   p = X.XXe-XX   │
├─────────────────────────────────────────┤
│  Distribuição de fitness por grupo      │
│  (box-plot: Q1/med/Q3 + whiskers)       │
└─────────────────────────────────────────┘
```

When ANOVA data is unavailable (`available: false`): show a one-line warning
with the expected file path.

**Frontend changes:**
- `src/cl3o/ui/frontend/src/plots/MiscPlot.tsx` — replace the stub block for
  `miscTab === "sensitivity"` with a new `<SensitivityPlot />` sub-component
  (or inline Plotly `<Plot>` pair).
- New fetch on mount: `GET /api/sensitivity`. Store result in local state.
- Remove the "spars / flanges / skin" sub-selector (no longer needed).
- `src/cl3o/ui/frontend/src/api/client.ts` — add `sensitivity()` function.
- `src/cl3o/ui/frontend/src/types.ts` — add `AnovaGroup` and `SensitivityData`
  types.

**Backend changes:**
- `src/cl3o/ui/backend/app.py` — new route:
  ```
  GET /api/sensitivity
  ```
  Reads `tools/output/sensitivity/anova_results.csv` + `anova_summary.csv`
  (paths relative to `ROOT_DIR`). Returns:
  ```json
  {
    "available": true,
    "groups": [
      { "group": "Revestimento", "eta_sq": 0.218, "mean_f": 481.96,
        "std_f": 433.25, "min_f": 36.24, "max_f": 1042.23 },
      ...
    ],
    "summary": { "F_stat": 47.075, "p_value": 1.54e-33,
                 "df_between": 4, "df_within": 495 }
  }
  ```
  If CSV files are missing → `{ "available": false }`.

---

## Affected Files Summary

| File | Changes |
|------|---------|
| `frontend/src/plots/Plot.tsx` | `meshTrace` hover params; remove native toImage button |
| `frontend/src/hooks/useSnapshotButton.ts` | New — custom snapshot hook |
| `frontend/src/plots/GeometryPlot.tsx` | Hover values; design vector table button |
| `frontend/src/plots/MeshPlot.tsx` | Hover values (displacement) |
| `frontend/src/plots/StressPlot.tsx` | Hover values; R mode button + fetch |
| `frontend/src/plots/MiscPlot.tsx` | Sensitivity ANOVA charts |
| `frontend/src/state/store.ts` | `end` default = `"B"`; extend `StressMode` |
| `frontend/src/api/client.ts` | Add `tsw3d()`, `sensitivity()`, `snaps()` |
| `frontend/src/types.ts` | Add `TswScene` (or extend `StressScene`); `SensitivityData` |
| `backend/app.py` | Add `POST /api/snaps`, `GET /api/sensitivity`, `GET /api/.../tsw3d` |
| `backend/surface.py` | Add `build_tsw_surface()` |

---

## Out of Scope

- No changes to the DE optimizer, FEA pipeline, or data generation.
- The ANOVA CSVs are read-only inputs; the UI does not regenerate them.
- No changes to existing API response schemas (backward-compatible).
