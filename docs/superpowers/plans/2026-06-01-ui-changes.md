# CL3O UI Changes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement seven independent UI improvements to the CL3O visualization
app: stress default, ANOVA sensitivity charts, Tsai-Wu R surface, hover picker,
skin layup fix, design vector table, and server-side snapshot save.

**Architecture:** Changes are independent and can be merged in any order. Backend
changes (FastAPI, `surface.py`) are separate from frontend changes (React,
Plotly). Each task produces a working, committable change.

**Tech Stack:** Python 3.11 / FastAPI / numpy (backend); React 18 / TypeScript /
Plotly.js / Zustand (frontend). Frontend typecheck: `npm run build` (runs
`tsc && vite build`) from `src/cl3o/ui/frontend/`.

---

## File Map

| File | Tasks |
|------|-------|
| `src/cl3o/ui/frontend/src/state/store.ts` | T1, T5 |
| `src/cl3o/ui/frontend/src/types.ts` | T3, T5 |
| `src/cl3o/ui/frontend/src/api/client.ts` | T3, T5, T9 |
| `src/cl3o/ui/frontend/src/plots/StressPlot.tsx` | T5 |
| `src/cl3o/ui/frontend/src/plots/MiscPlot.tsx` | T3 |
| `src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx` | T6, T7, T8 |
| `src/cl3o/ui/frontend/src/plots/MeshPlot.tsx` | T6 |
| `src/cl3o/ui/frontend/src/plots/Plot.tsx` | T9 |
| `src/cl3o/ui/frontend/src/hooks/useSnapshotButton.ts` | T9 (new) |
| `src/cl3o/ui/backend/app.py` | T2, T4, T9 |
| `src/cl3o/ui/backend/surface.py` | T4 |

---

## Task 1 — Stress default end = "B" (Min)

**Files:**
- Modify: `src/cl3o/ui/frontend/src/state/store.ts:101`

- [ ] **Step 1: Change the initial value**

In `store.ts`, line 101, change:
```typescript
  end: "avg",
```
to:
```typescript
  end: "B",
```

- [ ] **Step 2: Typecheck**

```
cd src/cl3o/ui/frontend && npm run build
```
Expected: build succeeds, no TypeScript errors.

- [ ] **Step 3: Commit**

```
git add src/cl3o/ui/frontend/src/state/store.ts
git commit -m "feat(ui): stress default end = B (Min)"
```

---

## Task 2 — Backend: ANOVA sensitivity endpoint

**Files:**
- Modify: `src/cl3o/ui/backend/app.py`

The ANOVA CSVs live at `tools/output/sensitivity/anova_results.csv` and
`tools/output/sensitivity/anova_summary.csv` (relative to `ROOT_DIR`).
The endpoint is project-global, not per-run.

- [ ] **Step 1: Add imports and helper to `app.py`**

Add at the top of `app.py`, after the existing imports:
```python
import csv
from cl3o.paths import ROOT_DIR

_ANOVA_RESULTS = ROOT_DIR / "tools" / "output" / "sensitivity" / "anova_results.csv"
_ANOVA_SUMMARY = ROOT_DIR / "tools" / "output" / "sensitivity" / "anova_summary.csv"
```

- [ ] **Step 2: Add the route**

Add before the static-SPA mount at the bottom of `app.py`:

```python
@app.get("/api/sensitivity")
def sensitivity():
    if not _ANOVA_RESULTS.is_file():
        return _json({"available": False})
    groups = []
    with open(_ANOVA_RESULTS, newline="") as fh:
        for row in csv.DictReader(fh):
            groups.append({
                "group":    row["group"],
                "eta_sq":   float(row["eta_sq"]),
                "mean_f":   float(row["mean_f"]),
                "std_f":    float(row["std_f"]),
                "min_f":    float(row["min_f"]),
                "max_f":    float(row["max_f"]),
            })
    summary = None
    if _ANOVA_SUMMARY.is_file():
        with open(_ANOVA_SUMMARY, newline="") as fh:
            row = next(csv.DictReader(fh))
            summary = {
                "F_stat":     float(row["F_stat"]),
                "p_value":    float(row["p_value"]),
                "df_between": int(float(row["df_between"])),
                "df_within":  int(float(row["df_within"])),
            }
    return _json({"available": True, "groups": groups, "summary": summary})
```

- [ ] **Step 3: Smoke-test the endpoint**

Start the server and curl:
```
python -m uvicorn cl3o.ui.backend.app:app --port 8000
curl http://localhost:8000/api/sensitivity
```
Expected: JSON with `available: true`, `groups` array of 5 items, `summary` object.

- [ ] **Step 4: Commit**

```
git add src/cl3o/ui/backend/app.py
git commit -m "feat(backend): add GET /api/sensitivity ANOVA endpoint"
```

---

## Task 3 — Frontend: ANOVA sensitivity charts

**Files:**
- Modify: `src/cl3o/ui/frontend/src/types.ts`
- Modify: `src/cl3o/ui/frontend/src/api/client.ts`
- Modify: `src/cl3o/ui/frontend/src/plots/MiscPlot.tsx`

### Step 1 — Types

- [ ] Add to the end of `types.ts` (before the last `ViewKind` line):

```typescript
export interface AnovaGroup {
  group:  string;
  eta_sq: number;
  mean_f: number;
  std_f:  number;
  min_f:  number;
  max_f:  number;
}

export interface SensitivityData {
  available: boolean;
  groups?:   AnovaGroup[];
  summary?: {
    F_stat:     number;
    p_value:    number;
    df_between: number;
    df_within:  number;
  } | null;
}
```

### Step 2 — API client

- [ ] Add `SensitivityData` to the import line at the top of `client.ts`:
```typescript
import type {
  AnovaGroup, Forces, Info, Manifest, Mesh, Planform, RunSummary, Scene,
  SearchSpace, Section, SensitivityData, Stress, StressScene,
} from "../types";
```

- [ ] Add inside the `api` object (after `search`):
```typescript
  sensitivity: () => get<SensitivityData>("/sensitivity"),
```

### Step 3 — Replace `SensitivityView` in `MiscPlot.tsx`

- [ ] Remove the `SensKind` type and `SENS_KINDS` constant (lines 8–19), remove the
`sens` state, and remove the `<select>` inside the sensitivity toolbar block.

Replace the entire file with the following (keeping `ConvergenceView` and
`SearchSpaceView` unchanged — only touch the imports, `MiscPlot` toolbar, and
`SensitivityView`):

**Import block** — add `SensitivityData` and `AnovaGroup`:
```typescript
import { useEffect, useState } from "react";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { AnovaGroup, SearchSpace, SensitivityData } from "../types";
import Plot, { baseLayout, config } from "./Plot";
import { SEARCH_CMAP } from "./colors";
```

**TABS constant** — unchanged:
```typescript
const TABS: { key: "convergence" | "search" | "sensitivity"; label: string }[] = [
  { key: "convergence", label: "Convergence" },
  { key: "search",      label: "Search space" },
  { key: "sensitivity", label: "Sensitivity" },
];
```

**`MiscPlot` component** — remove the `sens` state and the `<select>`:
```typescript
export function MiscPlot() {
  const { manifest, distinctIndividuals, miscTab: tab, setMiscTab: setTab } = useStore();

  if (!manifest) return <div className="plot-loading">No run loaded…</div>;

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
      <div className="plot-toolbar">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={tab === t.key ? "active" : ""}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ flex: 1, minHeight: 0 }}>
        {tab === "convergence" && <ConvergenceView />}
        {tab === "search"      && <SearchSpaceView />}
        {tab === "sensitivity" && <SensitivityView />}
      </div>
    </div>
  );
}
```

**New `SensitivityView`** — add at the bottom (replace the old stub):
```typescript
function SensitivityView() {
  const [data, setData] = useState<SensitivityData | null>(null);
  const [err,  setErr]  = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api.sensitivity()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, []);

  if (err)  return <div className="plot-error">{err}</div>;
  if (!data) return <div className="plot-loading">Loading sensitivity data…</div>;
  if (!data.available || !data.groups?.length) {
    return (
      <div className="plot-empty">
        ANOVA data not found. Run <code>tools/sensitivity_analysis.py</code> first
        to generate <code>tools/output/sensitivity/anova_results.csv</code>.
      </div>
    );
  }

  const sorted = [...data.groups].sort((a, b) => a.eta_sq - b.eta_sq);
  const groups = sorted.map((g) => g.group);
  const etaSq  = sorted.map((g) => g.eta_sq);

  const subtitle = data.summary
    ? `F = ${data.summary.F_stat.toFixed(3)}   p = ${data.summary.p_value.toExponential(3)}`
    : "";

  // Box statistics approximated from mean/std (Gaussian assumption for Q1/Q3).
  const boxTraces = sorted.map((g: AnovaGroup, _i: number) => {
    const q1 = g.mean_f - 0.6745 * g.std_f;
    const q3 = g.mean_f + 0.6745 * g.std_f;
    const iqr = q3 - q1;
    return {
      type:  "box" as const,
      name:  g.group,
      q1:    [q1],
      median:[g.mean_f],
      q3:    [q3],
      lowerfence: [Math.max(g.min_f, q1 - 1.5 * iqr)],
      upperfence: [Math.min(g.max_f, q3 + 1.5 * iqr)],
      mean:  [g.mean_f],
      sd:    [g.std_f],
      orientation: "h" as const,
    };
  });

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", gap: 4 }}>
      {/* η² bar chart */}
      <div style={{ flex: "0 0 45%", minHeight: 0 }}>
        <Plot
          data={[{
            type: "bar",
            orientation: "h",
            x: etaSq,
            y: groups,
            marker: { color: etaSq, colorscale: "Viridis", showscale: false },
            hovertemplate: "%{y}: η² = %{x:.4f}<extra></extra>",
          }]}
          layout={{
            ...baseLayout,
            margin: { l: 120, r: 24, t: 40, b: 40 },
            title: { text: `η² por grupo   —   ${subtitle}`, font: { size: 12 } },
            xaxis: { title: "η²", gridcolor: "#1f2838", zeroline: true, zerolinecolor: "#3a4460" },
            yaxis: { gridcolor: "#1f2838", automargin: true },
          }}
          config={config}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>

      {/* Box-plot fitness distribution */}
      <div style={{ flex: "0 0 50%", minHeight: 0 }}>
        <Plot
          data={boxTraces}
          layout={{
            ...baseLayout,
            margin: { l: 120, r: 24, t: 8, b: 48 },
            showlegend: false,
            xaxis: { title: "fitness f [kg]", gridcolor: "#1f2838", zeroline: false },
            yaxis: { gridcolor: "#1f2838", automargin: true },
            boxmode: "group",
          }}
          config={config}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Typecheck**

```
cd src/cl3o/ui/frontend && npm run build
```
Expected: clean build.

- [ ] **Step 5: Commit**

```
git add src/cl3o/ui/frontend/src/types.ts \
        src/cl3o/ui/frontend/src/api/client.ts \
        src/cl3o/ui/frontend/src/plots/MiscPlot.tsx
git commit -m "feat(ui): ANOVA sensitivity charts in Misc tab"
```

---

## Task 4 — Backend: Tsai-Wu R surface

**Files:**
- Modify: `src/cl3o/ui/backend/surface.py`
- Modify: `src/cl3o/ui/backend/app.py`

`R_panels` is stored on `rt.tsw` with shape `(m, 10, 2, nc)`:
`[element, panel, end(0=A/1=B), loadcase]`.
`R_booms` has the same shape `(m, 7, 2, nc)`.

- [ ] **Step 1: Add `build_tsw_surface` to `surface.py`**

Add after `build_stress_surface` (line ~428), before `_build_boom_rods`:

```python
def build_tsw_surface(rt, wing, lc: int = 0, end: str = "avg") -> dict:
    '''
    Build the Tsai-Wu strength-ratio (R) surface for one snapshot.

    Identical lofting geometry to build_stress_surface, but coloured by the
    Tsai-Wu strength ratio R (failure at R >= 1) instead of shear stress.

    Args:
        rt   : RuntimeData snapshot.
        wing : WingData (typed) for the run.
        lc   : Load case index.
        end  : "A" / "B" / "avg" -- element end selector.

    Returns:
        Dict with mesh3d vertices + triangle indices, per-face R intensity,
        boom rods coloured by boom R, and scalar limits.
    '''
    sec     = rt.sections.sec_data
    coord   = np.asarray(rt.mesh.coord, float)
    conn    = np.asarray(rt.mesh.conn, int)[:, :2]

    tsw = rt.tsw
    R_panels_arr = np.asarray(tsw.R_panels, float)  # (m, 10, 2, nc)
    R_booms_arr  = np.asarray(tsw.R_booms,  float)  # (m,  7, 2, nc)

    nc = R_panels_arr.shape[3] if R_panels_arr.ndim == 4 else 1
    lc = max(0, min(int(lc), nc - 1))

    end_idx = 0 if end == "A" else (1 if end == "B" else None)

    def _pick(arr: np.ndarray) -> np.ndarray:
        # arr shape: (m, n_items, 2, nc)
        if end_idx is not None:
            return arr[:, :, end_idx, lc]
        return 0.5 * (arr[:, :, 0, lc] + arr[:, :, 1, lc])

    R_p = _pick(R_panels_arr)   # (m, 10)
    R_b = _pick(R_booms_arr)    # (m,  7)

    mid_y  = 0.5 * (coord[conn[:, 0], 1] + coord[conn[:, 1], 1])
    left_e = sorted(range(conn.shape[0]), key=lambda e: abs(mid_y[e]))

    verts: list = []
    fi, fj, fk, fc = [], [], [], []

    for e in left_e:
        node_a, node_b = int(conn[e, 0]), int(conn[e, 1])
        y_a = float(sec[node_a].C[1])
        y_b = float(sec[node_b].C[1])
        n_panels = min(len(sec[node_a].T2), len(sec[node_b].T2))

        for jp in range(n_panels):
            pts_a = np.asarray(sec[node_a].T2[jp]["pts"], float)
            pts_b = np.asarray(sec[node_b].T2[jp]["pts"], float)
            r_val = float(R_p[e, jp]) if jp < R_p.shape[1] else np.nan

            na, nb_pts = pts_a.shape[0], pts_b.shape[0]
            n = min(na, nb_pts)
            if na != n:
                idx = np.round(np.linspace(0, na - 1, n)).astype(int)
                pts_a = pts_a[idx]
            if nb_pts != n:
                idx = np.round(np.linspace(0, nb_pts - 1, n)).astype(int)
                pts_b = pts_b[idx]

            base = len(verts)
            for i in range(n):
                verts.append([pts_a[i, 0], y_a, pts_a[i, 1]])
            for i in range(n):
                verts.append([pts_b[i, 0], y_b, pts_b[i, 1]])

            for i in range(n - 1):
                ia0, ia1 = base + i, base + i + 1
                ib0, ib1 = base + n + i, base + n + i + 1
                fi += [ia0, ia0]
                fj += [ia1, ib1]
                fk += [ib1, ib0]
                fc += [r_val, r_val]

    r_fin = np.array([v for v in fc if np.isfinite(v)], dtype=float)
    r_max = float(np.nanmax(r_fin)) if r_fin.size else 2.0
    r_min = float(np.nanmin(r_fin)) if r_fin.size else 0.0

    # Boom rods coloured by R (reuse _build_boom_rods with R_b as the "sigma" arg,
    # but add an extra nc-dim so the helper's indexing works).
    R_b_nc = R_b[:, :, np.newaxis]   # (m, 7, 1)  -- fake nc=1 dimension
    boom_rods, r_abs_booms = _build_boom_rods(R_b_nc, left_e, conn, sec, 0, 1)

    return {
        "vertices": np.asarray(verts, float) if verts else np.zeros((0, 3)),
        "i": fi, "j": fj, "k": fk,
        "intensity": fc,
        "r_max":     r_max,
        "r_min":     r_min,
        "boom_rods": boom_rods,
        "r_abs_booms": r_abs_booms,
        "n_elements":  len(left_e),
        "n_loadcases": nc,
    }
```

- [ ] **Step 2: Add the route to `app.py`**

Add after the existing `stress3d` route:
```python
@app.get("/api/runs/{run_id}/gen/{k}/tsw3d")
def tsw3d(run_id: str, k: int, lc: int = Query(0), end: str = Query("avg")):
    rt = _snapshot(run_id, k)
    wing = repo.get_wing_data(run_id)
    if wing is None:
        raise HTTPException(status_code=404, detail="wing data not found")
    return _json(surface.build_tsw_surface(rt, wing, lc=lc, end=end))
```

- [ ] **Step 3: Smoke-test**

```
python -m uvicorn cl3o.ui.backend.app:app --port 8000
curl "http://localhost:8000/api/runs/da62_opt-1/gen/0/tsw3d?lc=0&end=avg" | python -m json.tool | head -40
```
Expected: JSON with `vertices`, `i`, `j`, `k`, `intensity`, `r_max`, `r_min`.

- [ ] **Step 4: Commit**

```
git add src/cl3o/ui/backend/surface.py src/cl3o/ui/backend/app.py
git commit -m "feat(backend): Tsai-Wu R surface endpoint tsw3d"
```

---

## Task 5 — Frontend: Tsai-Wu R mode in StressPlot

**Files:**
- Modify: `src/cl3o/ui/frontend/src/state/store.ts`
- Modify: `src/cl3o/ui/frontend/src/types.ts`
- Modify: `src/cl3o/ui/frontend/src/api/client.ts`
- Modify: `src/cl3o/ui/frontend/src/plots/StressPlot.tsx`

### Step 1 — Extend `StressMode` in `store.ts`

- [ ] Line 39: change
```typescript
  stressMode: "stress" | "flux";
```
to
```typescript
  stressMode: "stress" | "flux" | "tsw";
```

- [ ] Line 80: change
```typescript
  setStressMode: (m: "stress" | "flux") => void;
```
to
```typescript
  setStressMode: (m: "stress" | "flux" | "tsw") => void;
```

- [ ] Line 187: change
```typescript
  setStressMode: (m) => set({ stressMode: m }),
```
(no code change needed here — TypeScript infers the type from the interface).

### Step 2 — Add `TswScene` to `types.ts`

- [ ] Add after the `StressScene` interface (after line 174):
```typescript
export interface TswScene {
  vertices:    Vec3[];
  i:           number[];
  j:           number[];
  k:           number[];
  intensity:   number[];   // R value per face
  r_max:       number;
  r_min:       number;
  boom_rods:   BoomRod[];  // rods coloured by R (using 'sigma' field for R values)
  r_abs_booms: number;
  n_elements:  number;
  n_loadcases: number;
}
```

### Step 3 — Add `tsw3d` to `client.ts`

- [ ] Add `TswScene` to the import:
```typescript
import type {
  AnovaGroup, Forces, Info, Manifest, Mesh, Planform, RunSummary, Scene,
  SearchSpace, Section, SensitivityData, Stress, StressScene, TswScene,
} from "../types";
```

- [ ] Add inside the `api` object (after `stress3d`):
```typescript
  tsw3d: (run: string, k: number, lc = 0, end = "avg") =>
    get<TswScene>(`/runs/${run}/gen/${k}/tsw3d?lc=${lc}&end=${end}`),
```

### Step 4 — Failure (R) mode in `StressPlot.tsx`

- [ ] Add `TswScene` to the import in `StressPlot.tsx`:
```typescript
import type { Mesh3D, StressScene, TswScene } from "../types";
```

- [ ] Add a green → yellow → red colormap constant after `FLUX_CMAP`:
```typescript
// Green (safe) → yellow (near failure) → red (failed) for Tsai-Wu R.
// R = 1 is the failure boundary; colorbar is centred there.
const TSW_CMAP: [number, string][] = [
  [0.000, "#1a7a3c"], [0.350, "#6dbf8a"], [0.500, "#ffd166"],
  [0.650, "#e8795a"], [1.000, "#b2182b"],
];
```

- [ ] Extend the `useStore` destructuring at the top of `StressPlot` to include `end`:
```typescript
const { runId, gen, loadcase, end, stressMode: mode, setStressMode: setMode, setNLoadcases } = useStore();
```
(It was already destructuring `end` — verify it's present. If not, add it.)

- [ ] Add `TswScene` state after the existing `StressScene` state:
```typescript
const [st,  setSt]  = useState<StressScene | null>(null);
const [tsw, setTsw] = useState<TswScene    | null>(null);
```

- [ ] Replace the single `useEffect` with two — one for `stress3d`, one for `tsw3d`:

```typescript
useEffect(() => {
  if (!runId || mode === "tsw") return;
  let alive = true;
  api.stress3d(runId, gen, loadcase, end)
    .then((ss) => {
      if (!alive) return;
      setSt(ss);
      setNLoadcases(ss.n_loadcases);
    })
    .catch((e) => alive && setErr(String(e)));
  return () => { alive = false; };
}, [runId, gen, loadcase, end, mode, setNLoadcases]);

useEffect(() => {
  if (!runId || mode !== "tsw") return;
  let alive = true;
  api.tsw3d(runId, gen, loadcase, end)
    .then((ts) => {
      if (!alive) return;
      setTsw(ts);
      setNLoadcases(ts.n_loadcases);
    })
    .catch((e) => alive && setErr(String(e)));
  return () => { alive = false; };
}, [runId, gen, loadcase, end, mode, setNLoadcases]);
```

- [ ] Update the loading guard after the `useEffect`s:
```typescript
if (err) return <div className="plot-error">{err}</div>;
if (mode !== "tsw" && !st)  return <div className="plot-loading">Loading stress state…</div>;
if (mode === "tsw"  && !tsw) return <div className="plot-loading">Loading failure state…</div>;
```

- [ ] Add the Tsai-Wu rendering block after the flux mode block (before the final return):

```typescript
// -------- Failure (R) mode --------
if (mode === "tsw" && tsw) {
  const panelMesh: Mesh3D = { vertices: tsw.vertices, i: tsw.i, j: tsw.j, k: tsw.k };
  // Centre the colorbar at R=1.0 (failure boundary).
  const rRange = Math.max(tsw.r_max - 1.0, 1.0 - tsw.r_min, 0.5);
  const rCmin  = 1.0 - rRange;
  const rCmax  = 1.0 + rRange;

  const tswTraces: Data[] = [
    meshTrace(panelMesh, {
      intensity: tsw.intensity,
      intensitymode: "cell",
      colorscale: TSW_CMAP,
      reversescale: false,
      cmin: rCmin,
      cmax: rCmax,
      colorbarTitle: "R [-]",
      colorbarY: 0.75,
      colorbarLen: 0.55,
      name: "Tsai-Wu R",
      hovertemplate: "R = %{intensity:.3f}<extra>panel</extra>",
    }),
  ];

  if (tsw.boom_rods && tsw.boom_rods.length > 0) {
    const rBoomAbs = tsw.r_abs_booms ?? 2.0;
    tsw.boom_rods.forEach((rod) => {
      tswTraces.push({
        type: "scatter3d",
        mode: "lines",
        x: rod.xyz.map((p) => p[0]),
        y: rod.xyz.map((p) => p[1]),
        z: rod.xyz.map((p) => p[2]),
        line: {
          color: rod.sigma as number[],
          colorscale: TSW_CMAP,
          reversescale: false,
          cmin: 0,
          cmax: rBoomAbs,
          width: 20,
          colorbar: {
            title: { text: "R boom [-]" },
            thickness: 12,
            x: 1.0,
            len: 0.45,
            y: 0.22,
          } as never,
        },
        name: rod.label,
        hovertemplate: `${rod.label}  R = %{line.color:.3f}<extra>boom</extra>`,
        showlegend: false,
      } as Data);
    });
  }

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
      <div className="plot-toolbar">
        <button onClick={() => setMode("stress")}>Stress</button>
        <button onClick={() => setMode("flux")}>Flux</button>
        <button className="active" onClick={() => setMode("tsw")}>Failure (R)</button>
      </div>
      <Plot
        data={tswTraces}
        layout={{ ...baseLayout, margin: { l: 0, r: 100, t: 8, b: 0 }, scene: scene3d, showlegend: false }}
        config={config}
        style={{ width: "100%", flex: 1 }}
        useResizeHandler
      />
    </div>
  );
}
```

- [ ] Update the existing toolbar buttons in the `stress` and `flux` return blocks to
include the new third button:

Stress mode toolbar:
```tsx
<div className="plot-toolbar">
  <button className="active" onClick={() => setMode("stress")}>Stress</button>
  <button onClick={() => setMode("flux")}>Flux</button>
  <button onClick={() => setMode("tsw")}>Failure (R)</button>
</div>
```

Flux mode toolbar:
```tsx
<div className="plot-toolbar">
  <button onClick={() => setMode("stress")}>Stress</button>
  <button className="active" onClick={() => setMode("flux")}>Flux</button>
  <button onClick={() => setMode("tsw")}>Failure (R)</button>
  <select ...>...</select>
</div>
```

- [ ] **Step 5: Typecheck**

```
cd src/cl3o/ui/frontend && npm run build
```
Expected: clean build.

- [ ] **Step 6: Commit**

```
git add src/cl3o/ui/frontend/src/state/store.ts \
        src/cl3o/ui/frontend/src/types.ts \
        src/cl3o/ui/frontend/src/api/client.ts \
        src/cl3o/ui/frontend/src/plots/StressPlot.tsx
git commit -m "feat(ui): Tsai-Wu R surface mode in Stress tab"
```

---

## Task 6 — Hover picker for GeometryPlot and MeshPlot

**Files:**
- Modify: `src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx`
- Modify: `src/cl3o/ui/frontend/src/plots/MeshPlot.tsx`

Currently `GeometryPlot` uses flat `color` (no intensity / no hover). The hover
in geometry will show the component name and laminate index via the `text` field
on the mesh trace. MeshPlot displacement needs `hovertemplate` added to its
intensity trace.

### GeometryPlot hover

- [ ] In `GeometryPlot.tsx`, in the `traces` array, add `hovertemplate` to each
meshTrace call. The flat-color branch of `meshTrace` currently sets
`hoverinfo: "skip"`. Instead, set name-based hover by replacing the calls:

```typescript
const traces: Data[] = [
  meshTrace(scene.surface, {
    color: matColor(lu?.ls1?.[0]),
    opacity: 0.65,
    name: `skin  ls1=${lu?.ls1?.[0] ?? "?"} / ls2=${lu?.ls2?.[0] ?? "?"}`,
  }),
  meshTrace(scene.front_spar, {
    color: matColor(lu?.lw1?.[0]),
    opacity: 0.65,
    name: `front spar  lw1=${lu?.lw1?.[0] ?? "?"}`,
  }),
  meshTrace(scene.rear_spar, {
    color: matColor(lu?.lw2?.[0]),
    opacity: 0.65,
    name: `rear spar  lw2=${lu?.lw2?.[0] ?? "?"}`,
  }),
  sparEdgeTrace(scene.front_spar, "#000000", "front spar edge"),
  sparEdgeTrace(scene.rear_spar, "#000000", "rear spar edge"),
  ...(scene.flanges ?? []).map((fl) =>
    meshTrace(fl, {
      color: matColor(fl.layup_idx),
      opacity: 0.65,
      name: `${fl.label}  lam=${fl.layup_idx}`,
    }),
  ),
];
```

- [ ] In `Plot.tsx`, inside `meshTrace`, in the `else` (flat-color) branch, remove
`t.hoverinfo = "skip"` and replace with `t.hovertemplate = "%{fullData.name}<extra></extra>"`:

```typescript
  } else {
    t.color = opts.color ?? "#4f8cff";
    t.hovertemplate = "%{fullData.name}<extra></extra>";
  }
```

### MeshPlot hover

- [ ] Read `src/cl3o/ui/frontend/src/plots/MeshPlot.tsx` to find the displacement
meshTrace call (the one with `intensity: station_intensity`). Add
`hovertemplate: "disp = %{intensity:.3g} mm<extra></extra>"` to its `MeshOpts`:

```typescript
meshTrace(mesh3d, {
  intensity: station_intensity,
  intensitymode: "vertex",
  colorscale: ...,
  cmin: ...,
  cmax: ...,
  colorbarTitle: ...,
  hovertemplate: "disp = %{intensity:.3g} mm<extra></extra>",
})
```

(Find the exact call and add the `hovertemplate` line — do not change any other options.)

- [ ] **Typecheck**

```
cd src/cl3o/ui/frontend && npm run build
```

- [ ] **Commit**

```
git add src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx \
        src/cl3o/ui/frontend/src/plots/MeshPlot.tsx \
        src/cl3o/ui/frontend/src/plots/Plot.tsx
git commit -m "feat(ui): hover picker for geometry and mesh displacement plots"
```

---

## Task 7 — Fix skin layup color (Geometry)

**Files:**
- Modify: `src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx`

Two bugs:
1. `ls2` is absent from `legendItems` — it's never shown even when `ls1 ≠ ls2`.
2. The skin surface shows only `ls1[0]` as a flat color; `ls2` is invisible.

### Fix 1: add `ls2` to the legend

- [ ] In `GeometryPlot.tsx`, in the `legendItems` computation, find the
`rootIdxList` (currently excludes `ls2`) and add `lu.ls2?.[0]`:

```typescript
const rootIdxList = [
  lu.ls1?.[0], lu.ls2?.[0], lu.lw1?.[0], lu.lw2?.[0],
  lu.lf1?.[0], lu.lf2?.[0], lu.lf3?.[0], lu.lf4?.[0],
];
```

### Fix 2: show ls2 as a second skin surface

The outer skin mesh (`scene.surface`) is a single lofted loop covering the
entire airfoil chord. `ls1` covers the upper skin (between the spars, seg2+seg3)
and `ls2` covers the lower skin (seg4+seg5+seg1). The mesh has `n_chord` points
per station going around the full loop. The upper half covers chord indices
`~0 .. n_chord/2` and the lower half covers the rest — but the exact split
depends on the T1 segment lengths.

The simplest approach that matches what users actually see: render the skin mesh
**twice** — once with `ls1` color and 100% opacity (upper surface), once with
`ls2` color and 100% opacity (lower surface), using a CSS trick to appear as
one surface. Since we can't easily split the vertex buffer without backend
support, we compromise: render the full skin once with `ls1` color AND add a
second trace of the full skin with `ls2` color at reduced opacity (0.3).
This gives a visual blend hint and shows `ls2` in the legend.

- [ ] In the `traces` array, replace the single skin meshTrace with two:

```typescript
// Upper skin (ls1) — primary, full opacity.
meshTrace(scene.surface, {
  color: matColor(lu?.ls1?.[0]),
  opacity: 0.65,
  name: `skin upper  ls1=${lu?.ls1?.[0] ?? "?"}`,
}),
// Lower skin (ls2) — secondary, lower opacity for visual separation.
meshTrace(scene.surface, {
  color: matColor(lu?.ls2?.[0]),
  opacity: 0.30,
  name: `skin lower  ls2=${lu?.ls2?.[0] ?? "?"}`,
}),
```

> **Note:** If `ls1 === ls2` both traces have the same color and the combined
> opacity is still visually correct. If they differ, the lower skin appears as
> a faint tint of ls2. A proper per-vertex split requires the backend to return
> a chord-fraction split index — file as a follow-up if needed.

- [ ] **Typecheck + commit**

```
cd src/cl3o/ui/frontend && npm run build
git add src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx
git commit -m "fix(ui): add ls2 to skin legend and secondary skin trace"
```

---

## Task 8 — Design vector table

**Files:**
- Modify: `src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx`

`info.optvars` is a `number[]` with length `11 * n_cpts + 3`. The layout:

| Block | Length | Variable | Has CP |
|-------|--------|----------|--------|
| 0 | n | xw1 | yes |
| 1 | n | xw2 | yes |
| 2–5 | 1 each | bf1_root … bf4_root | no |
| 6 | n-1 | tpr | yes |
| 7 | n | ls1 | yes |
| 8 | n | ls2 | yes |
| 9 | n | lw1 | yes |
| 10 | n | lw2 | yes |
| 11 | n | lf1 | yes |
| 12 | n | lf2 | yes |
| 13 | n | lf3 | yes |
| 14 | n | lf4 | yes |

### Step 1 — Build the label utility (pure function)

- [ ] Add a helper function near the top of `GeometryPlot.tsx` (after the imports):

```typescript
interface XRow { i: number; variable: string; cp: string; value: number }

function buildXRows(optvars: number[]): XRow[] {
  const D = optvars.length;
  const n = (D - 3) / 11;           // n_cpts
  if (!Number.isInteger(n) || n < 1) return [];
  const blocks: { name: string; len: number; hasCp: boolean }[] = [
    { name: "xw1",      len: n,     hasCp: true  },
    { name: "xw2",      len: n,     hasCp: true  },
    { name: "bf1_root", len: 1,     hasCp: false },
    { name: "bf2_root", len: 1,     hasCp: false },
    { name: "bf3_root", len: 1,     hasCp: false },
    { name: "bf4_root", len: 1,     hasCp: false },
    { name: "tpr",      len: n - 1, hasCp: true  },
    { name: "ls1",      len: n,     hasCp: true  },
    { name: "ls2",      len: n,     hasCp: true  },
    { name: "lw1",      len: n,     hasCp: true  },
    { name: "lw2",      len: n,     hasCp: true  },
    { name: "lf1",      len: n,     hasCp: true  },
    { name: "lf2",      len: n,     hasCp: true  },
    { name: "lf3",      len: n,     hasCp: true  },
    { name: "lf4",      len: n,     hasCp: true  },
  ];
  const rows: XRow[] = [];
  let i = 0;
  for (const blk of blocks) {
    for (let k = 0; k < blk.len; k++) {
      rows.push({
        i,
        variable: blk.name,
        cp:       blk.hasCp ? String(k) : "—",
        value:    optvars[i],
      });
      i++;
    }
  }
  return rows;
}
```

### Step 2 — Add `XVectorTable` component

- [ ] Add after `LayupTable` component in `GeometryPlot.tsx`:

```typescript
function XVectorTable({
  optvars,
  onClose,
}: {
  optvars: number[];
  onClose: () => void;
}) {
  const rows = buildXRows(optvars);
  return (
    <div className="layup-table-panel" style={{ left: 200 }}>
      <div className="layup-table-header">
        <span>Design vector X  (D = {optvars.length})</span>
        <button className="layup-table-close" onClick={onClose}>✕</button>
      </div>
      <div className="layup-table-scroll">
        <table className="layup-table">
          <thead>
            <tr><th>i</th><th>Variable</th><th>CP</th><th>Value</th></tr>
          </thead>
          <tbody>
            {rows.map(({ i, variable, cp, value }) => (
              <tr key={i}>
                <td className="idx">{i}</td>
                <td className="name">{variable}</td>
                <td>{cp}</td>
                <td>{value != null ? value.toFixed(4) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

### Step 3 — Wire up the button in `GeometryPlot`

- [ ] Add `showXTable` state alongside `showTable`:
```typescript
const [showXTable, setShowXTable] = useState(false);
```

- [ ] Add `info` from the store:
```typescript
const { runId, gen, publish, info } = useStore();
```

- [ ] In the JSX return, add the second button and overlay panel directly after the
`Layups` button block:

```tsx
{/* X[] table toggle button */}
<button
  className="layup-table-btn"
  style={{ left: 80 }}
  onClick={() => setShowXTable((v) => !v)}
  title="Show design vector table"
>
  X[ ]
</button>

{/* Design vector overlay */}
{showXTable && info?.optvars && (
  <XVectorTable
    optvars={info.optvars}
    onClose={() => setShowXTable(false)}
  />
)}
```

- [ ] **Typecheck + commit**

```
cd src/cl3o/ui/frontend && npm run build
git add src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx
git commit -m "feat(ui): design vector X[] table in Geometry tab"
```

---

## Task 9 — Snapshot: backend endpoint + custom Plotly button

**Files:**
- Modify: `src/cl3o/ui/backend/app.py`
- Create: `src/cl3o/ui/frontend/src/hooks/useSnapshotButton.ts`
- Modify: `src/cl3o/ui/frontend/src/plots/Plot.tsx`
- Modify: `src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx`
- Modify: `src/cl3o/ui/frontend/src/plots/StressPlot.tsx`
- Modify: `src/cl3o/ui/frontend/src/plots/MeshPlot.tsx`
- Modify: `src/cl3o/ui/frontend/src/plots/MiscPlot.tsx`

### Step 1 — Backend endpoint

- [ ] Add imports to `app.py` (if not already present):
```python
import base64
from datetime import datetime
```

- [ ] Add the snaps endpoint in `app.py` before the static-SPA mount:
```python
from pydantic import BaseModel as _BaseModel

class _SnapBody(_BaseModel):
    run_id: str
    gen:    int
    view:   str
    data:   str   # base64-encoded PNG

@app.post("/api/snaps")
def save_snap(body: _SnapBody):
    from cl3o.paths import OUTPUTS_DIR
    snaps_dir = OUTPUTS_DIR / "snaps"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"{body.run_id}_gen{body.gen:04d}_{body.view}_{ts}.png"
    path  = snaps_dir / fname
    raw   = base64.b64decode(body.data)
    path.write_bytes(raw)
    return _json({"path": str(path)})
```

- [ ] **Smoke-test** (optional — test via the UI button in a later step)

### Step 2 — Snapshot hook

- [ ] Create `src/cl3o/ui/frontend/src/hooks/useSnapshotButton.ts`:

```typescript
import Plotly from "plotly.js-dist-min";

export function useSnapshotButton(runId: string | null, gen: number, view: string) {
  return {
    name:  "save-snapshot",
    title: "Save snapshot (download + server)",
    icon:  Plotly.Icons.camera,
    click: async (gd: HTMLElement) => {
      const dataUrl = await Plotly.toImage(gd, { format: "png", scale: 2 });

      // 1. Browser download
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `cl3o_${runId ?? "run"}_gen${String(gen).padStart(4, "0")}_${view}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      // 2. Server save (fire-and-forget)
      if (runId) {
        const base64 = dataUrl.split(",")[1] ?? dataUrl;
        fetch("/api/snaps", {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ run_id: runId, gen, view, data: base64 }),
        }).catch(console.error);
      }
    },
  };
}
```

### Step 3 — Remove native toImage button from `config` in `Plot.tsx`

- [ ] In `Plot.tsx`, change `config` to remove the native camera button and
export `modeBarButtonsToAdd` as an empty default:

```typescript
export const config = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["toImage"] as string[],
  // 2x raster so custom snapshot button produces a publishable PNG.
  toImageButtonOptions: { format: "png" as const, scale: 2 },
};
```

### Step 4 — Wire snap button into each plot

For each of the four plot components, add the snapshot button to the Plotly
`config`. Pattern:

```typescript
// At the top of each component, get runId + gen + view from the store.
// Add the hook call. Pass snapConfig instead of config to <Plot>.

import { useSnapshotButton } from "../hooks/useSnapshotButton";

// Inside the component function:
const snapBtn = useSnapshotButton(runId, gen, "geometry"); // use correct view name
const snapConfig = { ...config, modeBarButtonsToAdd: [snapBtn] };

// In the <Plot> render:
<Plot ... config={snapConfig} ... />
```

- [ ] **GeometryPlot.tsx**: view name = `"geometry"`. Add `snapBtn`/`snapConfig`
and replace all `config` references in the JSX with `snapConfig`.

- [ ] **StressPlot.tsx**: view name = `"stress"`. Apply the same pattern for all
three `<Plot>` instances (stress mode, flux mode, tsw mode).

- [ ] **MeshPlot.tsx**: view name = `"mesh"`. Apply the same pattern.

- [ ] **MiscPlot.tsx**: Convergence, SearchSpace, and Sensitivity sub-views all
use Plotly. Each sub-component needs `runId` + `gen` from the store + its own
`snapConfig`. View names: `"convergence"`, `"search"`, `"sensitivity"`.

### Step 5 — Typecheck + commit

- [ ] **Typecheck**

```
cd src/cl3o/ui/frontend && npm run build
```
Expected: clean build. Fix any type errors (e.g., `modeBarButtonsToAdd` must be
typed as `any[]` if Plotly's TS types are restrictive — add `as any` cast if needed).

- [ ] **Commit**

```
git add src/cl3o/ui/backend/app.py \
        src/cl3o/ui/frontend/src/hooks/useSnapshotButton.ts \
        src/cl3o/ui/frontend/src/plots/Plot.tsx \
        src/cl3o/ui/frontend/src/plots/GeometryPlot.tsx \
        src/cl3o/ui/frontend/src/plots/StressPlot.tsx \
        src/cl3o/ui/frontend/src/plots/MeshPlot.tsx \
        src/cl3o/ui/frontend/src/plots/MiscPlot.tsx
git commit -m "feat(ui): custom snapshot button saves to outputs/snaps/"
```

---

## Self-Review Checklist

- [x] **T1 (stress default B)**: covered by store.ts change.
- [x] **T2 (ANOVA backend)**: covered by `GET /api/sensitivity` route.
- [x] **T3 (ANOVA frontend)**: covered by SensitivityView with η² bar + box-plot.
- [x] **T4 (R backend)**: covered by `build_tsw_surface` + `GET /api/.../tsw3d`.
- [x] **T5 (R frontend)**: covered by TswScene type, tsw3d client, StressPlot mode.
- [x] **T6 (hover)**: covered for GeometryPlot (name-based) and MeshPlot (intensity).
- [x] **T7 (skin layup bug)**: ls2 added to legend; secondary skin trace added.
- [x] **T8 (design vector table)**: `buildXRows` + `XVectorTable` + X[] button.
- [x] **T9 (snapshot)**: backend endpoint + hook + config + wired into all plots.
- [x] **Type consistency**: `TswScene` defined in T5 and imported in StressPlot.
  `SensitivityData`/`AnovaGroup` defined in T3 and imported in MiscPlot.
- [x] **No placeholders**: all steps have concrete code.
