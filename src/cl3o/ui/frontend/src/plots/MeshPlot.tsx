import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { Forces, Scene } from "../types";
import Plot, { baseLayout, config, meshTrace, scene3d } from "./Plot";
import type { Mesh3D } from "../types";
import { FEMAP_CMAP } from "./colors";

export const DISP_LABEL: Record<string, string> = {
  u: "T1 · u [mm]", v: "T2 · v [mm]", w: "T3 · w [mm]", t: "Total Translation [mm]",
  rx: "R1 · θx [deg]", ry: "R2 · θy [deg]", rz: "R3 · θz [deg]", r: "Total Rotation [deg]",
};

function compRange(
  station_disp: Record<string, number[]> | undefined,
  comp: string,
): [number, number] {
  if (!station_disp) return [-1, 1];
  const arr = station_disp[comp];
  if (!arr || arr.length === 0) return [-1, 1];
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) {
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!isFinite(lo)) return [-1, 1];
  if (lo === hi) { const m = Math.abs(lo) * 0.1 || 1; return [lo - m, hi + m]; }
  return [lo, hi];
}

// Mesh post-processing view: deformed wing surface colormapped by a
// displacement component, OR internal-force beam diagrams along the span.
export function MeshPlot() {
  const s = useStore();
  const {
    runId, gen, field, contourComp, forceFrame, forceComp,
    loadcase, scale, setNLoadcases, colorScaleFixed, colorMin, colorMax,
  } = s;
  const [scene, setScene] = useState<Scene | null>(null);
  const [forces, setForces] = useState<Forces | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    setErr(null);
    if (field === "disp") {
      api.geometry(runId, gen, true, loadcase, scale)
        .then((sc) => alive && setScene(sc))
        .catch((e) => alive && setErr(String(e)));
    } else {
      api.forces(runId, gen, loadcase)
        .then((f) => { if (alive) { setForces(f); setNLoadcases(f.n_loadcases); } })
        .catch((e) => alive && setErr(String(e)));
    }
    return () => { alive = false; };
  }, [runId, gen, field, loadcase, scale, setNLoadcases]);

  if (err) return <div className="plot-error">{err}</div>;

  // -------- Displacement: deformed surface colormapped by component --------
  if (field === "disp") {
    if (!scene) return <div className="plot-loading">Loading deformed mesh…</div>;
    const surf = scene.surface;
    const sd = surf.station_disp?.[contourComp] ?? [];
    const nC = surf.n_chord ?? 1;
    const nS = surf.n_span ?? sd.length;
    const intensity = new Array(nC * nS);
    for (let st = 0; st < nS; st++)
      for (let c = 0; c < nC; c++) intensity[st * nC + c] = sd[st];

    const fixed = colorScaleFixed && colorMin != null && colorMax != null;
    const [autoMin, autoMax] = compRange(surf.station_disp, contourComp);
    const cmin = fixed ? colorMin! : autoMin;
    const cmax = fixed ? colorMax! : autoMax;

    // Build a per-vertex intensity for any strip-style mesh (n_chord=2
    // along the span), pulling each station's contour value from the
    // same station_disp array used by the skin.
    const stripIntensity = (m: Mesh3D): number[] => {
      const n_s = m.n_span ?? Math.floor(m.vertices.length / 2);
      const out = new Array(n_s * 2);
      for (let s = 0; s < n_s; s++) {
        const v = sd[s] ?? 0;
        out[s * 2]     = v;
        out[s * 2 + 1] = v;
      }
      return out;
    };

    const traces: Data[] = [
      meshTrace(surf, {
        intensity, colorscale: FEMAP_CMAP, opacity: 0.97,
        cmin, cmax,
        colorbarTitle: DISP_LABEL[contourComp] ?? contourComp, name: "deformed",
      }),
      meshTrace(scene.front_spar, {
        intensity: stripIntensity(scene.front_spar),
        colorscale: FEMAP_CMAP, opacity: 0.97, cmin, cmax,
        name: "front spar", showscale: false,
      }),
      meshTrace(scene.rear_spar, {
        intensity: stripIntensity(scene.rear_spar),
        colorscale: FEMAP_CMAP, opacity: 0.97, cmin, cmax,
        name: "rear spar", showscale: false,
      }),
      ...(scene.flanges ?? []).map((fl) => meshTrace(fl, {
        intensity: stripIntensity(fl),
        colorscale: FEMAP_CMAP, opacity: 0.97, cmin, cmax,
        name: fl.label, showscale: false,
      })),
    ];
    return (
      <Plot
        data={traces}
        layout={{
          ...baseLayout, showlegend: true, legend: { x: 0, y: 1, font: { size: 11 } },
          scene: scene3d, uirevision: `mesh:${runId}`,
        }}
        config={config}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    );
  }

  // -------- Internal forces: beam diagram (local + global) along span --------
  if (!forces) return <div className="plot-loading">Loading internal forces…</div>;
  const unit = forces.units[forceComp] ?? "";
  const frameData = forceFrame === "local" ? forces.local : forces.global;
  const series = frameData[forceComp] ?? [];

  const finite = series.filter((v): v is number => v != null && Number.isFinite(v));
  const dataMin = finite.length ? Math.min(...finite) : 0;
  const dataMax = finite.length ? Math.max(...finite) : 0;
  const yMin = Math.min(dataMin, -100);
  const yMax = Math.max(dataMax,  100);

  const traces: Data[] = [
    {
      x: forces.span, y: series, type: "scatter", mode: "lines+markers",
      line: { color: "#4f8cff", width: 2 }, marker: { size: 4 },
      name: `${forceComp} (${forceFrame})`,
      hovertemplate: `span %{x:.0f} mm<br>${forceComp} = %{y:.3g} ${unit}<extra>${forceFrame}</extra>`,
    },
  ];
  return (
    <Plot
      data={traces}
      layout={{
        ...baseLayout, showlegend: true, legend: { x: 0, y: 1, font: { size: 11 } },
        xaxis: { title: "span  |Y| from root [mm]", gridcolor: "#1f2838", zeroline: false },
        yaxis: { title: `${forceComp}  [${unit}]`, gridcolor: "#1f2838", zerolinecolor: "#33405a", range: [yMin, yMax] },
      }}
      config={config}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
