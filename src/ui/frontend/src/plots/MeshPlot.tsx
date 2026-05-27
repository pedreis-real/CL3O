import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { Forces, Scene } from "../types";
import Plot, { baseLayout, config, lineTrace, meshTrace, scene3d } from "./Plot";

export const DISP_LABEL: Record<string, string> = {
  u: "T1 · u [mm]", v: "T2 · v [mm]", w: "T3 · w [mm]", t: "|translation| [mm]",
  rx: "R1 · θx [rad]", ry: "R2 · θy [rad]", rz: "R3 · θz [rad]", r: "|rotation| [rad]",
};

// Mesh post-processing view: deformed wing surface colormapped by a
// displacement component, OR internal-force beam diagrams along the span.
export function MeshPlot() {
  const s = useStore();
  const { runId, gen, field, dispComp, forceFrame, forceComp, loadcase, scale, setNLoadcases } = s;
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
    const sd = surf.station_disp?.[dispComp] ?? [];
    const nC = surf.n_chord ?? 1;
    const nS = surf.n_span ?? sd.length;
    const intensity = new Array(nC * nS);
    for (let st = 0; st < nS; st++)
      for (let c = 0; c < nC; c++) intensity[st * nC + c] = sd[st];

    const traces: Data[] = [
      meshTrace(surf, {
        intensity, colorscale: "Turbo", opacity: 0.97,
        colorbarTitle: DISP_LABEL[dispComp] ?? dispComp, name: "deformed",
      }),
      meshTrace(scene.front_spar, { color: "#3a6ea5", opacity: 0.6, name: "front spar" }),
      meshTrace(scene.rear_spar, { color: "#9a3a3a", opacity: 0.6, name: "rear spar" }),
      { ...lineTrace(scene.centroid_line, "#ffd166", "centroid"), visible: "legendonly" } as Data,
      { ...lineTrace(scene.shear_line, "#17b18a", "shear centre"), visible: "legendonly" } as Data,
    ];
    return (
      <Plot
        data={traces}
        layout={{ ...baseLayout, showlegend: true, legend: { x: 0, y: 1, font: { size: 11 } }, scene: scene3d }}
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
