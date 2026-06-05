import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { Mesh3D, StressScene, TswScene } from "../types";
import Plot, { baseLayout, meshTrace, scene3d } from "./Plot";
import { useSnapshotConfig } from "../hooks/useSnapshotButton";
import {
  TAU_CMAP, SIG_CMAP, FLUX_CMAP, TSW_CMAP, FLUX_OPTIONS,
  boomRodTraces, type FluxKey,
} from "./stressTraces";

export function StressPlot() {
  const { runId, gen, loadcase, end, stressMode: mode, setStressMode: setMode, setNLoadcases } = useStore();
  const [fluxKey, setFluxKey] = useState<FluxKey>("flux_qsX");
  const [st,  setSt]  = useState<StressScene | null>(null);
  const [tsw, setTsw] = useState<TswScene    | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const snapConfig = useSnapshotConfig("stress");

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

  if (err) return <div className="plot-error">{err}</div>;
  if (mode !== "tsw" && !st)  return <div className="plot-loading">Loading stress state…</div>;
  if (mode === "tsw"  && !tsw) return <div className="plot-loading">Loading failure state…</div>;

  // -------- Stress mode --------
  if (mode === "stress" && st) {
    const panelMesh: Mesh3D = { vertices: st.vertices, i: st.i, j: st.j, k: st.k };
    const traces: Data[] = [
      meshTrace(panelMesh, {
        intensity: st.intensity,
        intensitymode: "cell",
        colorscale: TAU_CMAP,
        reversescale: false,
        cmin: -st.tau_abs,
        cmax: st.tau_abs,
        colorbarTitle: "τ [MPa]",
        colorbarY: 0.75,
        colorbarLen: 0.55,
        name: "panel shear τ",
        hovertemplate: "τ = %{intensity:.3g} MPa<extra>panel</extra>",
      }),
    ];

    if (st.boom_rods && st.boom_rods.length > 0) {
      const sigAbs = st.sigma_abs ?? 1.0;
      traces.push(...boomRodTraces(st.boom_rods, {
        colorscale: SIG_CMAP,
        cmin: -sigAbs,
        cmax: sigAbs,
        colorbarTitle: "σ [MPa]",
        hoverValue: "σ = %{line.color:.3g} MPa",
      }));
    }

    return (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
        <div className="plot-toolbar">
          <button className="active" onClick={() => setMode("stress")}>Stress</button>
          <button onClick={() => setMode("flux")}>Flux</button>
          <button onClick={() => setMode("tsw")}>Failure (R)</button>
        </div>
        <Plot
          data={traces}
          layout={{ ...baseLayout, margin: { l: 0, r: 100, t: 8, b: 0 }, scene: scene3d, showlegend: false, uirevision: `stress:${runId}` }}
          config={snapConfig}
          style={{ width: "100%", flex: 1 }}
          useResizeHandler
        />
      </div>
    );
  }

  // -------- Flux mode --------
  if (mode === "flux" && st) {
    const panelMesh: Mesh3D = { vertices: st.vertices, i: st.i, j: st.j, k: st.k };
    const fluxRaw    = st[fluxKey] ?? [];
    const fluxAbsKey = (fluxKey + "_abs") as keyof StressScene;
    const fluxAbs    = (st[fluxAbsKey] as number) || 1.0;

    const traces: Data[] = [
      meshTrace(panelMesh, {
        intensity: fluxRaw as number[],
        intensitymode: "cell",
        colorscale: FLUX_CMAP,
        reversescale: false,
        cmin: -fluxAbs,
        cmax: fluxAbs,
        colorbarTitle: "q [N/mm]",
        colorbarY: 0.5,
        colorbarLen: 0.9,
        name: "panel flux q",
        hovertemplate: "q = %{intensity:.4g} mm⁻¹<extra>panel</extra>",
      }),
    ];

    return (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
        <div className="plot-toolbar">
          <button onClick={() => setMode("stress")}>Stress</button>
          <button className="active" onClick={() => setMode("flux")}>Flux</button>
          <button onClick={() => setMode("tsw")}>Failure (R)</button>
          <select value={fluxKey} onChange={(e) => setFluxKey(e.target.value as FluxKey)}>
            {FLUX_OPTIONS.map(({ key, label }) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>
        </div>
        <Plot
          data={traces}
          layout={{ ...baseLayout, margin: { l: 0, r: 80, t: 8, b: 0 }, scene: scene3d, showlegend: false, uirevision: `stress:${runId}` }}
          config={snapConfig}
          style={{ width: "100%", flex: 1 }}
          useResizeHandler
        />
      </div>
    );
  }

  // -------- Failure (R) mode --------
  if (mode === "tsw" && tsw) {
    const panelMesh: Mesh3D = { vertices: tsw.vertices, i: tsw.i, j: tsw.j, k: tsw.k };
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
      tswTraces.push(...boomRodTraces(tsw.boom_rods, {
        colorscale: TSW_CMAP,
        cmin: 0,
        cmax: rBoomAbs,
        colorbarTitle: "R boom [-]",
        hoverValue: "R = %{line.color:.3f}",
      }));
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
          layout={{ ...baseLayout, margin: { l: 0, r: 100, t: 8, b: 0 }, scene: scene3d, showlegend: false, uirevision: `stress:${runId}` }}
          config={snapConfig}
          style={{ width: "100%", flex: 1 }}
          useResizeHandler
        />
      </div>
    );
  }

  return null;
}
