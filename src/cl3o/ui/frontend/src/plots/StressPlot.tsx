import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { Mesh3D, StressScene, TswScene } from "../types";
import Plot, { baseLayout, config, meshTrace, scene3d } from "./Plot";
import { useSnapshotButton } from "../hooks/useSnapshotButton";

// FEMAP-style rainbow colormap for shear stress tau.
const TAU_CMAP: [number, string][] = [
  [0.000, "#0000c8"], [0.200, "#0096ff"], [0.375, "#00e6ff"],
  [0.500, "#00c800"], [0.625, "#ffff00"], [0.800, "#ff6400"], [1.000, "#c80000"],
];

// Blue–white–red diverging colormap for normal stress sigma.
const SIG_CMAP: [number, string][] = [
  [0.000, "#1a6faf"], [0.250, "#6db0d8"], [0.500, "#f5f5f5"],
  [0.750, "#e8795a"], [1.000, "#b2182b"],
];

// Purple–white–green diverging colormap for shear flux.
const FLUX_CMAP: [number, string][] = [
  [0.000, "#7b2d8b"], [0.250, "#c47fc4"], [0.500, "#f5f5f5"],
  [0.750, "#6dbf8a"], [1.000, "#1a7a3c"],
];

// Green (safe) -> yellow (near failure) -> red (failed) for Tsai-Wu R.
const TSW_CMAP: [number, string][] = [
  [0.000, "#1a7a3c"], [0.350, "#6dbf8a"], [0.500, "#ffd166"],
  [0.650, "#e8795a"], [1.000, "#b2182b"],
];

type FluxKey = "flux_qsX" | "flux_qsZ" | "flux_qT" | "flux_qbX" | "flux_qbZ";
const FLUX_OPTIONS: { key: FluxKey; label: string }[] = [
  { key: "flux_qsX", label: "q·S_X (total)" },
  { key: "flux_qsZ", label: "q·S_Z (total)" },
  { key: "flux_qT",  label: "q·T  (total)"  },
  { key: "flux_qbX", label: "qb·S_X (open)" },
  { key: "flux_qbZ", label: "qb·S_Z (open)" },
];

export function StressPlot() {
  const { runId, gen, loadcase, end, stressMode: mode, setStressMode: setMode, setNLoadcases } = useStore();
  const [fluxKey, setFluxKey] = useState<FluxKey>("flux_qsX");
  const [st,  setSt]  = useState<StressScene | null>(null);
  const [tsw, setTsw] = useState<TswScene    | null>(null);
  const [err, setErr] = useState<string | null>(null);

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

  const snapBtn = useSnapshotButton(runId, gen, "stress");
  const snapConfig = { ...config, modeBarButtonsToAdd: [snapBtn] as any[] };

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
      st.boom_rods.forEach((rod) => {
        traces.push({
          type: "scatter3d",
          mode: "lines",
          x: rod.xyz.map((p) => p[0]),
          y: rod.xyz.map((p) => p[1]),
          z: rod.xyz.map((p) => p[2]),
          line: {
            color: rod.sigma as number[],
            colorscale: SIG_CMAP,
            reversescale: false,
            cmin: -sigAbs,
            cmax: sigAbs,
            width: 20,
            colorbar: {
              title: { text: "σ [MPa]" },
              thickness: 12,
              x: 1.0,
              len: 0.45,
              y: 0.22,
            } as never,
          },
          name: rod.label,
          hovertemplate: `${rod.label}  σ = %{line.color:.3g} MPa<extra>boom</extra>`,
          showlegend: false,
        } as Data);
      });
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
          layout={{ ...baseLayout, margin: { l: 0, r: 100, t: 8, b: 0 }, scene: scene3d, showlegend: false }}
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
          layout={{ ...baseLayout, margin: { l: 0, r: 80, t: 8, b: 0 }, scene: scene3d, showlegend: false }}
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
          config={snapConfig}
          style={{ width: "100%", flex: 1 }}
          useResizeHandler
        />
      </div>
    );
  }

  return null;
}
