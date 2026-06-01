import { useEffect, useState } from "react";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { AnovaGroup, SearchSpace, SensitivityData } from "../types";
import Plot, { baseLayout, config } from "./Plot";
import { SEARCH_CMAP } from "./colors";

const TABS: { key: "convergence" | "search" | "sensitivity"; label: string }[] = [
  { key: "convergence", label: "Convergence" },
  { key: "search",      label: "Search space" },
  { key: "sensitivity", label: "Sensitivity" },
];

export function MiscPlot() {
  const { manifest, miscTab: tab, setMiscTab: setTab } = useStore();

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

function ConvergenceView() {
  const { manifest } = useStore();
  if (!manifest) return null;
  const k = (manifest.best_f_hist || []).map((_, i) => i);
  return (
    <Plot
      data={[
        {
          x: k, y: manifest.best_f_hist as number[],
          type: "scatter", mode: "lines",
          name: "best f",
          line: { color: "#4f8cff", width: 2 },
        },
        {
          x: k, y: manifest.mean_f_hist as number[],
          type: "scatter", mode: "lines",
          name: "mean f",
          line: { color: "#ffd166", width: 1.2, dash: "dot" },
        },
        {
          x: k, y: manifest.std_f_hist as number[],
          type: "scatter", mode: "lines", yaxis: "y2",
          name: "std f",
          line: { color: "#17b18a", width: 1.2 },
        },
      ]}
      layout={{
        ...baseLayout,
        showlegend: true,
        legend: { x: 0.5, xanchor: "center", y: 1.02, yanchor: "bottom", orientation: "h", font: { size: 11 } },
        margin: { l: 56, r: 56, t: 48, b: 48 },
        xaxis: { title: "generation", gridcolor: "#1f2838", zeroline: false },
        yaxis: { title: "fitness z(X)", gridcolor: "#1f2838", zeroline: false },
        yaxis2: {
          title: "std f", overlaying: "y", side: "right",
          showgrid: false, zeroline: false,
        },
      }}
      config={config}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}

function SearchSpaceView() {
  const { runId } = useStore();
  const [data, setData] = useState<SearchSpace | null>(null);
  const [err, setErr]   = useState<string | null>(null);
  const [show3D, setShow3D] = useState(false);

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    setErr(null);
    api.search(runId)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [runId]);

  if (err) return <div className="plot-error">{err}</div>;
  if (!data) return <div className="plot-loading">Loading search space…</div>;
  if (!data.x.length) {
    return (
      <div className="plot-empty">
        Need at least 2 distinct individuals to project the search space.
      </div>
    );
  }

  const f = data.f.map((v) => (v == null ? NaN : v));
  const finite = f.filter((v) => Number.isFinite(v)) as number[];
  const fMin = finite.length ? Math.min(...finite) : 0;
  const fMax = finite.length ? Math.max(...finite) : 1;
  const ev = data.explained_variance;
  // Show 3D toggle whenever we have enough distinct PC3 values to be non-trivial.
  const hasPC3 = (data.z ?? []).length === data.x.length &&
                 (data.z ?? []).some((v) => Math.abs(v) > 1e-10);

  if (show3D && hasPC3) {
    const z = data.z ?? [];
    return (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
        <div className="plot-toolbar" style={{ justifyContent: "flex-end" }}>
          <button onClick={() => setShow3D(false)}>2D</button>
          <button className="active" onClick={() => setShow3D(true)}>3D</button>
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          <Plot
            data={[
              {
                x: data.x, y: data.y, z,
                type: "scatter3d", mode: "lines",
                line: { color: "#ffd166", width: 2 },
                opacity: 0.55,
                name: "DE trajectory",
                hoverinfo: "skip",
              } as never,
              {
                x: data.x, y: data.y, z,
                type: "scatter3d", mode: "markers",
                marker: {
                  color: f as number[],
                  colorscale: SEARCH_CMAP,
                  reversescale: true,
                  cmin: fMin, cmax: fMax,
                  size: 5,
                  line: { color: "#1a1f2b", width: 1 },
                  colorbar: { title: { text: "fitness z(X)" }, thickness: 12 },
                },
                text: data.gen.map((g, i) => `distinct #${i + 1}<br>gen ${g}`),
                hovertemplate: "%{text}<br>z = %{marker.color:.4g}<extra></extra>",
                name: "distinct individuals",
              } as never,
            ]}
            layout={{
              ...baseLayout,
              showlegend: false,
              margin: { l: 0, r: 80, t: 48, b: 0 },
              scene: {
                xaxis: { title: `PC1 (${(ev[0] * 100).toFixed(1)}%)`, color: "#7c8aa5", gridcolor: "#1f2838", backgroundcolor: "rgba(0,0,0,0)" },
                yaxis: { title: `PC2 (${(ev[1] * 100).toFixed(1)}%)`, color: "#7c8aa5", gridcolor: "#1f2838", backgroundcolor: "rgba(0,0,0,0)" },
                zaxis: { title: `PC3 (${((ev[2] ?? 0) * 100).toFixed(1)}%)`, color: "#7c8aa5", gridcolor: "#1f2838", backgroundcolor: "rgba(0,0,0,0)" },
              } as never,
            }}
            config={config}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
          />
        </div>
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
      <div className="plot-toolbar" style={{ justifyContent: "flex-end" }}>
        <button className={!show3D ? "active" : ""} onClick={() => setShow3D(false)}>2D</button>
        <button className={show3D ? "active" : ""} onClick={() => setShow3D(true)} disabled={!hasPC3} title={hasPC3 ? "" : "Need ≥3 distinct individuals for 3D"}>3D</button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <Plot
          data={[
            {
              x: data.x, y: data.y,
              type: "scatter", mode: "lines",
              line: { color: "#ffd166", width: 1.2 },
              opacity: 0.55,
              name: "DE trajectory",
              hoverinfo: "skip",
            },
            {
              x: data.x, y: data.y,
              type: "scatter", mode: "markers",
              marker: {
                color: f as number[],
                colorscale: SEARCH_CMAP,
                reversescale: true,
                cmin: fMin, cmax: fMax,
                size: 10,
                line: { color: "#1a1f2b", width: 1 },
                colorbar: { title: { text: "fitness z(X)" }, thickness: 12 },
              },
              text: data.gen.map((g, i) => `distinct #${i + 1}<br>gen ${g}`),
              hovertemplate: "%{text}<br>z = %{marker.color:.4g}<extra></extra>",
              name: "distinct individuals",
            },
            {
              x: [data.x[0]], y: [data.y[0]],
              type: "scatter", mode: "text+markers",
              marker: { color: "#17b18a", size: 14, symbol: "circle-open", line: { width: 2 } },
              text: ["start"], textposition: "top center",
              textfont: { color: "#17b18a", size: 10 },
              hoverinfo: "skip", showlegend: false,
            },
            {
              x: [data.x[data.x.length - 1]], y: [data.y[data.y.length - 1]],
              type: "scatter", mode: "text+markers",
              marker: { color: "#e6786a", size: 14, symbol: "x", line: { width: 2 } },
              text: ["end"], textposition: "bottom center",
              textfont: { color: "#e6786a", size: 10 },
              hoverinfo: "skip", showlegend: false,
            },
          ]}
          layout={{
            ...baseLayout,
            showlegend: false,
            margin: { l: 56, r: 80, t: 24, b: 48 },
            xaxis: {
              title: `PC1 (${(ev[0] * 100).toFixed(1)} %)`,
              gridcolor: "#1f2838", zeroline: false,
            },
            yaxis: {
              title: `PC2 (${(ev[1] * 100).toFixed(1)} %)`,
              gridcolor: "#1f2838", zeroline: false, scaleanchor: "x", scaleratio: 1,
            },
            annotations: [{
              x: 0.5, y: 1.04, xref: "paper", yref: "paper",
              xanchor: "center", yanchor: "bottom", showarrow: false,
              text: `${data.n_distinct} distinct · explained var = ${((ev[0] + ev[1]) * 100).toFixed(1)} %`,
              font: { color: "#7c8aa5", size: 11 },
            }],
          }}
          config={config}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>
    </div>
  );
}

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
      {/* eta^2 bar chart */}
      <div style={{ flex: "0 0 45%", minHeight: 0 }}>
        <Plot
          data={[{
            type: "bar",
            orientation: "h",
            x: etaSq,
            y: groups,
            marker: { color: etaSq, colorscale: "Viridis", showscale: false },
            hovertemplate: "%{y}: eta2 = %{x:.4f}<extra></extra>",
          }]}
          layout={{
            ...baseLayout,
            margin: { l: 120, r: 24, t: 40, b: 40 },
            title: { text: `eta2 por grupo   --   ${subtitle}`, font: { size: 12 } },
            xaxis: { title: "eta2", gridcolor: "#1f2838", zeroline: true, zerolinecolor: "#3a4460" },
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
          data={boxTraces as never[]}
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
