import { useEffect, useState } from "react";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { SearchSpace } from "../types";
import Plot, { baseLayout } from "./Plot";
import { useSnapshotConfig } from "../hooks/useSnapshotButton";
import { SEARCH_CMAP } from "./colors";

// PCA projection (2-D or 3-D) of the distinct design vectors visited by DE.
export function SearchSpaceView() {
  const { runId } = useStore();
  const snapConfig = useSnapshotConfig("search");
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
            config={snapConfig}
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
          config={snapConfig}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
