import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { Section } from "../types";
import Plot, { baseLayout, config } from "./Plot";

const CELL_COLORS = ["rgba(79,140,255,0.12)", "rgba(46,204,113,0.12)", "rgba(255,209,102,0.12)"];

const PANEL_COLORS = [
    "#2D9CDB",
    "#26A76E",
    "#7A5AFF",
    "#FF3C9A",
    "#E56C00",
    "#E6CE00",
    "#F03C3C",
    "#63E3E3",
    "#2F7AFF",
    "#E0A62A",
]

export function SectionPlot() {
  const { runId, gen, station, publish } = useStore();
  const [sec, setSec] = useState<Section | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    api
      .section(runId, gen, station)
      .then((d) => {
        if (!alive) return;
        setSec(d);
        publish({ section: d });
      })
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [runId, gen, station, publish]);

  if (err) return <div className="plot-error">{err}</div>;
  if (!sec) return <div className="plot-loading">Loading section…</div>;

  const traces: Data[] = [];

  // Closed cells (filled background).
  sec.cells.forEach((c, i) => {
    traces.push({
      x: c.pts.map((p) => p[0]),
      y: c.pts.map((p) => p[1]),
      type: "scatter",
      mode: "lines",
      fill: "toself",
      fillcolor: CELL_COLORS[i % CELL_COLORS.length],
      line: { color: "rgba(120,140,170,0.35)", width: 1 },
      hoverinfo: "skip",
      name: c.label,
    });
  });

  // Panels (T2 walls) — each panel gets a distinct colour.
  // mode "lines+markers" ensures hover registers even on 2-point spar webs.
  sec.panels.forEach((p, i) => {
    const color = PANEL_COLORS[i % PANEL_COLORS.length];
    const tStr  = p.t != null ? `${p.t.toFixed(2)} mm` : "—";
    traces.push({
      x: p.pts.map((q) => q[0]),
      y: p.pts.map((q) => q[1]),
      type: "scatter",
      mode: "lines",
      line: { color, width: 2 },
      marker: { color, size: 4, opacity: 0.7 },
      hovertemplate: `${p.label ?? `p${i + 1}`}  t=${tStr}<extra></extra>`,
      name: p.label ?? `p${i + 1}`,
    });
  });

  // Booms (sized by area).
  const bx = sec.booms.xy.map((p) => p[0]);
  const by = sec.booms.xy.map((p) => p[1]);
  const amax = Math.max(1e-9, ...sec.booms.A);
  traces.push({
    x: bx,
    y: by,
    type: "scatter",
    mode: "text+markers",
    marker: {
      color: "#ffd166",
      line: { color: "#1a1f2b", width: 1 },
      size: sec.booms.A.map((a) => 8 + 18 * Math.sqrt(Math.max(0, a) / amax)),
    },
    text: sec.booms.labels,
    textposition: "top center",
    textfont: { size: 9, color: "#9fb0c8" },
    hovertemplate: "%{text}<br>A=%{customdata:.1f} mm²<extra></extra>",
    customdata: sec.booms.A,
    name: "booms",
  });

  // Centroid (C) — cross marker.
  if (sec.centroid[0] != null && sec.centroid[1] != null) {
    traces.push({
      x: [sec.centroid[0]],
      y: [sec.centroid[1]],
      type: "scatter",
      mode: "markers",
      marker: { symbol: "cross", color: "#ffd166", size: 6, line: { color: "#ffd166", width: 2.5 } },
      name: "centroid C",
      hovertemplate: "C  x=%{x:.1f}  z=%{y:.1f} mm<extra>centroid</extra>",
    });
  }

  // Shear centre (SC) — x marker.
  if (sec.shear_centre[0] != null && sec.shear_centre[1] != null) {
    traces.push({
      x: [sec.shear_centre[0]],
      y: [sec.shear_centre[1]],
      type: "scatter",
      mode: "markers",
      marker: { symbol: "x", color: "#17b18a", size: 6, line: { color: "#17b18a", width: 2.5 } },
      name: "shear centre SC",
      hovertemplate: "SC  x=%{x:.1f}  z=%{y:.1f} mm<extra>shear centre</extra>",
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        ...baseLayout,
        showlegend: true,
        legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.08, yanchor: "top", font: { size: 10 } },
        margin: { ...baseLayout.margin, b: 80 },
        xaxis: { title: "x [mm]", gridcolor: "#1f2838" },
        yaxis: { title: "z [mm]", scaleanchor: "x", scaleratio: 1, gridcolor: "#1f2838" },
      }}
      config={config}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
