import { useEffect, useRef, useState } from "react";
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

// Builds Plotly annotation arrows + dashed tail traces for one axis direction.
// `dx, dz` are unit-vector components in the section (x, z) plane.
function axisArrow(
  cx: number, cz: number, dx: number, dz: number,
  L: number, label: string, color: string,
): { trace: Data; annot: object } {
  const tx = cx + dx * L;
  const tz = cz + dz * L;
  return {
    // Dashed negative half
    trace: {
      x: [cx - dx * L * 0.6, cx],
      y: [cz - dz * L * 0.6, cz],
      type: "scatter",
      mode: "lines",
      line: { color, width: 1.2, dash: "dot" },
      hoverinfo: "skip",
      showlegend: false,
    } as Data,
    // Arrow from centroid → tip with label
    annot: {
      x: tx, y: tz,
      ax: cx, ay: cz,
      xref: "x", yref: "y",
      axref: "x", ayref: "y",
      text: `<b>${label}</b>`,
      font: { size: 11, color, family: "Inter, system-ui, sans-serif" },
      arrowhead: 2,
      arrowwidth: 1.8,
      arrowcolor: color,
      showarrow: true,
      standoff: 4,
    },
  };
}

export function SectionPlot() {
  const { runId, gen, station, publish, showSectionAxes } = useStore();
  const [sec, setSec] = useState<Section | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // Fixed axis range set on first section load per run/gen — keeps axes static
  // as the user slides through stations (tip section is smaller than root).
  const [fixedRange, setFixedRange] = useState<{ x: [number, number]; z: [number, number] } | null>(null);
  const prevRunGenRef = useRef<string | null>(null);

  // Reset fixed range when run or generation changes.
  useEffect(() => {
    const key = `${runId}-${gen}`;
    if (prevRunGenRef.current !== key) {
      prevRunGenRef.current = key;
      setFixedRange(null);
    }
  }, [runId, gen]);

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

  // Expand fixed axis range to always fit the loaded section, adding padding.
  useEffect(() => {
    if (!sec) return;
    const allX = sec.panels.flatMap((p) => p.pts.map((q) => q[0]));
    const allZ = sec.panels.flatMap((p) => p.pts.map((q) => q[1]));
    if (!allX.length) return;
    const xMin = Math.min(...allX), xMax = Math.max(...allX);
    const zMin = Math.min(...allZ), zMax = Math.max(...allZ);
    const span = Math.max(xMax - xMin, zMax - zMin);
    const pad = span * 0.10;
    setFixedRange((prev) => ({
      x: [
        Math.min(prev?.x[0] ?? xMin - pad, xMin - pad),
        Math.max(prev?.x[1] ?? xMax + pad, xMax + pad),
      ],
      z: [
        Math.min(prev?.z[0] ?? zMin - pad, zMin - pad),
        Math.max(prev?.z[1] ?? zMax + pad, zMax + pad),
      ],
    }));
  }, [sec]);

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

  // Centroidal axes (u, w) and principal/local axes (y, z) drawn as arrows.
  const axisAnnotations: object[] = [];
  if (
    showSectionAxes &&
    sec.centroid[0] != null && sec.centroid[1] != null &&
    sec.props.c_rad != null
  ) {
    const cx = sec.centroid[0];
    const cz = sec.centroid[1];
    const L  = (sec.chord ?? 200) * 0.24;
    const c_rad = sec.props.c_rad;

    // u-w centroidal axes: aligned with section x and z.
    // y-z local (principal) axes: rotated by c_rad from u-w.
    //   axis-1 (major, local z): direction (cos θ_P, sin θ_P)
    //   axis-2 (minor, local y): direction (−sin θ_P, cos θ_P)
    // u-w: centroidal axes (steel blue — neutral, not in PANEL_COLORS)
    // y-z: principal axes (warm amber — distinct from panels and booms)
    const axisDefs = [
      { dx: 1,                  dz: 0,                  label: "u", color: "#c4dcef" },
      { dx: 0,                  dz: 1,                  label: "w", color: "#c4dcef" },
      { dx:  Math.cos(c_rad),   dz: Math.sin(c_rad),    label: "z", color: "#ffebd4" },
      { dx: -Math.sin(c_rad),   dz: Math.cos(c_rad),    label: "y", color: "#ffebd4" },
    ] as const;

    for (const { dx, dz, label, color } of axisDefs) {
      const { trace, annot } = axisArrow(cx, cz, dx, dz, L, label, color);
      traces.push(trace);
      axisAnnotations.push(annot);
    }
  }

  return (
    <Plot
      data={traces}
      layout={{
        ...baseLayout,
        showlegend: true,
        legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.08, yanchor: "top", font: { size: 10 } },
        margin: { ...baseLayout.margin, b: 80 },
        annotations: axisAnnotations,
        xaxis: {
          title: "x [mm]",
          gridcolor: "#1f2838",
          ...(fixedRange ? { range: fixedRange.x, autorange: false } : {}),
        },
        yaxis: {
          title: "z [mm]",
          scaleanchor: "x",
          scaleratio: 1,
          gridcolor: "#1f2838",
          ...(fixedRange ? { range: fixedRange.z, autorange: false } : {}),
        },
        uirevision: `sec:${runId}:${gen}`,
      }}
      config={config}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
