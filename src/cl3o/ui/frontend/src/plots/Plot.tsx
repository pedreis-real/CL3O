// Build a lean react-plotly.js component from the minified dist bundle
// (avoids pulling the full ~3 MB default plotly build).
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import type { Data } from "plotly.js";
import type { Mesh3D, Vec3 } from "../types";

const Plot = createPlotlyComponent(Plotly);
export default Plot;

// Shared dark layout defaults so every plot matches the app chrome.
export const baseLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#c9d4e3", family: "Inter, system-ui, sans-serif", size: 12 },
  margin: { l: 48, r: 16, t: 16, b: 40 },
  showlegend: false,
  // Vertical modebar pushed off the plot area so it does not overlap the
  // colorbar/legend area on stress and mesh views.
  modebar: { orientation: "v" as const, bgcolor: "rgba(0,0,0,0)" },
  transition: { duration: 0, easing: "linear" as const },
};

export const config = {
  displaylogo: false,
  responsive: true,
  // 2x raster so the snapshot button produces a publishable PNG.
  toImageButtonOptions: { format: "png" as const, scale: 2 },
};

// Shared 3D scene config (data aspect ratio, dark axes).
export const scene3d = {
  aspectmode: "data",
  xaxis: { title: "x (chord)", color: "#7c8aa5", gridcolor: "#1f2838", backgroundcolor: "rgba(0,0,0,0)" },
  yaxis: { title: "y (span)", color: "#7c8aa5", gridcolor: "#1f2838", backgroundcolor: "rgba(0,0,0,0)" },
  zaxis: { title: "z", color: "#7c8aa5", gridcolor: "#1f2838", backgroundcolor: "rgba(0,0,0,0)" },
} as const;

export interface MeshOpts {
  color?: string;
  opacity?: number;
  name?: string;
  intensity?: number[];
  intensitymode?: "vertex" | "cell";
  colorscale?: string | [number, string][];
  reversescale?: boolean;
  cmin?: number;
  cmax?: number;
  colorbarTitle?: string;
  colorbarY?: number;
  colorbarLen?: number;
  hovertemplate?: string;
  // When false, the colorscale is applied but no colorbar is drawn — used
  // by secondary surfaces that share the primary surface's scale.
  showscale?: boolean;
}

// Build a mesh3d trace from a Mesh3D payload; pass `intensity` for a colormap.
export function meshTrace(m: Mesh3D, opts: MeshOpts): Data {
  const t: Record<string, unknown> = {
    type: "mesh3d",
    x: m.vertices.map((p) => p[0]),
    y: m.vertices.map((p) => p[1]),
    z: m.vertices.map((p) => p[2]),
    i: m.i,
    j: m.j,
    k: m.k,
    opacity: opts.opacity ?? 1,
    flatshading: true,
    name: opts.name,
  };
  if (opts.intensity) {
    t.intensity = opts.intensity;
    t.intensitymode = opts.intensitymode ?? "vertex";
    t.colorscale = opts.colorscale ?? "Turbo";
    t.reversescale = opts.reversescale ?? false;
    t.showscale = opts.showscale ?? true;
    if (opts.cmin != null) t.cmin = opts.cmin;
    if (opts.cmax != null) t.cmax = opts.cmax;
    if (t.showscale) {
      t.colorbar = {
        title: { text: opts.colorbarTitle ?? "" },
        thickness: 12,
        ...(opts.colorbarY != null && { y: opts.colorbarY }),
        ...(opts.colorbarLen != null && { len: opts.colorbarLen }),
      };
    }
    if (opts.hovertemplate) t.hovertemplate = opts.hovertemplate;
  } else {
    t.color = opts.color ?? "#4f8cff";
    t.hoverinfo = "skip";
  }
  return t as Data;
}

export function lineTrace(pts: Vec3[], color: string, name: string): Data {
  return {
    type: "scatter3d",
    mode: "lines",
    x: pts.map((p) => p[0]),
    y: pts.map((p) => p[1]),
    z: pts.map((p) => p[2]),
    line: { color, width: 5 },
    name,
  } as Data;
}

// Draw the 4 boundary edges of a spar strip (Mesh3D with n_chord=2).
// Vertex layout (order="F" reshape): idx = s*2 + r, r=0 top, r=1 bottom.
// Edges: top span, tip web, bottom span (reversed), root web — joined with
// null gaps so Plotly renders them as one disconnected scatter3d trace.
export function sparEdgeTrace(m: Mesh3D, color: string, name: string): Data {
  const v = m.vertices;
  const ns = m.n_span ?? Math.floor(v.length / 2);
  const top = Array.from({ length: ns }, (_, s) => v[s * 2]);
  const bot = Array.from({ length: ns }, (_, s) => v[s * 2 + 1]);
  const segs: (Vec3 | null)[] = [...top, null, ...bot];
  return {
    type: "scatter3d",
    mode: "lines",
    x: segs.map((p) => (p ? p[0] : null)),
    y: segs.map((p) => (p ? p[1] : null)),
    z: segs.map((p) => (p ? p[2] : null)),
    line: { color, width: 4 },
    name,
    showlegend: false,
  } as Data;
}
