// Colormaps, flux-field options and the shared boom-rod trace builder for
// the Stress / Flux / Failure(R) views. Kept separate from the component so
// the per-mode render blocks stay focused on layout.
import type { Data } from "plotly.js";
import type { BoomRod } from "../types";

// FEMAP-style rainbow colormap for shear stress tau.
export const TAU_CMAP: [number, string][] = [
  [0.000, "#0000c8"], [0.200, "#0096ff"], [0.375, "#00e6ff"],
  [0.500, "#00c800"], [0.625, "#ffff00"], [0.800, "#ff6400"], [1.000, "#c80000"],
];

// Blue–white–red diverging colormap for normal stress sigma.
export const SIG_CMAP: [number, string][] = [
  [0.000, "#1a6faf"], [0.250, "#6db0d8"], [0.500, "#f5f5f5"],
  [0.750, "#e8795a"], [1.000, "#b2182b"],
];

// Purple–white–green diverging colormap for shear flux.
export const FLUX_CMAP: [number, string][] = [
  [0.000, "#7b2d8b"], [0.250, "#c47fc4"], [0.500, "#f5f5f5"],
  [0.750, "#6dbf8a"], [1.000, "#1a7a3c"],
];

// Green (safe) -> yellow (near failure) -> red (failed) for Tsai-Wu R.
export const TSW_CMAP: [number, string][] = [
  [0.000, "#1a7a3c"], [0.350, "#6dbf8a"], [0.500, "#ffd166"],
  [0.650, "#e8795a"], [1.000, "#b2182b"],
];

export type FluxKey = "flux_qsX" | "flux_qsZ" | "flux_qT" | "flux_qbX" | "flux_qbZ";

export const FLUX_OPTIONS: { key: FluxKey; label: string }[] = [
  { key: "flux_qsX", label: "q·S_X (total)" },
  { key: "flux_qsZ", label: "q·S_Z (total)" },
  { key: "flux_qT",  label: "q·T  (total)"  },
  { key: "flux_qbX", label: "qb·S_X (open)" },
  { key: "flux_qbZ", label: "qb·S_Z (open)" },
];

interface BoomRodOpts {
  colorscale: [number, string][];
  cmin: number;
  cmax: number;
  colorbarTitle: string;
  // Hover value expression after the label, e.g. "σ = %{line.color:.3g} MPa".
  hoverValue: string;
}

// One coloured scatter3d line per structural boom rod (shared by the stress
// and failure views, which differ only in colormap / range / hover format).
export function boomRodTraces(rods: BoomRod[], opts: BoomRodOpts): Data[] {
  return rods.map((rod) => ({
    type: "scatter3d",
    mode: "lines",
    x: rod.xyz.map((p) => p[0]),
    y: rod.xyz.map((p) => p[1]),
    z: rod.xyz.map((p) => p[2]),
    line: {
      color: rod.sigma as number[],
      colorscale: opts.colorscale,
      reversescale: false,
      cmin: opts.cmin,
      cmax: opts.cmax,
      width: 20,
      colorbar: {
        title: { text: opts.colorbarTitle },
        thickness: 12,
        x: 1.0,
        len: 0.45,
        y: 0.22,
      } as never,
    },
    name: rod.label,
    hovertemplate: `${rod.label}  ${opts.hoverValue}<extra>boom</extra>`,
    showlegend: false,
  } as Data));
}
