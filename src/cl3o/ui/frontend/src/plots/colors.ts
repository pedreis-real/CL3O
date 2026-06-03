// Shared color palettes for CL3O plots.

// Viridis-style sequential palette for the Misc search-space view: low
// fitness reads as dark purple, high fitness as bright yellow. Easier on
// the eyes than the Femap spectrum for a gradient-descent trace.
export const SEARCH_CMAP: [number, string][] = [
  [0.00, "#440154"],
  [0.20, "#3b528b"],
  [0.40, "#21918c"],
  [0.60, "#5ec962"],
  [0.80, "#bfdf3a"],
  [1.00, "#fde725"],
];

// Femap-style spectrum (blue → cyan → green → yellow → red) for mesh
// contour plots. Stops are chosen to match the legacy Femap default so the
// thesis figures track Femap output.
export const FEMAP_CMAP: [number, string][] = [
  [0.000, "#0000c8"],
  [0.143, "#0064ff"],
  [0.286, "#00b4ff"],
  [0.429, "#00e6c8"],
  [0.571, "#7dff64"],
  [0.714, "#ffe600"],
  [0.857, "#ff7a00"],
  [1.000, "#c80000"],
];

// Laminate-family pastel palette (transparent fills for skin / web / flange
// surfaces in the Geometry view). One family = one hue; alpha kept low so
// overlapping panels remain readable.
export type LamFamily = "CFRP" | "GFRP" | "CFRP_sand" | "GFRP_sand" | "OTHER";

export const LAM_FAMILY_COLOR: Record<LamFamily, string> = {
  CFRP:      "#7aa9ff", // pastel blue
  GFRP:      "#8ed8a6", // pastel green
  CFRP_sand: "#ffb27a", // pastel orange
  GFRP_sand: "#ffe680", // pastel yellow
  OTHER:     "#c9d4e3", // neutral
};

export const LAM_FAMILY_LABEL: Record<LamFamily, string> = {
  CFRP:      "CFRP",
  GFRP:      "GFRP",
  CFRP_sand: "CFRP sandwich",
  GFRP_sand: "GFRP sandwich",
  OTHER:     "Other",
};

// Base hue (degrees) for each family — similar families share the same part of
// the colour wheel so the viewer can tell CFRP from GFRP at a glance.
const LAM_FAMILY_HUE: Record<LamFamily, number> = {
  CFRP:      215, // blue
  GFRP:      145, // green
  CFRP_sand:  25, // orange
  GFRP_sand:  55, // yellow
  OTHER:     195, // muted cyan-grey
};

// FNV-1a 32-bit hash — stable, no external deps.
function fnv1a(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

// Per-material HSL colour: same family → same hue neighbourhood; different
// material names within a family → varied sat / lightness / small hue jitter
// so each laminate has its own distinct shade.
export function materialColor(name: string, family: LamFamily): string {
  const baseHue = LAM_FAMILY_HUE[family];
  const h = fnv1a(name);
  const hJitter = ((h & 0x3f) - 32);        // −32 … +31 °
  const sat     = 55 + ((h >> 8)  & 0x1f);  // 55 – 86 %
  const lit     = 48 + ((h >> 16) & 0x1f);  // 48 – 79 %
  return `hsl(${(baseHue + hJitter + 360) % 360}, ${sat}%, ${lit}%)`;
}
