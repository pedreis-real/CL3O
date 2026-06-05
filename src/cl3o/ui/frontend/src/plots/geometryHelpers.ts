// Pure helpers for the Geometry view: design-vector decoding, laminate
// family/colour resolution, and the legend / used-index derivations. Kept
// JSX-free so they can be unit-reasoned and reused by the table overlays.
import type { Scene } from "../types";
import { LAM_FAMILY_COLOR, materialColor, type LamFamily } from "./colors";

export type Layups = NonNullable<Scene["layups"]>;

export interface XRow {
  i: number;
  variable: string;
  cp: string;
  value: number;
}

// Decode the flat design vector X into labelled rows. Layout is
// 11 control-point blocks + 3 scalar root widths (D = 11*n + 3).
export function buildXRows(optvars: number[]): XRow[] {
  const D = optvars.length;
  const n = (D - 3) / 11;
  if (!Number.isInteger(n) || n < 1) return [];
  const blocks: { name: string; len: number; hasCp: boolean }[] = [
    { name: "xw1",      len: n,     hasCp: true  },
    { name: "xw2",      len: n,     hasCp: true  },
    { name: "bf1_root", len: 1,     hasCp: false },
    { name: "bf2_root", len: 1,     hasCp: false },
    { name: "bf3_root", len: 1,     hasCp: false },
    { name: "bf4_root", len: 1,     hasCp: false },
    { name: "tpr",      len: n - 1, hasCp: true  },
    { name: "ls1",      len: n,     hasCp: true  },
    { name: "ls2",      len: n,     hasCp: true  },
    { name: "lw1",      len: n,     hasCp: true  },
    { name: "lw2",      len: n,     hasCp: true  },
    { name: "lf1",      len: n,     hasCp: true  },
    { name: "lf2",      len: n,     hasCp: true  },
    { name: "lf3",      len: n,     hasCp: true  },
    { name: "lf4",      len: n,     hasCp: true  },
  ];
  const rows: XRow[] = [];
  let i = 0;
  for (const blk of blocks) {
    for (let k = 0; k < blk.len; k++) {
      rows.push({
        i,
        variable: blk.name,
        cp:       blk.hasCp ? String(k) : "—",
        value:    optvars[i],
      });
      i++;
    }
  }
  return rows;
}

export function familyOf(scene: Scene | null, idx: number | undefined): LamFamily {
  if (scene == null || idx == null || !scene.laminate_catalog) return "OTHER";
  const entry = scene.laminate_catalog[String(Math.round(idx))];
  const f = (entry?.family ?? "OTHER") as LamFamily;
  return (LAM_FAMILY_COLOR[f] ? f : "OTHER");
}

export function fmtModulus(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v / 1000).toFixed(1)} GPa`;
}

// Raw laminate name (e.g. "MAT_CFRP_...") for a layup index, or "".
export function rawNameOf(scene: Scene | null, idx: number | undefined): string {
  if (idx == null || !scene?.laminate_catalog) return "";
  return scene.laminate_catalog[String(Math.round(idx))]?.name ?? "";
}

// Material colour for a layup index: grey when the catalog is absent,
// family-coloured otherwise, falling back to OTHER for unknown indices.
export function materialColorFor(scene: Scene | null, idx: number | undefined): string {
  if (scene?.laminate_catalog == null) return "#62666e";
  const rn = rawNameOf(scene, idx);
  return rn ? materialColor(rn, familyOf(scene, idx)) : LAM_FAMILY_COLOR["OTHER"];
}

// Every unique laminate index across all span-stations and panel groups.
export function collectUsedIndices(scene: Scene | null, lu: Layups | undefined): number[] {
  if (scene?.laminate_catalog == null || !lu) return [];
  const allArrays = [lu.ls1, lu.ls2, lu.lw1, lu.lw2, lu.lf1, lu.lf2, lu.lf3, lu.lf4];
  const seen = new Set<number>();
  for (const arr of allArrays) {
    if (!arr) continue;
    for (const v of arr) {
      if (v != null) seen.add(Math.round(v));
    }
  }
  return [...seen].sort((a, b) => a - b);
}

export interface LegendItem {
  name: string;
  color: string;
  idx: number;
  unknown?: boolean;
}

// One legend item per unique root laminate, ordered by first encounter.
export function buildLegendItems(scene: Scene | null, lu: Layups | undefined): LegendItem[] | null {
  if (scene?.laminate_catalog == null || !lu) return null;
  const rootIdxList = [
    lu.ls1?.[0], lu.ls2?.[0], lu.lw1?.[0], lu.lw2?.[0],
    lu.lf1?.[0], lu.lf2?.[0], lu.lf3?.[0], lu.lf4?.[0],
  ];
  const seen = new Set<string>();
  const items: LegendItem[] = [];
  for (const idx of rootIdxList) {
    if (idx == null) continue;
    const rn = rawNameOf(scene, idx);
    const key = rn || `#${Math.round(idx)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    items.push({
      name: rn ? rn.replace(/^MAT_/, "") : `unknown #${Math.round(idx)}`,
      color: rn ? materialColor(rn, familyOf(scene, idx)) : LAM_FAMILY_COLOR["OTHER"],
      idx: Math.round(idx),
      unknown: !rn,
    });
  }
  return items.length ? items : null;
}
