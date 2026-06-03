import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { LaminateEntry, Scene } from "../types";
import Plot, { baseLayout, config, meshTrace, scene3d, sparEdgeTrace } from "./Plot";
import { useSnapshotButton } from "../hooks/useSnapshotButton";
import { LAM_FAMILY_COLOR, materialColor, type LamFamily } from "./colors";

interface XRow { i: number; variable: string; cp: string; value: number }

function buildXRows(optvars: number[]): XRow[] {
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

function familyOf(scene: Scene | null, idx: number | undefined): LamFamily {
  if (scene == null || idx == null || !scene.laminate_catalog) return "OTHER";
  const entry = scene.laminate_catalog[String(Math.round(idx))];
  const f = (entry?.family ?? "OTHER") as LamFamily;
  return (LAM_FAMILY_COLOR[f] ? f : "OTHER");
}

function fmtModulus(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v / 1000).toFixed(1)} GPa`;
}

// Table of laminates used in the run — shown in the top-left overlay.
function LayupTable({
  catalog,
  usedIndices,
  onClose,
}: {
  catalog: Record<string, LaminateEntry>;
  usedIndices: number[];
  onClose: () => void;
}) {
  const [expandedPlyRow, setExpandedPlyRow] = useState<number | null>(null);

  return (
    <div className="layup-table-panel">
      <div className="layup-table-header">
        <span>Layup catalog · click Plies cell to expand</span>
        <button className="layup-table-close" onClick={onClose}>✕</button>
      </div>
      <div className="layup-table-scroll">
        <table className="layup-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Name</th>
              <th>Em1</th>
              <th>Em2</th>
              <th>Gm12</th>
              <th>Eb1</th>
              <th>Eb2</th>
              <th>Gb12</th>
              <th>t [mm]</th>
              <th>Stack</th>
              <th>Plies</th>
            </tr>
          </thead>
          <tbody>
            {usedIndices.map((idx) => {
              const entry = catalog[String(idx)];
              const pliesExpanded = expandedPlyRow === idx;
              if (!entry) return (
                <tr key={idx}>
                  <td className="idx">{idx}</td>
                  <td className="name" colSpan={9} style={{ color: "#888", fontStyle: "italic" }}>
                    material #{idx} — catalog mismatch (run used a different material set)
                  </td>
                </tr>
              );
              return (
                <tr key={idx}>
                  <td className="idx">{idx}</td>
                  <td className="name">{entry.name.replace(/^MAT_/, "")}</td>
                  <td>{fmtModulus(entry.E1)}</td>
                  <td>{fmtModulus(entry.E2)}</td>
                  <td>{fmtModulus(entry.G12)}</td>
                  <td>{fmtModulus(entry.E1_bend)}</td>
                  <td>{fmtModulus(entry.E2_bend)}</td>
                  <td>{fmtModulus(entry.G12_bend)}</td>
                  <td>{entry.thick != null ? entry.thick.toFixed(1) : "—"}</td>
                  <td className="stack">{entry.stacking_seq ?? "—"}</td>
                  <td
                    className={`plies${pliesExpanded ? " plies-expanded" : ""}`}
                    onClick={() => setExpandedPlyRow(pliesExpanded ? null : idx)}
                    title={pliesExpanded ? "Click to collapse" : "Click to expand"}
                  >
                    {(entry.plies ?? []).join(", ")}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function XVectorTable({
  optvars,
  onClose,
}: {
  optvars: number[];
  onClose: () => void;
}) {
  const rows = buildXRows(optvars);
  return (
    <div className="layup-table-panel" style={{ left: 200 }}>
      <div className="layup-table-header">
        <span>Design vector X  (D = {optvars.length})</span>
        <button className="layup-table-close" onClick={onClose}>✕</button>
      </div>
      <div className="layup-table-scroll">
        <table className="layup-table">
          <thead>
            <tr><th>i</th><th>Variable</th><th>CP</th><th>Value</th></tr>
          </thead>
          <tbody>
            {rows.map(({ i, variable, cp, value }) => (
              <tr key={i}>
                <td className="idx">{i}</td>
                <td className="name">{variable}</td>
                <td>{cp}</td>
                <td>{value != null ? value.toFixed(4) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// 3-D baseline scene: translucent left-wing skin and spar surfaces.
// Centroid / shear-centre lines have been removed (use Cross-section view instead).
export function GeometryPlot() {
  const { runId, gen, publish, info } = useStore();
  const [scene, setScene] = useState<Scene | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);
  const [showXTable, setShowXTable] = useState(false);

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    Promise.all([api.geometry(runId, gen), api.planform(runId)])
      .then(([sc, pf]) => {
        if (!alive) return;
        setScene(sc);
        publish({ planform: pf });
      })
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [runId, gen, publish]);

  if (err) return <div className="plot-error">{err}</div>;
  if (!scene) return <div className="plot-loading">Loading wing geometry…</div>;

  const lu = scene.layups;
  const hasFam = scene.laminate_catalog != null;
  const catalog = scene.laminate_catalog ?? {};

  function rawNameOf(idx: number | undefined): string {
    if (idx == null) return "";
    return catalog[String(Math.round(idx))]?.name ?? "";
  }

  function matColor(idx: number | undefined): string {
    if (!hasFam) return "#62666e";
    const rn = rawNameOf(idx);
    return rn ? materialColor(rn, familyOf(scene, idx)) : LAM_FAMILY_COLOR["OTHER"];
  }

  // All unique laminate indices across every span-station and panel group.
  const usedIndices: number[] = (() => {
    if (!hasFam || !lu) return [];
    const allArrays = [lu.ls1, lu.ls2, lu.lw1, lu.lw2, lu.lf1, lu.lf2, lu.lf3, lu.lf4];
    const seen = new Set<number>();
    for (const arr of allArrays) {
      if (!arr) continue;
      for (const v of arr) {
        if (v != null) seen.add(Math.round(v));
      }
    }
    return [...seen].sort((a, b) => a - b);
  })();

  // Legend items: one per unique laminate, ordered by first encounter.
  const legendItems: { name: string; color: string; idx: number; unknown?: boolean }[] | null = (() => {
    if (!hasFam || !lu) return null;
    const rootIdxList = [
      lu.ls1?.[0], lu.ls2?.[0], lu.lw1?.[0], lu.lw2?.[0],
      lu.lf1?.[0], lu.lf2?.[0], lu.lf3?.[0], lu.lf4?.[0],
    ];
    const seen = new Set<string>();
    const items: { name: string; color: string; idx: number; unknown?: boolean }[] = [];
    for (const idx of rootIdxList) {
      if (idx == null) continue;
      const rn = rawNameOf(idx);
      const key = rn || `#${Math.round(idx as number)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      items.push({
        name: rn ? rn.replace(/^MAT_/, "") : `unknown #${Math.round(idx as number)}`,
        color: rn ? materialColor(rn, familyOf(scene, idx)) : LAM_FAMILY_COLOR["OTHER"],
        idx: Math.round(idx as number),
        unknown: !rn,
      });
    }
    return items.length ? items : null;
  })();

  const snapBtn = useSnapshotButton(runId, gen, "geometry");
  const snapConfig = { ...config, modeBarButtonsToAdd: [snapBtn] as any[] };

  const traces: Data[] = [
    // Nose skin (ls1) — LE to front spar.
    meshTrace(scene.surface_ls1, {
      color: matColor(lu?.ls1?.[0]),
      opacity: 0.65,
      name: `skin LE→spar  ls1=${lu?.ls1?.[0] ?? "?"}`,
    }),
    // Box skin (ls2) — front spar to TE.
    meshTrace(scene.surface_ls2, {
      color: matColor(lu?.ls2?.[0]),
      opacity: 0.65,
      name: `skin spar→TE  ls2=${lu?.ls2?.[0] ?? "?"}`,
    }),
    meshTrace(scene.front_spar, {
      color: matColor(lu?.lw1?.[0]),
      opacity: 0.65,
      name: `front spar  lw1=${lu?.lw1?.[0] ?? "?"}`,
    }),
    meshTrace(scene.rear_spar, {
      color: matColor(lu?.lw2?.[0]),
      opacity: 0.65,
      name: `rear spar  lw2=${lu?.lw2?.[0] ?? "?"}`,
    }),
    sparEdgeTrace(scene.front_spar, "#000000", "front spar edge"),
    sparEdgeTrace(scene.rear_spar, "#000000", "rear spar edge"),
    ...(scene.flanges ?? []).map((fl) =>
      meshTrace(fl, {
        color: matColor(fl.layup_idx),
        opacity: 0.65,
        name: `${fl.label}  lam=${fl.layup_idx}`,
      }),
    ),
  ];

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <Plot
        data={traces}
        layout={{
          ...baseLayout,
          showlegend: true,
          legend: { x: 0, y: 1, font: { size: 11 } },
          scene: scene3d,
          uirevision: `geo:${runId}`,
        }}
        config={snapConfig}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />

      {/* Top-left: layup table toggle button */}
      <button
        className="layup-table-btn"
        onClick={() => setShowTable((v) => !v)}
        title="Show layup table"
      >
        Layups
      </button>

      {/* Layup table panel (top-left overlay) */}
      {showTable && hasFam && (
        <LayupTable
          catalog={catalog}
          usedIndices={usedIndices}
          onClose={() => setShowTable(false)}
        />
      )}

      {/* X[] table toggle button */}
      <button
        className="layup-table-btn"
        style={{ left: 80 }}
        onClick={() => setShowXTable((v) => !v)}
        title="Show design vector table"
      >
        X[ ]
      </button>

      {/* Design vector overlay */}
      {showXTable && info?.optvars && (
        <XVectorTable
          optvars={info.optvars}
          onClose={() => setShowXTable(false)}
        />
      )}

      {/* Bottom-left: material color legend */}
      {legendItems && (
        <div className="geo-layup-legend">
          <div className="geo-layup-legend__title">Layups</div>
          {legendItems.map(({ name, color, idx, unknown }) => (
            <div key={idx} className="geo-layup-legend__row">
              <span className="geo-layup-legend__swatch" style={{ background: color }} />
              <span className="geo-layup-legend__idx">#{idx}</span>
              <span
                className="geo-layup-legend__name"
                style={unknown ? { color: "#888", fontStyle: "italic" } : undefined}
              >
                {name}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
