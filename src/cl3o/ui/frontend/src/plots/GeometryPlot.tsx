import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { LaminateEntry, Scene } from "../types";
import Plot, { baseLayout, config, meshTrace, scene3d, sparEdgeTrace } from "./Plot";
import { LAM_FAMILY_COLOR, materialColor, type LamFamily } from "./colors";

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

// 3-D baseline scene: translucent left-wing skin and spar surfaces.
// Centroid / shear-centre lines have been removed (use Cross-section view instead).
export function GeometryPlot() {
  const { runId, gen, publish } = useStore();
  const [scene, setScene] = useState<Scene | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);

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
      lu.ls1?.[0], lu.lw1?.[0], lu.lw2?.[0],
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

  const traces: Data[] = [
    meshTrace(scene.surface, {
      color: matColor(lu?.ls1?.[0]),
      opacity: 0.65,
      name: "skin",
    }),
    meshTrace(scene.front_spar, {
      color: matColor(lu?.lw1?.[0]),
      opacity: 0.65,
      name: "aft spar",
    }),
    meshTrace(scene.rear_spar, {
      color: matColor(lu?.lw2?.[0]),
      opacity: 0.65,
      name: "rear spar",
    }),
    sparEdgeTrace(scene.front_spar, "#000000", "front spar edge"),
    sparEdgeTrace(scene.rear_spar, "#000000", "rear spar edge"),
    ...(scene.flanges ?? []).map((fl) =>
      meshTrace(fl, { color: matColor(fl.layup_idx), opacity: 0.65, name: fl.label }),
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
        config={config}
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
