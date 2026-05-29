import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { Scene } from "../types";
import Plot, { baseLayout, config, lineTrace, meshTrace, scene3d, sparEdgeTrace } from "./Plot";
import { LAM_FAMILY_COLOR, materialColor, type LamFamily } from "./colors";

function familyOf(scene: Scene | null, idx: number | undefined): LamFamily {
  if (scene == null || idx == null || !scene.laminate_catalog) return "OTHER";
  const entry = scene.laminate_catalog[String(Math.round(idx))];
  const f = (entry?.family ?? "OTHER") as LamFamily;
  return (LAM_FAMILY_COLOR[f] ? f : "OTHER");
}

// 3-D baseline scene: translucent left-wing skin, the two spar surfaces
// (front @ xw1, rear @ xw2), and the centroid / shear-centre lines.
export function GeometryPlot() {
  const { runId, gen, publish } = useStore();
  const [scene, setScene] = useState<Scene | null>(null);
  const [err, setErr] = useState<string | null>(null);

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

  // Raw laminate name from catalog index (empty string when not found).
  function rawNameOf(idx: number | undefined): string {
    if (idx == null) return "";
    return catalog[String(Math.round(idx))]?.name ?? "";
  }

  // Per-material colour: deterministic hue from family + hash-driven variation.
  // Falls back to neutral grey when catalog is absent (legacy run).
  function matColor(idx: number | undefined): string {
    if (!hasFam) return "#62666e";
    const rn = rawNameOf(idx);
    return rn ? materialColor(rn, familyOf(scene, idx)) : LAM_FAMILY_COLOR["OTHER"];
  }

  // Build deduplicated legend: one entry per unique raw name, in encounter order
  // (skin → aft spar → rear spar → F1…F4). Prefix "MAT_" stripped for display.
  const legendItems: { name: string; color: string }[] | null = (() => {
    if (!hasFam || !lu) return null;
    const idxList = [
      lu.ls1?.[0], lu.lw1?.[0], lu.lw2?.[0],
      lu.lf1?.[0], lu.lf2?.[0], lu.lf3?.[0], lu.lf4?.[0],
    ];
    const seen = new Set<string>();
    const items: { name: string; color: string }[] = [];
    for (const idx of idxList) {
      const rn = rawNameOf(idx);
      if (!rn || seen.has(rn)) continue;
      seen.add(rn);
      items.push({
        name: rn.replace(/^MAT_/, ""),
        color: materialColor(rn, familyOf(scene, idx)),
      });
    }
    return items.length ? items : null;
  })();

  const traces: Data[] = [
    meshTrace(scene.surface, {
      color: matColor(lu?.ls1?.[0]),
      opacity: 0.30,
      name: "skin",
    }),
    meshTrace(scene.front_spar, {
      color: matColor(lu?.lw1?.[0]),
      opacity: hasFam ? 0.65 : 0.85,
      name: "aft spar",
    }),
    meshTrace(scene.rear_spar, {
      color: matColor(lu?.lw2?.[0]),
      opacity: hasFam ? 0.65 : 0.85,
      name: "rear spar",
    }),
    sparEdgeTrace(scene.front_spar, "#000000", "front spar edge"),
    sparEdgeTrace(scene.rear_spar, "#000000", "rear spar edge"),
    ...(scene.flanges ?? []).map((fl) =>
      meshTrace(fl, { color: matColor(fl.layup_idx), opacity: 0.85, name: fl.label }),
    ),
    { ...lineTrace(scene.centroid_line, "#ffd166", "centroid line"), visible: "legendonly" } as Data,
    { ...lineTrace(scene.shear_line, "#17b18a", "shear-centre line"), visible: "legendonly" } as Data,
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
      {legendItems && (
        <div className="geo-layup-legend">
          <div className="geo-layup-legend__title">Layups</div>
          {legendItems.map(({ name, color }) => (
            <div key={name} className="geo-layup-legend__row">
              <span className="geo-layup-legend__swatch" style={{ background: color }} />
              <span className="geo-layup-legend__name">{name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
