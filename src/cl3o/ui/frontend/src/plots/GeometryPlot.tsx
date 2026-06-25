import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { Scene } from "../types";
import Plot, { baseLayout, meshTrace, scene3d, sparEdgeTrace } from "./Plot";
import { useSnapshotConfig } from "../hooks/useSnapshotButton";
import {
  buildLegendItems, collectUsedIndices, materialColorFor,
} from "./geometryHelpers";
import { LayupTable, XVectorTable } from "./GeometryTables";

// 3-D baseline scene: translucent wing skin and spar surfaces.
const OPACITY = 0.80
export function GeometryPlot() {
  const { runId, gen, publish, info } = useStore();
  const [scene, setScene] = useState<Scene | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);
  const [showXTable, setShowXTable] = useState(false);
  const snapConfig = useSnapshotConfig("geometry");

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

  const matColor = (idx: number | undefined) => materialColorFor(scene, idx);
  const usedIndices = collectUsedIndices(scene, lu);
  const legendItems = buildLegendItems(scene, lu);

  const traces: Data[] = [
    // Nose skin (ls1) — LE to front spar.
    meshTrace(scene.surface_ls1, {
      color: matColor(lu?.ls1?.[0]),
      opacity: OPACITY,
      name: `skin LE→spar  ls1=${lu?.ls1?.[0] ?? "?"}`,
    }),
    // Box skin (ls2) — front spar to TE.
    meshTrace(scene.surface_ls2, {
      color: matColor(lu?.ls2?.[0]),
      opacity: OPACITY,
      name: `skin spar→TE  ls2=${lu?.ls2?.[0] ?? "?"}`,
    }),
    meshTrace(scene.front_spar, {
      color: matColor(lu?.lw1?.[0]),
      opacity: OPACITY,
      name: `front spar  lw1=${lu?.lw1?.[0] ?? "?"}`,
    }),
    meshTrace(scene.rear_spar, {
      color: matColor(lu?.lw2?.[0]),
      opacity: OPACITY,
      name: `rear spar  lw2=${lu?.lw2?.[0] ?? "?"}`,
    }),
    sparEdgeTrace(scene.front_spar, "#000000", "front spar edge"),
    sparEdgeTrace(scene.rear_spar, "#000000", "rear spar edge"),
    ...(scene.flanges ?? []).map((fl) =>
      meshTrace(fl, {
        color: matColor(fl.layup_idx),
        opacity: OPACITY,
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
