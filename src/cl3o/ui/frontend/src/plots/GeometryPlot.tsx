import { useEffect, useState } from "react";
import type { Data } from "plotly.js";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { Scene } from "../types";
import Plot, { baseLayout, config, lineTrace, meshTrace, scene3d, sparEdgeTrace } from "./Plot";

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

  const traces: Data[] = [
    meshTrace(scene.surface, { color: "#62666e", opacity: 0.30, name: "skin" }),
    meshTrace(scene.front_spar, { color: "#092441", opacity: 0.85, name: "front spar (xw1)" }),
    meshTrace(scene.rear_spar, { color: "#4b1919", opacity: 0.85, name: "rear spar (xw2)" }),
    sparEdgeTrace(scene.front_spar, "#000000", "front spar edge"),
    sparEdgeTrace(scene.rear_spar, "#000000", "rear spar edge"),
    { ...lineTrace(scene.centroid_line, "#ffd166", "centroid line"), visible: "legendonly" } as Data,
    { ...lineTrace(scene.shear_line, "#17b18a", "shear-centre line"), visible: "legendonly" } as Data,
  ];

  return (
    <Plot
      data={traces}
      layout={{
        ...baseLayout,
        showlegend: true,
        legend: { x: 0, y: 1, font: { size: 11 } },
        scene: scene3d,
      }}
      config={config}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
