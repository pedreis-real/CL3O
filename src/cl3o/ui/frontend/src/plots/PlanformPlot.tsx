import { useEffect, useState } from "react";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { Planform } from "../types";
import Plot, { baseLayout, config } from "./Plot";

export function PlanformPlot() {
  const { runId, publish } = useStore();
  const [pf, setPf] = useState<Planform | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    api
      .planform(runId)
      .then((d) => {
        if (!alive) return;
        setPf(d);
        publish({ planform: d });
      })
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [runId, publish]);

  if (err) return <div className="plot-error">{err}</div>;
  if (!pf) return <div className="plot-loading">Loading planform…</div>;

  // Plan view: span on x, chordwise on y (reversed so LE sits on top).
  const spanX = [...pf.le.map((p) => p[1]), ...pf.te.map((p) => p[1]).reverse()];
  const chordY = [...pf.le.map((p) => p[0]), ...pf.te.map((p) => p[0]).reverse()];

  return (
    <Plot
      data={[
        {
          x: spanX,
          y: chordY,
          type: "scatter",
          mode: "lines",
          fill: "toself",
          fillcolor: "rgba(79,140,255,0.15)",
          line: { color: "#4f8cff", width: 2 },
          hovertemplate: "span %{x:.0f} mm<br>chord-x %{y:.0f} mm<extra></extra>",
        },
      ]}
      layout={{
        ...baseLayout,
        xaxis: { title: "span  y [mm]", zeroline: true, zerolinecolor: "#33405a", gridcolor: "#1f2838" },
        yaxis: {
          title: "chord  x [mm]",
          autorange: "reversed",
          scaleanchor: "x",
          scaleratio: 1,
          gridcolor: "#1f2838",
        },
      }}
      config={config}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
