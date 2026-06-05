import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AnovaGroup, SensitivityData } from "../types";
import Plot, { baseLayout } from "./Plot";
import { useSnapshotConfig } from "../hooks/useSnapshotButton";

// ANOVA eta^2 ranking + per-group fitness box-plot (from pre-computed CSVs).
export function SensitivityView() {
  const snapConfig = useSnapshotConfig("sensitivity");
  const [data, setData] = useState<SensitivityData | null>(null);
  const [err,  setErr]  = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api.sensitivity()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, []);

  if (err)  return <div className="plot-error">{err}</div>;
  if (!data) return <div className="plot-loading">Loading sensitivity data…</div>;
  if (!data.available || !data.groups?.length) {
    return (
      <div className="plot-empty">
        ANOVA data not found. Run <code>tools/sensitivity_analysis.py</code> first
        to generate <code>tools/output/sensitivity/anova_results.csv</code>.
      </div>
    );
  }

  const sorted = [...data.groups].sort((a, b) => a.eta_sq - b.eta_sq);
  const groups = sorted.map((g) => g.group);
  const etaSq  = sorted.map((g) => g.eta_sq);

  const subtitle = data.summary
    ? `F = ${data.summary.F_stat.toFixed(3)}   p = ${data.summary.p_value.toExponential(3)}`
    : "";

  // Box statistics approximated from mean/std (Gaussian assumption for Q1/Q3).
  const boxTraces = sorted.map((g: AnovaGroup, _i: number) => {
    const q1 = g.mean_f - 0.6745 * g.std_f;
    const q3 = g.mean_f + 0.6745 * g.std_f;
    const iqr = q3 - q1;
    return {
      type:  "box" as const,
      name:  g.group,
      q1:    [q1],
      median:[g.mean_f],
      q3:    [q3],
      lowerfence: [Math.max(g.min_f, q1 - 1.5 * iqr)],
      upperfence: [Math.min(g.max_f, q3 + 1.5 * iqr)],
      mean:  [g.mean_f],
      sd:    [g.std_f],
      orientation: "h" as const,
    };
  });

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", gap: 4 }}>
      {/* eta^2 bar chart */}
      <div style={{ flex: "0 0 45%", minHeight: 0 }}>
        <Plot
          data={[{
            type: "bar",
            orientation: "h",
            x: etaSq,
            y: groups,
            marker: { color: etaSq, colorscale: "Viridis", showscale: false },
            hovertemplate: "%{y}: eta2 = %{x:.4f}<extra></extra>",
          }]}
          layout={{
            ...baseLayout,
            margin: { l: 120, r: 24, t: 40, b: 40 },
            title: { text: `eta2 por grupo   --   ${subtitle}`, font: { size: 12 } },
            xaxis: { title: "eta2", gridcolor: "#1f2838", zeroline: true, zerolinecolor: "#3a4460" },
            yaxis: { gridcolor: "#1f2838", automargin: true },
          }}
          config={snapConfig}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>

      {/* Box-plot fitness distribution */}
      <div style={{ flex: "0 0 50%", minHeight: 0 }}>
        <Plot
          data={boxTraces as never[]}
          layout={{
            ...baseLayout,
            margin: { l: 120, r: 24, t: 8, b: 48 },
            showlegend: false,
            xaxis: { title: "fitness f [kg]", gridcolor: "#1f2838", zeroline: false },
            yaxis: { gridcolor: "#1f2838", automargin: true },
            boxmode: "group",
          }}
          config={snapConfig}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
