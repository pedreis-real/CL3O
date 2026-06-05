import { useStore } from "../state/store";
import Plot, { baseLayout } from "./Plot";
import { useSnapshotConfig } from "../hooks/useSnapshotButton";

// Best / mean / std fitness history across DE generations.
export function ConvergenceView() {
  const { manifest } = useStore();
  const snapConfig = useSnapshotConfig("convergence");
  if (!manifest) return null;
  const k = (manifest.best_f_hist || []).map((_, i) => i);
  return (
    <Plot
      data={[
        {
          x: k, y: manifest.best_f_hist as number[],
          type: "scatter", mode: "lines",
          name: "best f",
          line: { color: "#4f8cff", width: 2 },
        },
        {
          x: k, y: manifest.mean_f_hist as number[],
          type: "scatter", mode: "lines",
          name: "mean f",
          line: { color: "#ffd166", width: 1.2, dash: "dot" },
        },
        {
          x: k, y: manifest.std_f_hist as number[],
          type: "scatter", mode: "lines", yaxis: "y2",
          name: "std f",
          line: { color: "#17b18a", width: 1.2 },
        },
      ]}
      layout={{
        ...baseLayout,
        showlegend: true,
        legend: { x: 0.5, xanchor: "center", y: 1.02, yanchor: "bottom", orientation: "h", font: { size: 11 } },
        margin: { l: 56, r: 56, t: 48, b: 48 },
        xaxis: { title: "generation", gridcolor: "#1f2838", zeroline: false },
        yaxis: { title: "fitness z(X)", gridcolor: "#1f2838", zeroline: false },
        yaxis2: {
          title: "std f", overlaying: "y", side: "right",
          showgrid: false, zeroline: false,
        },
      }}
      config={snapConfig}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
