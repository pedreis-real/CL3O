import { useEffect, useState } from "react";
import { useStore } from "../state/store";
import { api } from "../api/client";
import type { SearchSpace } from "../types";
import Plot, { baseLayout, config } from "./Plot";
import { SEARCH_CMAP } from "./colors";

type SensKind = "spars" | "flanges" | "skin";

const TABS: { key: "convergence" | "search" | "sensitivity"; label: string }[] = [
  { key: "convergence", label: "Convergence" },
  { key: "search",      label: "Search space" },
  { key: "sensitivity", label: "Sensitivity" },
];

const SENS_KINDS: { key: SensKind; label: string }[] = [
  { key: "spars",   label: "Spars" },
  { key: "flanges", label: "Flanges" },
  { key: "skin",    label: "Skin" },
];

export function MiscPlot() {
  const { manifest, distinctIndividuals, miscTab: tab, setMiscTab: setTab } = useStore();
  const [sens, setSens] = useState<SensKind>("spars");

  if (!manifest) return <div className="plot-loading">No run loaded…</div>;

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
      <div className="plot-toolbar">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={tab === t.key ? "active" : ""}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
        {tab === "sensitivity" && (
          <select
            value={sens}
            onChange={(e) => setSens(e.target.value as SensKind)}
            style={{ marginLeft: 8 }}
          >
            {SENS_KINDS.map((k) => (
              <option key={k.key} value={k.key}>{k.label}</option>
            ))}
          </select>
        )}
      </div>

      <div style={{ flex: 1, minHeight: 0 }}>
        {tab === "convergence" && <ConvergenceView />}
        {tab === "search"      && <SearchSpaceView />}
        {tab === "sensitivity" && <SensitivityView kind={sens} hasDistinct={distinctIndividuals.length > 0} />}
      </div>
    </div>
  );
}

function ConvergenceView() {
  const { manifest } = useStore();
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
        legend: { x: 0, y: 1, font: { size: 11 } },
        xaxis: { title: "generation", gridcolor: "#1f2838", zeroline: false },
        yaxis: { title: "fitness z(X)", gridcolor: "#1f2838", zeroline: false },
        yaxis2: {
          title: "std f", overlaying: "y", side: "right",
          showgrid: false, zeroline: false,
        },
      }}
      config={config}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}

function SearchSpaceView() {
  const { runId } = useStore();
  const [data, setData] = useState<SearchSpace | null>(null);
  const [err, setErr]   = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    setErr(null);
    api.search(runId)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [runId]);

  if (err) return <div className="plot-error">{err}</div>;
  if (!data) return <div className="plot-loading">Loading search space…</div>;
  if (!data.x.length) {
    return (
      <div className="plot-empty">
        Need at least 2 distinct individuals to project the search space.
      </div>
    );
  }

  // Trajectory: ordered scatter+line on (PC1, PC2) coloured by fitness,
  // start/end markers, and a connecting line to make the DE walk readable.
  const f = data.f.map((v) => (v == null ? NaN : v));
  const finite = f.filter((v) => Number.isFinite(v)) as number[];
  const fMin = finite.length ? Math.min(...finite) : 0;
  const fMax = finite.length ? Math.max(...finite) : 1;
  const ev = data.explained_variance;

  return (
    <Plot
      data={[
        {
          x: data.x, y: data.y,
          type: "scatter", mode: "lines",
          line: { color: "#ffd166", width: 1.2 },
          opacity: 0.55,
          name: "DE trajectory",
          hoverinfo: "skip",
        },
        {
          x: data.x, y: data.y,
          type: "scatter", mode: "markers",
          marker: {
            color: f as number[],
            colorscale: SEARCH_CMAP,
            reversescale: true,
            cmin: fMin, cmax: fMax,
            size: 10,
            line: { color: "#1a1f2b", width: 1 },
            colorbar: { title: { text: "fitness z(X)" }, thickness: 12 },
          },
          text: data.gen.map((g, i) => `distinct #${i + 1}<br>gen ${g}`),
          hovertemplate: "%{text}<br>z = %{marker.color:.4g}<extra></extra>",
          name: "distinct individuals",
        },
        {
          x: [data.x[0]], y: [data.y[0]],
          type: "scatter", mode: "text+markers",
          marker: { color: "#17b18a", size: 14, symbol: "circle-open", line: { width: 2 } },
          text: ["start"], textposition: "top center",
          textfont: { color: "#17b18a", size: 10 },
          hoverinfo: "skip", showlegend: false,
        },
        {
          x: [data.x[data.x.length - 1]], y: [data.y[data.y.length - 1]],
          type: "scatter", mode: "text+markers",
          marker: { color: "#e6786a", size: 14, symbol: "x", line: { width: 2 } },
          text: ["end"], textposition: "bottom center",
          textfont: { color: "#e6786a", size: 10 },
          hoverinfo: "skip", showlegend: false,
        },
      ]}
      layout={{
        ...baseLayout,
        showlegend: false,
        margin: { l: 56, r: 80, t: 24, b: 48 },
        xaxis: {
          title: `PC1 (${(ev[0] * 100).toFixed(1)} %)`,
          gridcolor: "#1f2838", zeroline: false,
        },
        yaxis: {
          title: `PC2 (${(ev[1] * 100).toFixed(1)} %)`,
          gridcolor: "#1f2838", zeroline: false, scaleanchor: "x", scaleratio: 1,
        },
        annotations: [{
          x: 1, y: 1.04, xref: "paper", yref: "paper",
          xanchor: "right", yanchor: "bottom", showarrow: false,
          text: `${data.n_distinct} distinct · explained var = ${((ev[0] + ev[1]) * 100).toFixed(1)} %`,
          font: { color: "#7c8aa5", size: 11 },
        }],
      }}
      config={config}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}

function SensitivityView({ kind, hasDistinct }: { kind: SensKind; hasDistinct: boolean }) {
  return (
    <div className="plot-empty">
      <p style={{ margin: 0 }}>
        Sensitivity — <b>{kind}</b>
      </p>
      <p style={{ marginTop: 8, color: "var(--text-dim)", fontSize: 12 }}>
        {hasDistinct
          ? "Endpoint `/api/runs/:id/sensitivity` not published yet. Generate the dataset offline via run_single + perturbations on the best vector."
          : "No distinct individuals in the manifest — run an optimization with archive enabled."}
      </p>
    </div>
  );
}
