import { useEffect } from "react";
import { useStore } from "../state/store";
import Plot, { baseLayout, config } from "../plots/Plot";

export function GenerationSlider() {
  const { manifest, gen, setGen, playing, togglePlay } = useStore();

  const maxGen = manifest ? Math.max(0, manifest.n_gens) : 0;
  const best = manifest?.best_f_hist ?? [];
  const x = best.map((_, i) => i);

  // Animate through generations while "playing".
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const s = useStore.getState();
      const next = s.gen >= maxGen ? 0 : s.gen + 1;
      s.setGen(next);
    }, 150);
    return () => clearInterval(id);
  }, [playing, maxGen]);

  if (!manifest) return <div className="gen-slider empty">No run loaded</div>;

  return (
    <div className="gen-slider">
      <button className="play" onClick={togglePlay} title="Play / pause">
        {playing ? "❚❚" : "▶"}
      </button>

      <div className="spark">
        <Plot
          data={[
            {
              x,
              y: best as number[],
              type: "scatter",
              mode: "lines",
              line: { color: "#4f8cff", width: 1.5 },
              hovertemplate: "gen %{x}<br>best f = %{y:.4g}<extra></extra>",
            },
            {
              x: [gen],
              y: [best[gen] ?? null],
              type: "scatter",
              mode: "markers",
              marker: { color: "#ffd166", size: 9, line: { color: "#1a1f2b", width: 1 } },
              hoverinfo: "skip",
            },
          ]}
          layout={{
            ...baseLayout,
            height: 56,
            margin: { l: 36, r: 8, t: 4, b: 16 },
            xaxis: { showgrid: false, zeroline: false, tickfont: { size: 9 } },
            yaxis: { showgrid: false, zeroline: false, tickfont: { size: 9 }, nticks: 3 },
          }}
          config={config}
          style={{ width: "100%", height: "56px" }}
          useResizeHandler
        />
      </div>

      <div className="slider-row">
        <input
          type="range"
          min={0}
          max={maxGen}
          value={gen}
          onChange={(e) => setGen(Number(e.target.value))}
        />
        <span className="gen-label">
          gen <b>{gen}</b> / {maxGen}
        </span>
      </div>
    </div>
  );
}
