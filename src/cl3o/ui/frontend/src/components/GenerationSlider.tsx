import { useEffect } from "react";
import { useStore } from "../state/store";
import Plot, { baseLayout, config } from "../plots/Plot";

export function GenerationSlider() {
  const {
    manifest, gen, setGen, playing, togglePlay,
    distinctIndividuals, distinctIndex, setDistinctIndex,
  } = useStore();

  const maxGen = manifest ? Math.max(0, manifest.n_gens) : 0;
  const best = manifest?.best_f_hist ?? [];
  const x = best.map((_, i) => i);

  // Play steps through distinct individuals (where best X actually changed)
  // when the manifest exposes them; otherwise falls back to consecutive gens.
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const s = useStore.getState();
      if (s.distinctIndividuals.length > 1) {
        const next = (s.distinctIndex + 1) % s.distinctIndividuals.length;
        s.setDistinctIndex(next);
      } else {
        const next = s.gen >= maxGen ? 0 : s.gen + 1;
        s.setGen(next);
      }
    }, 400);
    return () => clearInterval(id);
  }, [playing, maxGen]);

  if (!manifest) return <div className="gen-slider empty">No run loaded</div>;

  const denom = Math.max(1, maxGen);
  const markers = distinctIndividuals.map((d, i) => ({
    pct: ((d.first_seen_gen ?? d.k) / denom) * 100,
    k:   d.first_seen_gen ?? d.k,
    idx: i,
  }));
  const playheadPct = (gen / denom) * 100;

  return (
    <div className="gen-slider">
      <button className="play" onClick={togglePlay} title="Play / pause">
        {playing ? "❚❚" : "▶"}
      </button>

      {/* Video-editor-style timeline: spark plot is the background,
          the range input and the markers ride directly on top of it. */}
      <div className="timeline">
        <div className="timeline-spark">
          <Plot
            data={[
              {
                x,
                y: best as number[],
                type: "scatter",
                mode: "lines",
                line: { color: "#4f8cff", width: 1.5 },
                hoverinfo: "skip",
              },
            ]}
            layout={{
              ...baseLayout,
              height: 64,
              margin: { l: 0, r: 0, t: 4, b: 4 },
              xaxis: { visible: false, fixedrange: true, range: [0, maxGen] },
              yaxis: { visible: false, fixedrange: true },
            }}
            config={{ ...config, displayModeBar: false, staticPlot: true }}
            style={{ width: "100%", height: "64px" }}
            useResizeHandler
          />
        </div>

        {markers.length > 0 && (
          <div className="timeline-markers" aria-hidden="true">
            {markers.map((m) => (
              <button
                key={m.idx}
                className={`marker${m.idx === distinctIndex ? " active" : ""}`}
                style={{ left: `${m.pct}%` }}
                title={`distinct #${m.idx + 1} — first seen gen ${m.k}`}
                onClick={() => setDistinctIndex(m.idx)}
              />
            ))}
          </div>
        )}

        <div className="timeline-playhead" style={{ left: `${playheadPct}%` }} />

        <input
          className="timeline-range"
          type="range"
          min={0}
          max={maxGen}
          value={gen}
          onChange={(e) => setGen(Number(e.target.value))}
        />
      </div>

      <span className="gen-label">
        gen <b>{gen}</b> / {maxGen}
        {markers.length > 0 && (
          <>
            {" · "}
            <span className="distinct-label">
              distinct {distinctIndex + 1}/{markers.length}
            </span>
          </>
        )}
      </span>
    </div>
  );
}
