import React from "react";
import { useStore } from "../state/store";
import { ForceSelector } from "./ForceSelector";
import { FluxDisclaimer } from "./FluxDisclaimer";

function fmt(v: number | null | undefined, digits = 2, unit = ""): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  const s = a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(2) : v.toFixed(digits);
  return unit ? `${s} ${unit}` : s;
}

// Chord-fraction value (e.g., 0.3214) -> "32.14 %c"
function fmtPct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(2)} %c`;
}

function Row({ label, value }: { label: React.ReactNode; value: string }) {
  return (
    <div className="kv">
      <span className="k">{label}</span>
      <span className="v">{value}</span>
    </div>
  );
}

// Display names for stress endpoints; backend keeps A/B/avg enums.
const STRESS_END_LABEL: Record<string, string> = {
  A: "Max",
  B: "Min",
  avg: "Avg",
};
const STRESS_END_ORDER = ["A", "B", "avg"] as const;

// Femap-style 8 components used by both the Deform and Contour selectors.
const DISP_COMPONENTS: [string, string][] = [
  ["u",  "T1 · u (mm)"],
  ["v",  "T2 · v (mm)"],
  ["w",  "T3 · w (mm)"],
  ["t",  "Total Translation"],
  ["rx", "R1 · θx (deg)"],
  ["ry", "R2 · θy (deg)"],
  ["rz", "R3 · θz (deg)"],
  ["r",  "Total Rotation (deg)"],
];

export function Sidebar() {
  const s = useStore();
  const { view, info } = s;

  return (
    <aside className="sidebar">
      {/* Always-on run / generation header */}
      <section className="panel">
        <h3>{s.manifest?.run_label ?? "—"}</h3>
        <Row label="generation" value={`${s.gen} / ${s.manifest?.n_gens ?? 0}`} />
        <Row label="mass m(X)" value={fmt(info?.fitness.score, 3, "kg")} />
        <Row label="penalty P(X)" value={fmt(info?.fitness.penalty, 3)} />
        <Row label="fitness z(X)" value={fmt(info?.fitness.total, 3)} />
        <div className={info?.fitness.is_feasible ? "badge ok" : "badge bad"}>
          {info?.fitness.is_feasible ? "FEASIBLE" : "INFEASIBLE"}
        </div>
        <Row label="min TSW MS" value={fmt(info?.tsw.MS_min, 3)} />
        <Row label="min displ MS" value={fmt(info?.displacement.MS_min, 3)} />
      </section>

      {/* View-specific panel, driven by the top-right toggle */}
      {view === "misc" && (
        <section className="panel">
          <h4>Optimization</h4>
          <Row label="schema"   value={s.manifest?.schema_version ?? "—"} />
          <Row label="created"  value={s.manifest?.created_at?.slice(0, 19).replace("T", " ") ?? "—"} />
          <Row label="D"        value={fmt(s.manifest?.D, 0)} />
          <Row label="NP"       value={fmt(s.manifest?.NP, 0)} />
          <Row label="seed"     value={fmt(s.manifest?.seed, 0)} />
          <Row label="n_gens"   value={fmt(s.manifest?.n_gens, 0)} />
          <Row label="snapshots" value={fmt(s.manifest?.snapshots?.length, 0)} />
          <Row label="distinct" value={fmt(s.distinctIndividuals.length, 0)} />
          <Row label="best_gen" value={fmt(s.manifest?.best_gen, 0)} />
          <Row label="best f"   value={fmt(
            s.manifest?.best_f_hist?.[s.manifest?.best_gen ?? 0] ?? null,
            3,
          )} />
        </section>
      )}

      {view === "misc" && s.miscTab === "search" && (
        <section className="panel">
          <h4>About PC1 / PC2</h4>
          <p className="hint">
            2-D PCA projection of every distinct design vector X (dim D = 11·n_cpts + 3).
            PC1 and PC2 are the two orthogonal directions in design space that capture
            the most variance across the DE walk. Each axis label shows the explained
            variance — higher = the trajectory is well represented in 2-D. Marker colour
            encodes fitness z(X).
          </p>
        </section>
      )}

      {view === "geometry" && (
        <section className="panel">
          <h4>Planform</h4>
          <Row label="b"   value={fmt(s.planform?.span, 0, "mm")} />
          <Row label="S"   value={fmt(s.planform?.area, 0, "mm²")} />
          <Row label="AR"  value={fmt(s.planform?.AR, 2)} />
          <Row label="MAC" value={fmt(s.planform?.mac, 1, "mm")} />
          <Row label="cr"  value={fmt(s.planform?.root_chord, 0, "mm")} />
          <Row label="ct"  value={fmt(s.planform?.tip_chord, 0, "mm")} />
        </section>
      )}

      {view === "geometry" && s.info?.optvars && s.info.optvars.length > 0 && (
        <OptVarsPanel optvars={s.info.optvars} />
      )}

      {view === "section" && (
        <section className="panel">
          <h4>Cross-section</h4>
          <label className="control">
            station {s.section?.station ?? s.station} / {(s.section?.n_stations ?? 1) - 1}
            <input
              type="range"
              min={0}
              max={(s.section?.n_stations ?? 1) - 1}
              value={s.station}
              onChange={(e) => s.setStation(Number(e.target.value))}
            />
          </label>
          <Row label="y position" value={fmt(s.section?.y, 0, "mm")} />
          <Row label="chord" value={fmt(s.section?.chord, 0, "mm")} />
          <Row label="area" value={fmt(s.section?.props.area, 1, "mm²")} />
          <Row label={<>I<sub>XX</sub></>} value={fmt(s.section?.props.I_XX, 2, "mm⁴")} />
          <Row label={<>I<sub>ZZ</sub></>} value={fmt(s.section?.props.I_ZZ, 2, "mm⁴")} />
          <Row label={<>I<sub>XZ</sub></>} value={fmt(s.section?.props.I_XZ, 2, "mm⁴")} />
          <Row label="J" value={fmt(s.section?.props.J, 2, "mm⁴")} />
          <Row label="aft spar" value={fmtPct(s.section?.props.xw1)} />
          <Row label="rear spar" value={fmtPct(s.section?.props.xw2)} />
          <label className="control inline-toggle">
            <input
              type="checkbox"
              checked={s.showSectionAxes}
              onChange={(e) => s.setShowSectionAxes(e.target.checked)}
            />
            show axes
          </label>
        </section>
      )}

      {view === "mesh" && (
        <section className="panel">
          <h4>Post-processing</h4>
          <label className="control">
            field
            <select value={s.field} onChange={(e) => s.setField(e.target.value as "disp" | "forces")}>
              <option value="disp">Displacement (surface colormap)</option>
              <option value="forces">Internal forces (beam diagram)</option>
            </select>
          </label>

          {s.field === "disp" ? (
            <>
              <label className="control">
                contour
                <select value={s.contourComp} onChange={(e) => s.setContourComp(e.target.value)}>
                  {DISP_COMPONENTS.map(([v, l]) => (
                    <option key={v} value={v}>{l}</option>
                  ))}
                </select>
              </label>
              <label className="control">
                deform scale ×{s.scale.toFixed(s.scale < 1 ? 2 : 1)}
                <input
                  type="range"
                  min={s.scaleLog ? 0 : 0}
                  max={10}
                  step={0.1}
                  value={s.scaleLog ? Math.log10(Math.max(1e-2, s.scale)) * (10 / 1) : s.scale}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    s.setScale(s.scaleLog ? Math.pow(10, v / 10) : v);
                  }}
                />
              </label>
              <label className="control inline-toggle">
                <input
                  type="checkbox"
                  checked={s.scaleLog}
                  onChange={(e) => s.setScaleLog(e.target.checked)}
                />
                log scale
              </label>
            </>
          ) : (
            <ForceSelector />
          )}
          <LoadcaseSelect />
        </section>
      )}

      {view === "stress" && (
        <section className="panel">
          <h4>Stress / Flux</h4>
          <div className="force-selector">
            <div className="force-col">
              <span className="force-col-label">element end</span>
              {STRESS_END_ORDER.map((e) => (
                <label key={e} className={`force-row${s.end === e ? " active" : ""}`}>
                  <input
                    type="radio"
                    name="stress-end"
                    checked={s.end === e}
                    onChange={() => s.setEnd(e)}
                  />
                  {STRESS_END_LABEL[e]}
                </label>
              ))}
            </div>
          </div>
          <LoadcaseSelect />
          {s.stressMode === "flux" && <FluxDisclaimer />}
        </section>
      )}

      {s.error && <section className="panel error">{s.error}</section>}
    </aside>
  );
}

function LoadcaseSelect() {
  const s = useStore();
  const nc = s.nLoadcases;
  if (nc <= 1) return null;
  return (
    <label className="control">
      load case
      <select value={s.loadcase} onChange={(e) => s.setLoadcase(Number(e.target.value))}>
        {Array.from({ length: nc }, (_, i) => (
          <option key={i} value={i}>
            LC {i}
          </option>
        ))}
      </select>
    </label>
  );
}

function OptVarsPanel({ optvars }: { optvars: number[] }) {
  const [open, setOpen] = React.useState(false);
  return (
    <section className="panel">
      <div
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
        onClick={() => setOpen((v) => !v)}
      >
        <h4 style={{ margin: 0 }}>Design vector X (D={optvars.length})</h4>
        <span style={{ color: "var(--text-dim)", fontSize: 11 }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <pre className="optvars-pre">
          {optvars.map((v, i) => `x[${String(i).padStart(2, "0")}] = ${v.toFixed(5)}`).join("\n")}
        </pre>
      )}
    </section>
  );
}
