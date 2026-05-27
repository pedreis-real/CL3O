import { useStore } from "../state/store";
import { ForceSelector } from "./ForceSelector";

function fmt(v: number | null | undefined, digits = 2, unit = ""): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  const s = a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(2) : v.toFixed(digits);
  return unit ? `${s} ${unit}` : s;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="kv">
      <span className="k">{label}</span>
      <span className="v">{value}</span>
    </div>
  );
}

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
      {view === "geometry" && (
        <section className="panel">
          <h4>Planform</h4>
          <Row label="span b" value={fmt(s.planform?.span, 0, "mm")} />
          <Row label="area" value={fmt(s.planform?.area, 0, "mm²")} />
          <Row label="aspect ratio" value={fmt(s.planform?.AR, 2)} />
          <Row label="MAC" value={fmt(s.planform?.mac, 1, "mm")} />
          <Row label="root chord" value={fmt(s.planform?.root_chord, 0, "mm")} />
          <Row label="tip chord" value={fmt(s.planform?.tip_chord, 0, "mm")} />
        </section>
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
          <Row label="y (span)" value={fmt(s.section?.y, 0, "mm")} />
          <Row label="chord" value={fmt(s.section?.chord, 0, "mm")} />
          <Row label="area" value={fmt(s.section?.props.area, 1, "mm²")} />
          <Row label="I_XX" value={fmt(s.section?.props.I_XX, 2, "mm⁴")} />
          <Row label="I_ZZ" value={fmt(s.section?.props.I_ZZ, 2, "mm⁴")} />
          <Row label="J" value={fmt(s.section?.props.J, 2, "mm⁴")} />
          <Row label="spar xw1" value={fmt(s.section?.props.xw1, 3)} />
          <Row label="spar xw2" value={fmt(s.section?.props.xw2, 3)} />
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
                component
                <select value={s.dispComp} onChange={(e) => s.setDispComp(e.target.value)}>
                  <option value="u">T1 · u (mm)</option>
                  <option value="v">T2 · v (mm)</option>
                  <option value="w">T3 · w (mm)</option>
                  <option value="t">total |translation|</option>
                  <option value="rx">R1 · θx (rad)</option>
                  <option value="ry">R2 · θy (rad)</option>
                  <option value="rz">R3 · θz (rad)</option>
                  <option value="r">total |rotation|</option>
                </select>
              </label>
              <label className="control">
                deform scale ×{s.scale.toFixed(s.scale < 1 ? 2 : 1)}
                <input type="range" min={0} max={50} step={0.05} value={s.scale}
                       onChange={(e) => s.setScale(Number(e.target.value))} />
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
              {(["A", "avg", "B"] as const).map((e) => (
                <label key={e} className={`force-row${s.end === e ? " active" : ""}`}>
                  <input
                    type="radio"
                    name="stress-end"
                    checked={s.end === e}
                    onChange={() => s.setEnd(e)}
                  />
                  {e}
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

function FluxDisclaimer() {
  return (
    <div className="flux-disclaimer">
      <p className="disclaimer-text">
        Positive shear flow <b>q &gt; 0</b> follows the
        counter-clockwise (CCW) convention for each closed cell,
        consistent with the Megson idealisation used in CL3O.
        Arrows below show the positive sense for cells I–III.
      </p>
      {/* Schematic: 3-cell box with CCW arrows on each cell */}
      <svg viewBox="0 0 220 100" className="flux-svg">
        {/* cell outlines */}
        <rect x="10"  y="20" width="60" height="60" rx="3"
              fill="rgba(79,140,255,0.08)" stroke="#4f8cff" strokeWidth="1.2"/>
        <rect x="80"  y="20" width="60" height="60" rx="3"
              fill="rgba(46,204,113,0.08)" stroke="#26a76e" strokeWidth="1.2"/>
        <rect x="150" y="20" width="60" height="60" rx="3"
              fill="rgba(255,209,102,0.08)" stroke="#e6ce00" strokeWidth="1.2"/>
        {/* cell labels */}
        <text x="40"  y="55" textAnchor="middle" fontSize="11" fill="#c9d4e3">I</text>
        <text x="110" y="55" textAnchor="middle" fontSize="11" fill="#c9d4e3">II</text>
        <text x="180" y="55" textAnchor="middle" fontSize="11" fill="#c9d4e3">III</text>
        {/* CCW arrows — top going left, bottom going right */}
        {/* Cell I */}
        <path d="M55,22 L25,22" stroke="#4f8cff" strokeWidth="1.5" markerEnd="url(#arr-b)" fill="none"/>
        <path d="M15,78 L45,78" stroke="#4f8cff" strokeWidth="1.5" markerEnd="url(#arr-b)" fill="none"/>
        {/* Cell II */}
        <path d="M125,22 L95,22" stroke="#26a76e" strokeWidth="1.5" markerEnd="url(#arr-g)" fill="none"/>
        <path d="M85,78 L115,78" stroke="#26a76e" strokeWidth="1.5" markerEnd="url(#arr-g)" fill="none"/>
        {/* Cell III */}
        <path d="M195,22 L165,22" stroke="#e6ce00" strokeWidth="1.5" markerEnd="url(#arr-y)" fill="none"/>
        <path d="M155,78 L185,78" stroke="#e6ce00" strokeWidth="1.5" markerEnd="url(#arr-y)" fill="none"/>
        {/* arrow markers */}
        <defs>
          <marker id="arr-b" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#4f8cff"/>
          </marker>
          <marker id="arr-g" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#26a76e"/>
          </marker>
          <marker id="arr-y" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#e6ce00"/>
          </marker>
        </defs>
        {/* CCW label */}
        <text x="110" y="96" textAnchor="middle" fontSize="9" fill="#7c8aa5">↺ positive direction (CCW)</text>
      </svg>
    </div>
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
