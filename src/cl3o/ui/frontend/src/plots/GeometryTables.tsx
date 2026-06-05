import { useState } from "react";
import type { LaminateEntry } from "../types";
import { buildXRows, fmtModulus } from "./geometryHelpers";

// Table of laminates used in the run — shown in the top-left overlay.
export function LayupTable({
  catalog,
  usedIndices,
  onClose,
}: {
  catalog: Record<string, LaminateEntry>;
  usedIndices: number[];
  onClose: () => void;
}) {
  const [expandedPlyRow, setExpandedPlyRow] = useState<number | null>(null);

  return (
    <div className="layup-table-panel">
      <div className="layup-table-header">
        <span>Layup catalog · click Plies cell to expand</span>
        <button className="layup-table-close" onClick={onClose}>✕</button>
      </div>
      <div className="layup-table-scroll">
        <table className="layup-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Name</th>
              <th>Em1</th>
              <th>Em2</th>
              <th>Gm12</th>
              <th>Eb1</th>
              <th>Eb2</th>
              <th>Gb12</th>
              <th>t [mm]</th>
              <th>Stack</th>
              <th>Plies</th>
            </tr>
          </thead>
          <tbody>
            {usedIndices.map((idx) => {
              const entry = catalog[String(idx)];
              const pliesExpanded = expandedPlyRow === idx;
              if (!entry) return (
                <tr key={idx}>
                  <td className="idx">{idx}</td>
                  <td className="name" colSpan={9} style={{ color: "#888", fontStyle: "italic" }}>
                    material #{idx} — catalog mismatch (run used a different material set)
                  </td>
                </tr>
              );
              return (
                <tr key={idx}>
                  <td className="idx">{idx}</td>
                  <td className="name">{entry.name.replace(/^MAT_/, "")}</td>
                  <td>{fmtModulus(entry.E1)}</td>
                  <td>{fmtModulus(entry.E2)}</td>
                  <td>{fmtModulus(entry.G12)}</td>
                  <td>{fmtModulus(entry.E1_bend)}</td>
                  <td>{fmtModulus(entry.E2_bend)}</td>
                  <td>{fmtModulus(entry.G12_bend)}</td>
                  <td>{entry.thick != null ? entry.thick.toFixed(1) : "—"}</td>
                  <td className="stack">{entry.stacking_seq ?? "—"}</td>
                  <td
                    className={`plies${pliesExpanded ? " plies-expanded" : ""}`}
                    onClick={() => setExpandedPlyRow(pliesExpanded ? null : idx)}
                    title={pliesExpanded ? "Click to collapse" : "Click to expand"}
                  >
                    {(entry.plies ?? []).join(", ")}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function XVectorTable({
  optvars,
  onClose,
}: {
  optvars: number[];
  onClose: () => void;
}) {
  const rows = buildXRows(optvars);
  return (
    <div className="layup-table-panel" style={{ left: 200 }}>
      <div className="layup-table-header">
        <span>Design vector X  (D = {optvars.length})</span>
        <button className="layup-table-close" onClick={onClose}>✕</button>
      </div>
      <div className="layup-table-scroll">
        <table className="layup-table">
          <thead>
            <tr><th>i</th><th>Variable</th><th>CP</th><th>Value</th></tr>
          </thead>
          <tbody>
            {rows.map(({ i, variable, cp, value }) => (
              <tr key={i}>
                <td className="idx">{i}</td>
                <td className="name">{variable}</td>
                <td>{cp}</td>
                <td>{value != null ? value.toFixed(4) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
