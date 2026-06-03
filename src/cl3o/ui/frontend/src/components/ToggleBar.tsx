import { useStore } from "../state/store";
import type { ViewKind } from "../types";

const VIEWS: { key: ViewKind; label: string }[] = [
  { key: "geometry", label: "Geometry" },
  { key: "section", label: "Cross-section" },
  { key: "mesh", label: "Mesh" },
  { key: "stress", label: "Stress" },
  { key: "misc", label: "Misc" },
];

export function ToggleBar() {
  const { view, setView } = useStore();
  return (
    <div className="toggle-bar" role="tablist" aria-label="View">
      {VIEWS.map((v) => (
        <button
          key={v.key}
          role="tab"
          aria-selected={view === v.key}
          className={view === v.key ? "toggle active" : "toggle"}
          onClick={() => setView(v.key)}
        >
          {v.label}
        </button>
      ))}
    </div>
  );
}
