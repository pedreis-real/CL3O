import { useStore } from "../state/store";
import { ConvergenceView } from "./MiscConvergence";
import { SearchSpaceView } from "./MiscSearchSpace";
import { SensitivityView } from "./MiscSensitivity";

const TABS: { key: "convergence" | "search" | "sensitivity"; label: string }[] = [
  { key: "convergence", label: "Convergence" },
  { key: "search",      label: "Search space" },
  { key: "sensitivity", label: "Sensitivity" },
];

// Tab shell for the auxiliary analytics views (convergence / search / ANOVA).
export function MiscPlot() {
  const { manifest, miscTab: tab, setMiscTab: setTab } = useStore();

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
      </div>

      <div style={{ flex: 1, minHeight: 0 }}>
        {tab === "convergence" && <ConvergenceView />}
        {tab === "search"      && <SearchSpaceView />}
        {tab === "sensitivity" && <SensitivityView />}
      </div>
    </div>
  );
}
