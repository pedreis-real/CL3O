import { useStore } from "../state/store";

const LOCAL_COMPS  = ["N", "Sy", "Sz", "T", "My", "Mz"] as const;
const GLOBAL_COMPS = ["N", "SX", "SZ", "T", "MX", "MZ"] as const;

export function ForceSelector() {
  const { forceFrame, forceComp, setForceFrame, setForceComp } = useStore();

  function pick(frame: "local" | "global", comp: string) {
    if (frame !== forceFrame) setForceFrame(frame);
    setForceComp(comp);
  }

  return (
    <div className="force-selector">
      <div className="force-col">
        <span className="force-col-label">local</span>
        {LOCAL_COMPS.map((c) => {
          const active = forceFrame === "local" && forceComp === c;
          return (
            <label key={c} className={`force-row${active ? " active" : ""}`}>
              <input
                type="radio"
                name="force-local"
                checked={active}
                onChange={() => pick("local", c)}
              />
              {c}
            </label>
          );
        })}
      </div>

      <div className="force-col">
        <span className="force-col-label">global</span>
        {GLOBAL_COMPS.map((c) => {
          const active = forceFrame === "global" && forceComp === c;
          return (
            <label key={c} className={`force-row${active ? " active" : ""}`}>
              <input
                type="radio"
                name="force-global"
                checked={active}
                onChange={() => pick("global", c)}
              />
              {c}
            </label>
          );
        })}
      </div>
    </div>
  );
}
