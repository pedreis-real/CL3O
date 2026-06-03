import { useStore } from "../state/store";

export function RunPicker() {
  const { runs, runId, selectRun } = useStore();
  return (
    <div className="run-picker">
      <span className="brand">CL3O</span>
      <select
        value={runId ?? ""}
        onChange={(e) => void selectRun(e.target.value)}
        disabled={!runs.length}
      >
        {runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.run_label} · {r.n_gens + 1} gens
          </option>
        ))}
      </select>
    </div>
  );
}
