import Plotly from "plotly.js-dist-min";
import { useStore } from "../state/store";
import { config } from "../plots/Plot";

// Build the "save snapshot" modebar button: downloads a PNG locally and
// fires it to the server (POST /api/snaps). Not a React hook despite the
// payload it carries — kept as a plain factory so it can be composed by
// useSnapshotConfig below.
export function snapshotButton(runId: string | null, gen: number, view: string) {
  return {
    name:  "save-snapshot",
    title: "Save snapshot (download + server)",
    icon:  (Plotly as any).Icons.camera,
    click: async (gd: HTMLElement) => {
      const dataUrl = await (Plotly as any).toImage(gd, { format: "png", scale: 2 });

      // 1. Browser download
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `cl3o_${runId ?? "run"}_gen${String(gen).padStart(4, "0")}_${view}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      // 2. Server save (fire-and-forget)
      if (runId) {
        const b64 = dataUrl.split(",")[1] ?? dataUrl;
        fetch("/api/snaps", {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ run_id: runId, gen, view, data: b64 }),
        }).catch(console.error);
      }
    },
  };
}

// Shared Plotly config with the snapshot button wired in for `view`. Reads
// the current run/generation from the store so callers pass only the view
// label. Call at the top of a component (it subscribes to the store).
export function useSnapshotConfig(view: string) {
  const runId = useStore((s) => s.runId);
  const gen   = useStore((s) => s.gen);
  return { ...config, modeBarButtonsToAdd: [snapshotButton(runId, gen, view)] as any[] };
}
