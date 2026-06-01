import Plotly from "plotly.js-dist-min";

export function useSnapshotButton(runId: string | null, gen: number, view: string) {
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
