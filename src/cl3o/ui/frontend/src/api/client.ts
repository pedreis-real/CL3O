import type {
  Forces, Info, Manifest, Mesh, Planform, RunSummary, Scene, Section, Stress, StressScene,
} from "../types";

// All requests go to /api (proxied to the FastAPI backend by Vite in dev).
async function get<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`);
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${path}\n${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  runs: () => get<RunSummary[]>("/runs"),
  manifest: (run: string) => get<Manifest>(`/runs/${run}/manifest`),
  planform: (run: string) => get<Planform>(`/runs/${run}/planform`),
  geometry: (run: string, k: number, deformed = false, lc = 0, scale = 1) =>
    get<Scene>(`/runs/${run}/gen/${k}/geometry?deformed=${deformed}&lc=${lc}&scale=${scale}`),
  forces: (run: string, k: number, lc = 0) =>
    get<Forces>(`/runs/${run}/gen/${k}/forces?lc=${lc}`),
  info: (run: string, k: number) => get<Info>(`/runs/${run}/gen/${k}/info`),
  section: (run: string, k: number, station: number) =>
    get<Section>(`/runs/${run}/gen/${k}/section/${station}`),
  mesh: (run: string, k: number, lc: number, deformed: boolean, scale: number) =>
    get<Mesh>(`/runs/${run}/gen/${k}/mesh?lc=${lc}&deformed=${deformed}&scale=${scale}`),
  stress: (run: string, k: number, lc: number, end: string) =>
    get<Stress>(`/runs/${run}/gen/${k}/stress?lc=${lc}&end=${end}`),
  stress3d: (run: string, k: number, lc = 0, end = "avg") =>
    get<StressScene>(`/runs/${run}/gen/${k}/stress3d?lc=${lc}&end=${end}`),
};
