import { create } from "zustand";
import { api } from "../api/client";
import type { Info, Manifest, Mesh, Planform, RunSummary, Section, Stress, ViewKind } from "../types";

interface AppState {
  // selection
  runs: RunSummary[];
  runId: string | null;
  manifest: Manifest | null;
  info: Info | null;

  // view controls
  view: ViewKind;
  gen: number;
  station: number;
  loadcase: number;
  deformed: boolean;
  scale: number;
  end: string;
  playing: boolean;

  // mesh post-processing controls
  field: "disp" | "forces";
  dispComp: string;
  forceFrame: "local" | "global";
  forceComp: string;
  nLoadcases: number;

  // stress tab controls
  stressMode: "stress" | "flux";

  // latest payloads (published by plots, read by the sidebar)
  planform: Planform | null;
  section: Section | null;
  mesh: Mesh | null;
  stress: Stress | null;

  error: string | null;

  // actions
  init: () => Promise<void>;
  selectRun: (id: string) => Promise<void>;
  setView: (v: ViewKind) => void;
  setGen: (k: number) => void;
  setStation: (s: number) => void;
  setLoadcase: (lc: number) => void;
  setDeformed: (d: boolean) => void;
  setScale: (s: number) => void;
  setEnd: (e: string) => void;
  setField: (f: "disp" | "forces") => void;
  setDispComp: (c: string) => void;
  setForceFrame: (fr: "local" | "global") => void;
  setForceComp: (c: string) => void;
  setNLoadcases: (n: number) => void;
  setStressMode: (m: "stress" | "flux") => void;
  togglePlay: () => void;
  refreshInfo: () => Promise<void>;
  publish: (p: Partial<Pick<AppState, "planform" | "section" | "mesh" | "stress">>) => void;
}

export const useStore = create<AppState>((set, get) => ({
  runs: [],
  runId: null,
  manifest: null,
  info: null,
  view: "geometry",
  gen: 0,
  station: 0,
  loadcase: 0,
  deformed: false,
  scale: 1,
  end: "avg",
  playing: false,
  field: "disp",
  dispComp: "w",
  forceFrame: "local",
  forceComp: "Mz",
  nLoadcases: 1,
  stressMode: "stress",
  planform: null,
  section: null,
  mesh: null,
  stress: null,
  error: null,

  init: async () => {
    try {
      const runs = await api.runs();
      set({ runs });
      if (runs.length) await get().selectRun(runs[0].run_id);
    } catch (e) {
      set({ error: String(e) });
    }
  },

  selectRun: async (id) => {
    try {
      const manifest = await api.manifest(id);
      set({
        runId: id,
        manifest,
        gen: manifest.best_gen ?? 0,
        station: 0,
        playing: false,
        planform: null,
        section: null,
        mesh: null,
        stress: null,
      });
      await get().refreshInfo();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  setView: (v) => set({ view: v }),
  setGen: (k) => {
    set({ gen: k });
    void get().refreshInfo();
  },
  setStation: (s) => set({ station: s }),
  setLoadcase: (lc) => set({ loadcase: lc }),
  setDeformed: (d) => set({ deformed: d }),
  setScale: (s) => set({ scale: s }),
  setEnd: (e) => set({ end: e }),
  setField: (f) => set({ field: f }),
  setDispComp: (c) => set({ dispComp: c }),
  setForceFrame: (fr) => set({
    forceFrame: fr,
    forceComp: fr === "local" ? "Mz" : "MZ",
  }),
  setForceComp: (c) => set({ forceComp: c }),
  setNLoadcases: (n) => set({ nLoadcases: n }),
  setStressMode: (m) => set({ stressMode: m }),
  togglePlay: () => set((s) => ({ playing: !s.playing })),

  refreshInfo: async () => {
    const { runId, gen } = get();
    if (runId == null) return;
    try {
      set({ info: await api.info(runId, gen) });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  publish: (p) => set(p),
}));
