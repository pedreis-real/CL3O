import { create } from "zustand";
import { api } from "../api/client";
import type {
  DistinctEntry, Info, Manifest, Mesh, Planform, RunSummary,
  Section, Stress, ViewKind,
} from "../types";

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
  dispComp: string;       // legacy alias for contourComp (Femap "contour")
  deformComp: string;     // Femap-style "Deform" selector
  contourComp: string;    // Femap-style "Contour" selector
  scaleLog: boolean;      // log-spaced deform scale slider
  forceFrame: "local" | "global";
  forceComp: string;
  nLoadcases: number;

  // section tab controls
  showSectionAxes: boolean;

  // stress tab controls
  stressMode: "stress" | "flux";

  // misc tab sub-view
  miscTab: "convergence" | "search" | "sensitivity";

  // distinct individuals (cross-generation dedup index)
  distinctIndividuals: DistinctEntry[];
  distinctIndex: number;

  // colorbar limits — when fixed, plots use [colorMin, colorMax]
  colorScaleFixed: boolean;
  colorMin: number | null;
  colorMax: number | null;

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
  setDeformComp: (c: string) => void;
  setContourComp: (c: string) => void;
  setScaleLog: (b: boolean) => void;
  setForceFrame: (fr: "local" | "global") => void;
  setForceComp: (c: string) => void;
  setNLoadcases: (n: number) => void;
  setShowSectionAxes: (b: boolean) => void;
  setStressMode: (m: "stress" | "flux") => void;
  setMiscTab: (t: "convergence" | "search" | "sensitivity") => void;
  setDistinctIndex: (i: number) => void;
  setColorScaleFixed: (b: boolean) => void;
  setColorLimits: (lo: number | null, hi: number | null) => void;
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
  deformComp: "t",
  contourComp: "w",
  scaleLog: false,
  forceFrame: "local",
  forceComp: "Mz",
  nLoadcases: 1,
  showSectionAxes: true,
  stressMode: "stress",
  miscTab: "convergence",
  distinctIndividuals: [],
  distinctIndex: 0,
  colorScaleFixed: false,
  colorMin: null,
  colorMax: null,
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
      const distinct = (manifest.distinct_individuals ?? []).slice();
      // Pre-position to the distinct entry whose first_gen == best_gen
      // (or 0 when none of the distinct individuals matches).
      const bg = manifest.best_gen ?? 0;
      const dIdx = Math.max(
        0,
        distinct.findIndex((d) => (d.first_seen_gen ?? d.k) === bg),
      );
      set({
        runId: id,
        manifest,
        gen: bg,
        station: 0,
        playing: false,
        planform: null,
        section: null,
        mesh: null,
        stress: null,
        distinctIndividuals: distinct,
        distinctIndex: dIdx,
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
  setDispComp: (c) => set({ dispComp: c, contourComp: c }),
  setDeformComp: (c) => set({ deformComp: c }),
  setContourComp: (c) => set({ contourComp: c, dispComp: c }),
  setScaleLog: (b) => set({ scaleLog: b }),
  setForceFrame: (fr) => set({
    forceFrame: fr,
    forceComp: fr === "local" ? "Mz" : "MZ",
  }),
  setForceComp: (c) => set({ forceComp: c }),
  setNLoadcases: (n) => set({ nLoadcases: n }),
  setShowSectionAxes: (b) => set({ showSectionAxes: b }),
  setStressMode: (m) => set({ stressMode: m }),
  setMiscTab: (t) => set({ miscTab: t }),
  setDistinctIndex: (i) => {
    const d = get().distinctIndividuals;
    if (!d.length) return;
    const ci = Math.max(0, Math.min(i, d.length - 1));
    const k = d[ci].first_seen_gen ?? d[ci].k;
    set({ distinctIndex: ci, gen: k });
    void get().refreshInfo();
  },
  setColorScaleFixed: (b) => set({ colorScaleFixed: b }),
  setColorLimits: (lo, hi) => set({ colorMin: lo, colorMax: hi }),
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
