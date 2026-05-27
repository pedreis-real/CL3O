// Response shapes from the FastAPI backend (src/ui/backend). Arrays come
// through as nested number arrays; non-finite values arrive as null.

export type Vec3 = [number, number, number];
export type Vec2 = [number, number];

export interface RunSummary {
  run_id: string;
  run_label: string;
  n_gens: number;
  D: number;
  NP: number;
  seed: number;
  created_at: string;
}

export interface Snapshot {
  k: number;
  file: string;
  best_f: number | null;
  mass: number | null;
  penalty: number | null;
  is_feasible: boolean;
}

export interface Manifest {
  schema_version: string;
  run_label: string;
  created_at: string;
  n_gens: number;
  D: number;
  NP: number;
  seed: number;
  best_f_hist: (number | null)[];
  mean_f_hist: (number | null)[];
  std_f_hist: (number | null)[];
  feasible_f: number | null;
  snapshots: Snapshot[];
  best_gen: number;
}

export interface Planform {
  le: Vec3[];
  te: Vec3[];
  pos: number[];
  chords: number[];
  span: number | null;
  area: number | null;
  AR: number | null;
  mac: number | null;
  root_chord: number | null;
  tip_chord: number | null;
}

export interface Panel {
  label: string;
  pts: Vec2[];
  t: number | null;
  boomA: number;
  boomB: number;
}

export interface Cell {
  label: string;
  pts: Vec2[];
}

export interface Section {
  station: number;
  n_stations: number;
  chord: number | null;
  y: number | null;
  panels: Panel[];
  cells: Cell[];
  skin: { label: string; pts: Vec2[]; t: number | null }[];
  booms: { xy: Vec2[]; A: number[]; labels: string[] };
  centroid: [number | null, number | null];
  shear_centre: [number | null, number | null];
  props: {
    area: number | null;
    I_XX: number | null;
    I_ZZ: number | null;
    J: number | null;
    A_cells: number[];
    xw1: number | null;
    xw2: number | null;
  };
  fluxes: {
    qsX: number[];   // total shear flux per unit S_X  (10)
    qsZ: number[];   // total shear flux per unit S_Z  (10)
    qT:  number[];   // total shear flux per unit T    (10)
    qbX: number[];   // open-section flux per unit S_X (10)
    qbZ: number[];   // open-section flux per unit S_Z (10)
    qs0X: number[];  // cell constants per unit S_X    (3)
    qs0Z: number[];  // cell constants per unit S_Z    (3)
    qs0T: number[];  // cell constants per unit T      (3)
  };
}

export interface Mesh {
  nodes: Vec3[];
  elements: [number, number][];
  n_loadcases: number;
  n_nodes: number;
  n_elements: number;
  displacement?: Vec3[];
  deformed?: Vec3[];
  max_disp?: number | null;
}

export interface Stress {
  tau: number[][];
  q: number[][] | null;
  elem_scalar: number[];
  elem_mid: Vec3[];
  panel_labels: string[];
  end: string;
  min: number | null;
  max: number | null;
  n_loadcases: number;
  n_elements: number;
  n_panels: number;
}

export interface Info {
  fitness: { score: number | null; penalty: number | null; total: number | null; is_feasible: boolean };
  tsw: { MS_min: number | null; R_min: number | null; n_violations: number };
  displacement: { MS_min: number | null; n_violations: number };
  mass: { total: number | null; panels: number | null; flanges: number | null };
}

export interface Mesh3D {
  vertices: Vec3[];
  i: number[];
  j: number[];
  k: number[];
  n_chord?: number;
  n_span?: number;
  station_disp?: Record<string, number[]>;
}

export interface BoomRod {
  xyz: Vec3[];
  sigma: number[];
  label: string;
}

export interface StressScene {
  vertices: Vec3[];
  i: number[];
  j: number[];
  k: number[];
  intensity: number[];
  tau_abs: number;
  flux_qsX: number[]; flux_qsX_abs: number;
  flux_qsZ: number[]; flux_qsZ_abs: number;
  flux_qT:  number[]; flux_qT_abs:  number;
  flux_qbX: number[]; flux_qbX_abs: number;
  flux_qbZ: number[]; flux_qbZ_abs: number;
  boom_rods: BoomRod[];
  sigma_abs: number;
  n_elements: number;
  n_loadcases: number;
}

export interface Forces {
  span: number[];
  local: Record<string, number[]>;
  global: Record<string, number[]>;
  components: string[];
  n_loadcases: number;
  units: Record<string, string>;
}

export interface Scene {
  surface: Mesh3D;
  front_spar: Mesh3D;
  rear_spar: Mesh3D;
  centroid_line: Vec3[];
  shear_line: Vec3[];
  n_stations: number;
  y_span: number[];
  deformed: boolean;
}

export type ViewKind = "geometry" | "section" | "mesh" | "stress";
