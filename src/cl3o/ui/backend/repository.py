'''
================================================================================
CL3O Visualization UI - Run repository (the data-retrieval layer).

Reads the archived optimization runs that GenerationArchiver
(src/optimization/de_opt.py) writes under `outputs/`:

    outputs/<run>/manifest.json
    outputs/<run>/opt_files/gen_XXXX.pkl     (or generations/ - both handled)

Each pickle is a `RuntimeData` snapshot of the best individual of one DE
generation. This module is the ONLY place pickles are loaded; everything
above it works with plain dicts / numpy arrays.

@ CL3O Authors - MIT License
================================================================================
'''

from . import paths

import importlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

from cl3o.utils.lru_cache import LRUCache
from .serialize import to_jsonable

# Backwards-compat for legacy pickles produced before the cl3o package
# layout (`outputs/da62_optteste3/...` and friends). They reference the
# top-level `optimization`, `geometry`, `fea`, etc. modules; alias them
# to the current `cl3o.*` namespace so pickle.load can resolve them.
_LEGACY_ROOTS = (
    "optimization", "geometry", "fea", "materials", "utils",
)
for _root in _LEGACY_ROOTS:
    _full = f"cl3o.{_root}"
    if _root not in sys.modules:
        try:
            sys.modules[_root] = importlib.import_module(_full)
        except Exception:
            pass

# Sub-directory names a run may use for its per-generation pickles.
_PKL_SUBDIRS = ("opt_files", "generations", "")

# Full-run cache: all snapshots for a run loaded at once.
# Key: str(run_dir), Value: {k: RuntimeData}. Bounded LRU so browsing
# many runs in one session cannot grow the cache without limit (each
# entry holds every generation pickle of one run).
_RUN_CACHE_MAXSIZE = 4
_run_full_cache: LRUCache = LRUCache(maxsize=_RUN_CACHE_MAXSIZE)

# Static-file caches: keyed by the resolved file path string.
_wing_cache: dict[str, object] = {}       # path -> WingData
_wing_raw_cache: dict[str, dict] = {}     # path -> raw dict
_airfoil_cache: dict[str, object] = {}    # path -> AirfoilData
# Laminate catalog cache: shared across runs (process-wide).
_laminate_catalog_cache: dict[str, dict] | None = None


def _infer_family(name: str, plies: list[str]) -> str:
    '''
    Classify a laminate as CFRP / GFRP / CFRP_sand / GFRP_sand / OTHER.
    Sandwich is detected by the presence of a core ply (honeycomb /
    divinycell / foam) anywhere in the stack; the face family comes from
    the first non-core ply.
    '''
    core_kw = ("honeycomb", "divinycell", "foam")
    is_sand = any(any(kw in p.lower() for kw in core_kw) for p in plies)
    face_kw = None
    for p in plies:
        pl = p.lower()
        if any(kw in pl for kw in core_kw):
            continue
        if "cfrp" in pl:
            face_kw = "CFRP"; break
        if "gfrp" in pl:
            face_kw = "GFRP"; break
    if face_kw is None:
        nl = name.lower()
        if "cfrp" in nl: face_kw = "CFRP"
        elif "gfrp" in nl: face_kw = "GFRP"
        else: face_kw = "OTHER"
    return f"{face_kw}_sand" if is_sand and face_kw in ("CFRP", "GFRP") else face_kw


def _load_laminate_catalog() -> dict[str, dict]:
    '''
    Build the shared laminate catalog (process-wide cache).
    Indexed by 1-based string keys ("1", "2", ...) to match the layup
    indices stored on OptVars.lsX / lwX / lfX.
    '''
    global _laminate_catalog_cache
    if _laminate_catalog_cache is not None:
        return _laminate_catalog_cache
    catalog: dict[str, dict] = {}
    mats_dir = paths.MATERIALS_DIR
    if mats_dir.is_dir():
        files = sorted(mats_dir.glob("MAT_*_LaminateData.json"))
        for k, fp in enumerate(files, start=1):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            name = str(raw.get("name", fp.stem))
            plies = list(raw.get("plies", []) or [])
            catalog[str(k)] = {
                "name"        : name,
                "family"      : _infer_family(name, plies),
                "E1"          : raw.get("E1"),
                "E2"          : raw.get("E2"),
                "G12"         : raw.get("G12"),
                "E1_bend"     : raw.get("E1_bend"),
                "E2_bend"     : raw.get("E2_bend"),
                "G12_bend"    : raw.get("G12_bend"),
                "stacking_seq": raw.get("stacking_seq"),
                "plies"       : plies,
                "thick"       : raw.get("thick"),
                "n_plies"     : raw.get("n_plies"),
            }
    _laminate_catalog_cache = catalog
    return catalog


@dataclass
class RunSummary:
    run_id     : str
    run_label  : str
    n_gens     : int
    D          : int
    NP         : int
    seed       : int
    created_at : str


class RunRepository:
    '''Discover, read and cache archived DE runs under `outputs/`.'''

    def __init__(self, outputs_dir: Path | None = None) -> None:
        self.outputs_dir = Path(outputs_dir) if outputs_dir else paths.OUTPUTS_DIR

    # ------------------------------------------------------------------
    # Run discovery
    # ------------------------------------------------------------------

    def list_runs(self) -> list[dict]:
        '''Every sub-directory holding a manifest.json, newest first.'''
        runs: list[RunSummary] = []
        if not self.outputs_dir.is_dir():
            return []
        for d in sorted(self.outputs_dir.iterdir()):
            mf = d / "manifest.json"
            if not mf.is_file():
                continue
            try:
                m = self._read_json(mf)
            except Exception:
                continue
            runs.append(RunSummary(
                run_id     = d.name,
                run_label  = str(m.get("run_label", d.name)),
                n_gens     = int(m.get("n_gens", 0)),
                D          = int(m.get("D", 0)),
                NP         = int(m.get("NP", 0)),
                seed       = int(m.get("seed", 0)),
                created_at = str(m.get("created_at", "")),
            ))
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return [r.__dict__ for r in runs]

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def get_manifest(self, run_id: str) -> dict:
        '''Parsed + sanitized manifest (Infinity/NaN -> null).

        Legacy archives (schema_version "1.0") lack `distinct_individuals`
        and per-snapshot `is_duplicate` / `first_seen_gen` fields; this
        method back-fills them by treating every snapshot as distinct so
        the frontend keeps working uniformly.
        '''
        mf = self._run_dir(run_id) / "manifest.json"
        if not mf.is_file():
            raise FileNotFoundError(f"manifest.json not found for run '{run_id}'")
        manifest = to_jsonable(self._read_json(mf))
        manifest["best_gen"] = self.best_generation(run_id)
        self._backfill_distinct(manifest)
        return manifest

    @staticmethod
    def _backfill_distinct(manifest: dict) -> None:
        '''Mutates `manifest` to ensure distinct_individuals + per-snapshot
        is_duplicate/first_seen_gen exist (no-op when already present).'''
        snaps = manifest.get("snapshots") or []
        if not snaps:
            manifest.setdefault("distinct_individuals", [])
            return
        already_tagged = any("is_duplicate" in s for s in snaps)
        if not already_tagged:
            for s in snaps:
                s["is_duplicate"]   = False
                s["first_seen_gen"] = int(s.get("k", 0))
        if "distinct_individuals" not in manifest:
            manifest["distinct_individuals"] = [
                s for s in snaps if not s.get("is_duplicate", False)
            ]

    def best_generation(self, run_id: str) -> int:
        '''Index of the best generation: best feasible if any, else argmin best_f.'''
        m = self._read_json(self._run_dir(run_id) / "manifest.json")
        snaps = m.get("snapshots", []) or []
        feasible = [s for s in snaps if s.get("is_feasible")]
        pool = feasible if feasible else snaps
        if not pool:
            return 0
        best = min(pool, key=lambda s: _finite(s.get("best_f")))
        return int(best.get("k", 0))

    # ------------------------------------------------------------------
    # Snapshots (pickled RuntimeData)
    # ------------------------------------------------------------------

    def get_snapshot(self, run_id: str, k: int):
        '''Deserialized RuntimeData for generation `k` (full-run cached).

        On first access for a run, ALL generation pickles are loaded at once
        and stored in `_run_full_cache`. Subsequent calls are pure dict lookup.
        '''
        run_dir = self._run_dir(run_id)
        run_key = str(run_dir)
        cache = _run_full_cache.get(run_key)
        if cache is None:
            manifest = self._read_json(run_dir / "manifest.json")
            cache = _preload_run(run_dir, manifest)
            _run_full_cache[run_key] = cache
        ki = int(k)
        if ki not in cache:
            raise FileNotFoundError(
                f"[CL3O] Snapshot k={k} not found in run '{run_id}'."
            )
        return cache[ki]

    def distinct_snapshots(self, run_id: str) -> list[tuple[dict, object]]:
        '''Return [(snapshot_record, RuntimeData)] for each distinct individual.

        The record is the manifest entry under `distinct_individuals` (or a
        back-filled equivalent). RuntimeData comes from get_snapshot which
        is cached, so repeat calls in a run session stay cheap.
        '''
        manifest = self.get_manifest(run_id)
        records = manifest.get("distinct_individuals") or []
        out: list[tuple[dict, object]] = []
        for rec in records:
            try:
                rt = self.get_snapshot(run_id, int(rec.get("k", 0)))
            except Exception:
                continue
            out.append((rec, rt))
        return out

    def n_gens(self, run_id: str) -> int:
        m = self._read_json(self._run_dir(run_id) / "manifest.json")
        return int(m.get("n_gens", 0))

    # ------------------------------------------------------------------
    # Wing database (for the planform / geometry view)
    # ------------------------------------------------------------------

    def _wing_path(self, run_id: str) -> Path | None:
        '''Resolve the wing-DB JSON path best matching the run label.

        Matches `<prefix>_WingData.json` against the run label / id prefix
        (e.g. label "DA62_..." -> da62_WingData.json); falls back to the
        only file present.
        '''
        files = sorted(paths.WINGS_DIR.glob("*.json")) if paths.WINGS_DIR.is_dir() else []
        if not files:
            return None
        label = str(self._read_json(self._run_dir(run_id) / "manifest.json")
                    .get("run_label", run_id)).lower()
        for fp in files:
            prefix = fp.stem.lower().split("_wingdata")[0]
            if prefix and (label.startswith(prefix) or prefix in label):
                return fp
        return files[0]

    def get_wing(self, run_id: str) -> dict | None:
        '''Best-matching wing DB as a raw dict (cached by file path).'''
        fp = self._wing_path(run_id)
        if fp is None:
            return None
        key = str(fp)
        if key not in _wing_raw_cache:
            _wing_raw_cache[key] = self._read_json(fp)
        return _wing_raw_cache[key]

    def get_wing_data(self, run_id: str):
        '''Best-matching wing DB as a typed WingData (cached by file path).'''
        from cl3o.utils import io_utils as io
        from cl3o.geometry.wing import WingData
        fp = self._wing_path(run_id)
        if fp is None:
            return None
        key = str(fp)
        if key not in _wing_cache:
            _wing_cache[key] = io.read_json(fp, WingData)
        return _wing_cache[key]

    def get_laminate_catalog(self, run_id: str | None = None) -> dict[str, dict]:
        '''Return the shared laminate catalog {idx: {name, family}}.'''
        return _load_laminate_catalog()

    def get_airfoil(self, run_id: str):
        '''AirfoilData for the run's wing (cached by file path).'''
        from cl3o.utils import io_utils as io
        from cl3o.geometry.airfoil import AirfoilData
        wing = self.get_wing(run_id)
        if wing is None:
            return None
        afl_dir = paths.ROOT_DIR / "data" / "airfoils"
        for nm in (wing.get("afl_lst") or []):
            cand = afl_dir / f"{nm}_AirfoilData.json"
            if cand.is_file():
                key = str(cand)
                if key not in _airfoil_cache:
                    _airfoil_cache[key] = io.read_json(cand, AirfoilData)
                return _airfoil_cache[key]
        files = sorted(afl_dir.glob("*_AirfoilData.json"))
        if not files:
            return None
        key = str(files[0])
        if key not in _airfoil_cache:
            _airfoil_cache[key] = io.read_json(files[0], AirfoilData)
        return _airfoil_cache[key]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        d = self.outputs_dir / run_id
        if not d.is_dir():
            raise FileNotFoundError(f"run '{run_id}' not found under {self.outputs_dir}")
        return d

    @staticmethod
    def _read_json(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def _preload_run(run_dir: Path, manifest: dict) -> dict[int, object]:
    '''Load every generation pickle for a run into a {k: snapshot} dict.'''
    cache: dict[int, object] = {}
    for s in manifest.get("snapshots", []) or []:
        k = int(s.get("k", -1))
        fname = str(s.get("file", f"gen_{k:04d}.pkl"))  # CAUTION: gen iter k are written :04d by default
        try:
            path = _resolve_pkl(run_dir, fname)
            with open(path, "rb") as fh:
                cache[k] = pickle.load(fh)
        except Exception:
            pass
    return cache


def _finite(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("inf")
    return f if f == f and abs(f) != float("inf") else float("inf")


def _resolve_pkl(run_dir: Path, filename: str) -> Path:
    '''Locate `filename` across the known sub-dirs; fall back to a glob.'''
    name = Path(filename).name
    for sub in _PKL_SUBDIRS:
        cand = (run_dir / sub / name) if sub else (run_dir / name)
        if cand.is_file():
            return cand
    hits = list(run_dir.rglob(name))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"snapshot '{name}' not found under {run_dir}")


