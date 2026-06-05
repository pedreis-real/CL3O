'''
================================================================================
CL3O Visualization UI - FastAPI backend.

Serves the archived DE-run data (under outputs/) as browser-safe JSON for the
React frontend. Every payload is sanitized (numpy -> list, NaN/Inf -> null).

Run (after `pip install -e .[ui]`):
    python -m uvicorn cl3o.ui.backend.app:app --reload --port 8000

@ CL3O Authors - MIT License
================================================================================
'''

import base64
import csv
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as _BaseModel

from cl3o.paths import ROOT_DIR, OUTPUTS_DIR
from . import extract
from . import surface
from .repository import RunRepository
from .serialize import to_jsonable

_ANOVA_RESULTS = ROOT_DIR / "tools" / "output" / "sensitivity" / "anova_results.csv"
_ANOVA_SUMMARY = ROOT_DIR / "tools" / "output" / "sensitivity" / "anova_summary.csv"

app = FastAPI(title="CL3O Visualization API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

repo = RunRepository()


def _json(payload) -> JSONResponse:
    return JSONResponse(content=to_jsonable(payload))


def _snapshot(run_id: str, k: int):
    try:
        return repo.get_snapshot(run_id, k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def _wing_typed(run_id: str):
    '''Typed WingData for the run, or 404 when no wing DB matches.'''
    wing = repo.get_wing_data(run_id)
    if wing is None:
        raise HTTPException(status_code=404, detail="wing data not found")
    return wing


def _wing_raw(run_id: str):
    '''Raw wing-DB dict for the run, or 404 when no wing DB matches.'''
    wing = repo.get_wing(run_id)
    if wing is None:
        raise HTTPException(
            status_code=404, detail="no wing database found in data/wings"
        )
    return wing


_VALID_ENDS = ("A", "B", "avg")


def valid_end(end: str = Query("avg")) -> str:
    '''Validate the element-end selector shared by the stress routes.'''
    if end not in _VALID_ENDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"[CL3O] Invalid 'end' value '{end}'. "
                f"Expected one of {_VALID_ENDS}."
            ),
        )
    return end


# ------------------------------------------------------------------
# Meta
# ------------------------------------------------------------------

@app.get("/api/health")
def health():
    return _json({"status": "ok"})


@app.get("/api/runs")
def list_runs():
    return _json(repo.list_runs())


@app.get("/api/runs/{run_id}/manifest")
def manifest(run_id: str):
    try:
        return _json(repo.get_manifest(run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/runs/{run_id}/planform")
def planform(run_id: str):
    return _json(extract.planform(_wing_raw(run_id)))


@app.get("/api/runs/{run_id}/search")
def search_space(run_id: str):
    try:
        manifest = repo.get_manifest(run_id)
        snaps    = repo.distinct_snapshots(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _json(extract.search_space(snaps, manifest))


# ------------------------------------------------------------------
# Per-generation views
# ------------------------------------------------------------------

@app.get("/api/runs/{run_id}/gen/{k}/info")
def info(run_id: str, k: int):
    return _json(extract.info(_snapshot(run_id, k)))


@app.get("/api/runs/{run_id}/gen/{k}/section/{station}")
def section(run_id: str, k: int, station: int):
    return _json(extract.section(_snapshot(run_id, k), station))


@app.get("/api/runs/{run_id}/gen/{k}/mesh")
def mesh(
    run_id: str,
    k: int,
    lc: int = Query(0),
    deformed: bool = Query(False),
    scale: float = Query(1.0),
):
    return _json(extract.mesh(_snapshot(run_id, k), lc=lc, deformed=deformed, scale=scale))


@app.get("/api/runs/{run_id}/gen/{k}/stress")
def stress(
    run_id: str,
    k: int,
    lc: int = Query(0),
    end: str = Depends(valid_end),
):
    return _json(extract.stress(_snapshot(run_id, k), lc=lc, end=end))


@app.get("/api/runs/{run_id}/gen/{k}/forces")
def forces(run_id: str, k: int, lc: int = Query(0)):
    return _json(extract.forces(_snapshot(run_id, k), lc=lc))


@app.get("/api/runs/{run_id}/gen/{k}/stress3d")
def stress3d(run_id: str, k: int, lc: int = Query(0), end: str = Depends(valid_end)):
    rt = _snapshot(run_id, k)
    return _json(surface.build_stress_surface(rt, _wing_typed(run_id), lc=lc, end=end))


@app.get("/api/runs/{run_id}/gen/{k}/tsw3d")
def tsw3d(run_id: str, k: int, lc: int = Query(0), end: str = Depends(valid_end)):
    rt = _snapshot(run_id, k)
    return _json(surface.build_tsw_surface(rt, _wing_typed(run_id), lc=lc, end=end))


@app.get("/api/runs/{run_id}/gen/{k}/geometry")
def geometry(
    run_id: str,
    k: int,
    deformed: bool = Query(False),
    lc: int = Query(0),
    scale: float = Query(1.0),
):
    rt = _snapshot(run_id, k)
    wing = _wing_typed(run_id)
    afl = repo.get_airfoil(run_id)
    if afl is None:
        raise HTTPException(status_code=404, detail="airfoil data not found")
    scene = surface.build_scene(rt, wing, afl, lc=lc, scale=scale, deform=deformed)
    scene["laminate_catalog"] = repo.get_laminate_catalog(run_id)
    return _json(scene)


@app.get("/api/sensitivity")
def sensitivity():
    if not _ANOVA_RESULTS.is_file():
        return _json({"available": False})
    groups = []
    with open(_ANOVA_RESULTS, newline="") as fh:
        for row in csv.DictReader(fh):
            groups.append({
                "group":    row["group"],
                "eta_sq":   float(row["eta_sq"]),
                "mean_f":   float(row["mean_f"]),
                "std_f":    float(row["std_f"]),
                "min_f":    float(row["min_f"]),
                "max_f":    float(row["max_f"]),
            })
    summary = None
    if _ANOVA_SUMMARY.is_file():
        with open(_ANOVA_SUMMARY, newline="") as fh:
            row = next(csv.DictReader(fh))
            summary = {
                "F_stat":     float(row["F_stat"]),
                "p_value":    float(row["p_value"]),
                "df_between": int(float(row["df_between"])),
                "df_within":  int(float(row["df_within"])),
            }
    return _json({"available": True, "groups": groups, "summary": summary})


class _SnapBody(_BaseModel):
    run_id: str
    gen:    int
    view:   str
    data:   str   # base64-encoded PNG

@app.post("/api/snaps")
def save_snap(body: _SnapBody):
    snaps_dir = OUTPUTS_DIR / "snaps"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"{body.run_id}_gen{body.gen:04d}_{body.view}_{ts}.png"
    path  = snaps_dir / fname
    raw   = base64.b64decode(body.data)
    path.write_bytes(raw)
    return _json({"path": str(path)})


# ------------------------------------------------------------------
# Static SPA - serve the built frontend (production / desktop mode)
# ------------------------------------------------------------------
# Mounted LAST so the /api/* routes above always match first. Only
# active once the frontend is built (npm run build -> frontend/dist),
# letting one server serve both the API and the UI on a single port.
_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
