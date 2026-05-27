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

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import extract
from . import surface
from .repository import RunRepository
from .serialize import to_jsonable

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
    wing = repo.get_wing(run_id)
    if wing is None:
        raise HTTPException(status_code=404, detail="no wing database found in data/wings")
    return _json(extract.planform(wing))


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
    end: str = Query("avg"),
):
    return _json(extract.stress(_snapshot(run_id, k), lc=lc, end=end))


@app.get("/api/runs/{run_id}/gen/{k}/forces")
def forces(run_id: str, k: int, lc: int = Query(0)):
    return _json(extract.forces(_snapshot(run_id, k), lc=lc))


@app.get("/api/runs/{run_id}/gen/{k}/stress3d")
def stress3d(run_id: str, k: int, lc: int = Query(0), end: str = Query("avg")):
    rt = _snapshot(run_id, k)
    wing = repo.get_wing_data(run_id)
    if wing is None:
        raise HTTPException(status_code=404, detail="wing data not found")
    return _json(surface.build_stress_surface(rt, wing, lc=lc, end=end))


@app.get("/api/runs/{run_id}/gen/{k}/geometry")
def geometry(
    run_id: str,
    k: int,
    deformed: bool = Query(False),
    lc: int = Query(0),
    scale: float = Query(1.0),
):
    rt = _snapshot(run_id, k)
    wing = repo.get_wing_data(run_id)
    afl = repo.get_airfoil(run_id)
    if wing is None or afl is None:
        raise HTTPException(status_code=404, detail="wing/airfoil data not found")
    return _json(surface.build_scene(rt, wing, afl, lc=lc, scale=scale, deform=deformed))


# ------------------------------------------------------------------
# Static SPA - serve the built frontend (production / desktop mode)
# ------------------------------------------------------------------
# Mounted LAST so the /api/* routes above always match first. Only
# active once the frontend is built (npm run build -> frontend/dist),
# letting one server serve both the API and the UI on a single port.
_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
