'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
UI Backend Tests Module.

First test coverage for the FastAPI visualization backend: the shared
route guards (end-selector validation, bounded run cache), the spar-strip
defensive lookup, and an end-to-end smoke over a real archived DE run
(built by the de_history fixture) through the repository / extract /
surface layers.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from types import SimpleNamespace

import numpy as np
import pytest

# Skip the whole module when the [ui] extra (FastAPI) is not installed - the
# fast unit-test CI job installs only [dev], so these backend tests are
# collected only where fastapi is present.
pytest.importorskip("fastapi")
from fastapi import HTTPException

# ================ Module imports ================
from cl3o.ui.backend import app as ui_app
from cl3o.ui.backend import surface
from cl3o.ui.backend import extract
from cl3o.ui.backend.repository import RunRepository
from cl3o.utils.lru_cache import LRUCache


# ================================================================================
# Unit - end-selector validation
# ================================================================================

@pytest.mark.parametrize("end", ["A", "B", "avg"])
def test_valid_end_accepts_known_selectors(end):
    '''valid_end returns the value unchanged for the supported selectors.'''
    assert ui_app.valid_end(end) == end


@pytest.mark.parametrize("bad", ["Z", "", "average", "a"])
def test_valid_end_rejects_unknown_selectors(bad):
    '''valid_end raises HTTP 422 for any unsupported end value.'''
    with pytest.raises(HTTPException) as exc:
        ui_app.valid_end(bad)
    assert exc.value.status_code == 422
    assert "[CL3O]" in str(exc.value.detail)


# ================================================================================
# Unit - bounded run cache
# ================================================================================

def test_run_full_cache_is_bounded_lru():
    '''The module-level run cache is an LRUCache with a finite ceiling.'''
    from cl3o.ui.backend import repository
    assert isinstance(repository._run_full_cache, LRUCache)
    assert repository._run_full_cache.maxsize == repository._RUN_CACHE_MAXSIZE
    assert repository._RUN_CACHE_MAXSIZE > 0


def test_lru_cache_evicts_least_recently_used():
    '''Inserting past maxsize drops the oldest, recency-aware, entry.'''
    cache = LRUCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache.get("a")          # touch 'a' so 'b' becomes least-recent
    cache["c"] = 3          # evicts 'b'
    assert "a" in cache and "c" in cache
    assert "b" not in cache


# ================================================================================
# Unit - spar-strip defensive lookup
# ================================================================================

def test_spar_strip_missing_label_raises():
    '''A missing spar web label yields a clear [CL3O] error, not StopIteration.'''
    gd = SimpleNamespace(
        T1=[{"label": "seg1", "pts": [[0.0, 0.0], [1.0, 0.0]]}],
        C=np.zeros(3),
        chord=1.0,
    )
    rt = SimpleNamespace(sections=SimpleNamespace(sec_data=[gd]))
    with pytest.raises(ValueError) as exc:
        surface._spar_strip(
            rt, left=[0], seg_label="seg_absent", dmat=None,
            LE=np.zeros((1, 3)), chord=1.0, twist=0.0,
            lrow=[0], scale=1.0, lc=0,
        )
    msg = str(exc.value)
    assert "[CL3O]" in msg
    assert "seg_absent" in msg


# ================================================================================
# Integration - end-to-end smoke over a real archived run
# ================================================================================

@pytest.mark.slow
@pytest.mark.integration
def test_backend_smoke_over_archived_run(de_history):
    '''Drive repository -> extract -> surface against a freshly archived run.'''
    repo = RunRepository()
    runs = repo.list_runs()
    assert runs, "de_history fixture should have archived at least one run"
    run_id = next(
        (r["run_id"] for r in runs if "pytestsessionhist" in r["run_id"].lower()),
        runs[0]["run_id"],
    )

    manifest = repo.get_manifest(run_id)
    k = manifest["best_gen"]
    rt = repo.get_snapshot(run_id, k)

    # Scalar / series extractors.
    info = extract.info(rt)
    assert "fitness" in info and "mass" in info
    forces = extract.forces(rt, lc=0)
    assert len(forces["span"]) == forces["n_loadcases"] or forces["span"]
    stress = extract.stress(rt, lc=0, end="avg")
    assert stress["n_panels"] >= 1

    # Search-space PCA trajectory (refactored extract helpers).
    space = extract.search_space(repo.distinct_snapshots(run_id), manifest)
    assert len(space["explained_variance"]) == 3
    assert len(space["x"]) == space["n_distinct"]

    # 3-D surface builders need the typed wing DB.
    wing = repo.get_wing_data(run_id)
    assert wing is not None
    stress3d = surface.build_stress_surface(rt, wing, lc=0, end="avg")
    assert stress3d["vertices"].shape[1] == 3
    afl = repo.get_airfoil(run_id)
    scene = surface.build_scene(rt, wing, afl, lc=0, scale=1.0, deform=True)
    assert "surface" in scene and "front_spar" in scene
