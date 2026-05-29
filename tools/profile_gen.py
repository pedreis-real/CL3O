'''
Profile one DE generation (NP=8, k_max=1) to identify the heaviest hotspots.

Usage:
    python -m tools.profile_gen

Output: prints top-30 cumtime entries, then per-hotspot breakdown.
'''
from __future__ import annotations

import cProfile
import io
import pstats
from pathlib import Path
import sys

# --- project import ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cl3o.main import (
    RunCLEO, _resolve_db_specs, DatabaseSpec, MainHelpers,
)
from cl3o.Constants import DE_HYPERPAR
from cl3o.paths import (
    DATA_DIR,
)
from cl3o.geometry.wing        import WingData
from cl3o.geometry.airfoil     import AirfoilData
from cl3o.materials.laminate import LaminateData
from cl3o.utils.oppoints       import OppData
from cl3o.fea.loads.load_mapper import ExLoadsData, InLoadsData

_MAT_DIR = DATA_DIR / "materials"
_AFL_DIR = DATA_DIR / "airfoils"
_WNG_DIR = DATA_DIR / "wings"
_OPP_DIR = DATA_DIR / "oppoints"
_LDS_DIR = DATA_DIR / "loads"

_AIRCRAFT = "DA62"
_NP       = 8
_K_MAX    = 1          # profile exactly 1 generation


def _build_specs() -> list:
    materials = sorted(
        f.stem.removesuffix("_LaminateData")
        for f in _MAT_DIR.glob("MAT_*_LaminateData.json")
    )
    specs: list = []
    specs.append(DatabaseSpec(WingData,    _WNG_DIR, _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(AirfoilData, _AFL_DIR, "wortmannfx63137"))
    for mat in materials:
        specs.append(DatabaseSpec(LaminateData, _MAT_DIR, mat))
    specs.append(DatabaseSpec(OppData,     _OPP_DIR, _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(ExLoadsData, _LDS_DIR, _AIRCRAFT.lower()))
    specs.append(DatabaseSpec(InLoadsData, _LDS_DIR, _AIRCRAFT.lower()))
    return _resolve_db_specs(specs)


def main() -> None:
    print(f"\n[profile_gen] Building RunCLEO (NP={_NP}, k_max={_K_MAX}) ...")
    db_specs = _build_specs()
    MainHelpers.verify_missing_database(db_specs)

    runner = RunCLEO(
        aircraft_name    = _AIRCRAFT,
        opt_name         = "profile",
        db_specs         = db_specs,
        pipeline_logging = False,
        enable_logging   = False,
        de_hyperpar      = {**DE_HYPERPAR, "NP": _NP, "k_max": _K_MAX},
    )

    print(f"[profile_gen] Profiling {_NP} candidates × {_K_MAX} generation(s)...")

    pr = cProfile.Profile()
    pr.enable()
    runner.run_optimization(out_dir=None)
    pr.disable()

    # ---- summary ----
    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(40)
    report = buf.getvalue()
    print(report)

    # ---- per-hotspot grep ----
    hotspots = [
        ("mesh_builder",   "MeshBuilder / assemble"),
        ("tsw_failure",    "TsaiWuFailure / _evaluate"),
        ("section_builder","SectionBuilder"),
        ("static_analysis","LinearStaticSolver"),
        ("beam_element",   "BeamElement"),
    ]
    print("\n=== Hotspot grep (tottime lines) ===")
    for fname, label in hotspots:
        buf2 = io.StringIO()
        ps2  = pstats.Stats(pr, stream=buf2).sort_stats("tottime")
        ps2.print_stats(fname)
        lines = [l for l in buf2.getvalue().splitlines() if fname in l.lower()]
        total = sum(
            float(l.split()[1]) for l in lines if len(l.split()) > 1
        )
        print(f"  {label:35s}: {total:.3f} s total (tottime)")
        for l in lines[:5]:
            print(f"    {l.strip()}")


if __name__ == "__main__":
    main()
