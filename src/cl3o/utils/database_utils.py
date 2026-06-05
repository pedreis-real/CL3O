'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Database Discovery Utilities Module.

Small filesystem-discovery helpers shared by the test fixtures, the runtime
validation scripts, and the standalone tools under tools/. Exposes the
curated-laminate catalogue glob so the identical discovery idiom is no longer
re-implemented at every call site.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path

# ================ Default Database Paths ================
from cl3o.paths import MATERIALS_DIR as _DFLT_MAT_DIR

# ================ Module imports ================


# ================================================================================
# PUBLIC API - Laminate catalogue discovery
# ================================================================================

def discover_laminates(
    mat_dir: str | Path = _DFLT_MAT_DIR,
) -> list[str]:
    '''
    Discover the curated laminate catalogue on disk.

    Globs MAT_*_LaminateData.json under mat_dir and returns the sorted
    laminate names (each file stem with the _LaminateData suffix stripped).
    The underscore prefix selects the curated catalogue and skips legacy
    MAT{int} test laminates (no underscore).

    Args:
        mat_dir: Directory holding the curated laminate JSON files.
            Defaults to the canonical data/materials directory.

    Returns:
        Sorted list of laminate name strings, e.g. ["MAT_AS4_8552", ...].
    '''
    return sorted(
        f.stem.removesuffix("_LaminateData")
        for f in Path(mat_dir).glob("MAT_*_LaminateData.json")
    )
