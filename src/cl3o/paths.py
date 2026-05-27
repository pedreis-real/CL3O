'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Project Paths Module.

Single source of truth for the on-disk locations of the JSON databases under
``data/`` and the run artefacts under ``outputs/``. Every module imports the
directory constants from here instead of recomputing ``__file__``-relative
parent chains, so the layout is defined in exactly one place.

The package lives at ``<repo>/src/cl3o``; the data/ and outputs/ trees live at
the repository root (``<repo>``), i.e. three levels up from this file:
``cl3o/ -> src/ -> <repo>``.

@ CL3O Authors - MIT License
================================================================================
'''

from pathlib import Path

# This file: <repo>/src/cl3o/paths.py
#   parents[0] = cl3o/   parents[1] = src/   parents[2] = <repo>/
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR     = ROOT_DIR / "data"
WINGS_DIR    = DATA_DIR / "wings"
AIRFOILS_DIR = DATA_DIR / "airfoils"
MATERIALS_DIR = DATA_DIR / "materials"
PLIES_DIR    = MATERIALS_DIR / "plies"
OPPOINTS_DIR = DATA_DIR / "oppoints"
LOADS_DIR    = DATA_DIR / "loads"

OUTPUTS_DIR  = ROOT_DIR / "outputs"
