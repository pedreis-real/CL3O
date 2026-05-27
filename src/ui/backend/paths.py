'''
================================================================================
CL3O Visualization UI - Backend pathing.

Resolves project directories and puts `src/` on sys.path so that
`pickle.load` can import the dataclasses referenced inside the archived
RuntimeData snapshots (optimization.fobjective.RuntimeData, geometry.*,
fea.* ...). Importing this module is a prerequisite for unpickling.

@ CL3O Authors - MIT License
================================================================================
'''

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent      # src/ui/backend/
SRC_DIR  = _HERE.parents[1]                   # src/
ROOT_DIR = _HERE.parents[2]                   # project root (CWSS/)

OUTPUTS_DIR = ROOT_DIR / "outputs"
WINGS_DIR   = ROOT_DIR / "data" / "wings"

# Pickle resolves classes via their module path; the snapshots were written
# with `src/` on sys.path, so we mirror that here before any pickle.load.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
