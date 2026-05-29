'''
================================================================================
CL3O Visualization UI - Backend pathing.

Thin re-export of the project path constants (see ``cl3o.paths``) under the
names the backend modules expect. Pickle resolves the dataclasses referenced
inside the archived RuntimeData snapshots (``cl3o.optimization.fobjective``,
``cl3o.geometry.*``, ``cl3o.fea.*`` ...) via the installed ``cl3o`` package, so
no sys.path manipulation is needed.

@ CL3O Authors - MIT License
================================================================================
'''

from cl3o.paths import ROOT_DIR, OUTPUTS_DIR, WINGS_DIR, MATERIALS_DIR

__all__ = ["ROOT_DIR", "OUTPUTS_DIR", "WINGS_DIR", "MATERIALS_DIR"]
