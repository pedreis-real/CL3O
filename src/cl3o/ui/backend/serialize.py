'''
================================================================================
CL3O Visualization UI - JSON serialization helpers.

The archived data carries numpy arrays and non-finite floats (the manifest
even stores `Infinity`). Standard JSON parsers in the browser reject
NaN/Infinity, so every payload is funnelled through `to_jsonable`, which
maps numpy -> native python and every non-finite float -> None (null).

@ CL3O Authors - MIT License
================================================================================
'''

import math

import numpy as np


def to_jsonable(obj):
    '''Recursively convert `obj` into JSON-safe native python.

    - numpy arrays / scalars  -> lists / python scalars
    - NaN, +-inf (any source) -> None
    - dict keys               -> str
    Unknown objects fall back to their repr so a stray type can never
    crash a response.
    '''
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)
