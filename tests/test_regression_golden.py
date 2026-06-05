'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Golden-Output Regression Harness.

Characterization tests that pin the current numerical behaviour of the core
pipeline so that the staged refactor can prove it changed nothing. Two
baselines are captured:

    single_eval : a full RuntimeData snapshot from one deterministic
                  evaluation (X = centre of the DE bounds, via the shared
                  ``runtime`` fixture).
    de_history  : the HistoryData of a tiny seeded DE run (``de_history``).

Each object is reduced to a flat, JSON-native *fingerprint*: every numpy
array becomes a handful of deterministic reductions (shape, finite sum,
sum-of-squares, abs-max, non-finite count) and every scalar its value, keyed
by dotted field path. Fingerprints are compared against the committed
baselines under ``tests/golden/`` with a tight tolerance.

Regenerate the baselines deliberately (e.g. after an intentional Phase 3/4
behaviour change) with:

    CL3O_REGEN_GOLDEN=1 python -m pytest tests/test_regression_golden.py \
        -m "slow or integration"

Override the comparison tolerance for the looser performance phases with
CL3O_GOLDEN_RTOL / CL3O_GOLDEN_ATOL.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import os
import json
import math
from dataclasses import is_dataclass, fields
from pathlib import Path

import numpy as np
import pytest

# ================ Module imports ================
# (the heavy ``runtime`` / ``de_history`` fixtures live in tests/conftest.py)


# ================ Test configuration ================
_GOLDEN_DIR = Path(__file__).parent / "golden"

_RTOL  = float(os.environ.get("CL3O_GOLDEN_RTOL", "1e-9"))
_ATOL  = float(os.environ.get("CL3O_GOLDEN_ATOL", "1e-12"))
_REGEN = os.environ.get("CL3O_REGEN_GOLDEN", "").strip().lower() not in ("", "0", "false", "no")

_MAX_DEPTH    = 60      # guard against pathological nesting / cycles
_MAX_REPORTED = 40      # cap the number of diffs surfaced in a failure


# ================================================================================
# Internal helper - fingerprinting
# ================================================================================

def _enc_num(x: float) -> float | str:
    '''Encode a scalar as a JSON-native value; non-finite -> string token.'''
    xf = float(x)
    if math.isnan(xf):
        return "nan"
    if math.isinf(xf):
        return "inf" if xf > 0.0 else "-inf"
    return xf


def _fingerprint_array(arr: np.ndarray, path: str, out: dict) -> None:
    '''Reduce one numeric array to deterministic, tolerance-comparable stats.'''
    a = np.asarray(arr)
    if a.dtype == object:
        # Ragged / object array: recurse element-wise so nested arrays count.
        for i, el in enumerate(a.ravel()):
            _fingerprint(el, f"{path}[{i}]", out, 0, set())
        return
    af = a.astype(float, copy=False)
    finite = np.isfinite(af)
    out[f"{path}::shape"] = ",".join(str(d) for d in a.shape)
    out[f"{path}::nfin"]  = int((~finite).sum())
    if finite.any():
        vals = af[finite]
        out[f"{path}::sum"]    = _enc_num(np.sum(vals))
        out[f"{path}::sumsq"]  = _enc_num(np.sum(vals * vals))
        out[f"{path}::absmax"] = _enc_num(np.max(np.abs(vals)))
    else:
        out[f"{path}::sum"]    = 0.0
        out[f"{path}::sumsq"]  = 0.0
        out[f"{path}::absmax"] = 0.0


def _fingerprint(obj, path: str, out: dict, depth: int, seen: set) -> dict:
    '''
    Recursively flatten an object tree into ``{dotted_path: json_value}``.

    Handled leaves: numpy arrays (reduced via _fingerprint_array), python /
    numpy scalars, and bools. Dataclasses, dicts, lists and tuples are walked.
    Strings and None are skipped (structural, not numerical). Unknown objects
    (loggers, scipy sparse, etc.) are recorded as a skipped marker so that a
    container appearing/disappearing is still visible in the diff.
    '''
    if obj is None or isinstance(obj, str):
        return out
    if depth > _MAX_DEPTH:
        out[f"{path}::truncated"] = True
        return out

    if isinstance(obj, (bool, np.bool_)):
        out[path] = bool(obj)
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        out[path] = _enc_num(obj)
    elif isinstance(obj, np.ndarray):
        _fingerprint_array(obj, path, out)
    elif is_dataclass(obj) and not isinstance(obj, type):
        if id(obj) in seen:
            return out
        seen.add(id(obj))
        for f in fields(obj):
            child = f"{path}.{f.name}" if path else f.name
            _fingerprint(getattr(obj, f.name), child, out, depth + 1, seen)
    elif isinstance(obj, dict):
        if id(obj) in seen:
            return out
        seen.add(id(obj))
        for k in sorted(obj, key=str):
            _fingerprint(obj[k], f"{path}.{k}", out, depth + 1, seen)
    elif isinstance(obj, (list, tuple)):
        if id(obj) in seen:
            return out
        seen.add(id(obj))
        for i, el in enumerate(obj):
            _fingerprint(el, f"{path}[{i}]", out, depth + 1, seen)
    elif hasattr(obj, "toarray"):
        # scipy sparse and similar: densify then reduce.
        _fingerprint_array(obj.toarray(), path, out)
    else:
        out[f"{path}::skipped"] = type(obj).__name__
    return out


def fingerprint(obj) -> dict:
    '''Public entry point: flat fingerprint dict for an object tree.'''
    return _fingerprint(obj, "", {}, 0, set())


# ================================================================================
# Internal helper - comparison
# ================================================================================

def _values_match(cur, base, rtol: float, atol: float) -> bool:
    '''Compare two fingerprint leaves with type-aware tolerance.'''
    if isinstance(base, bool) or isinstance(cur, bool):
        return bool(cur) == bool(base)
    cur_num  = isinstance(cur, (int, float))
    base_num = isinstance(base, (int, float))
    if cur_num and base_num:
        return math.isclose(float(cur), float(base), rel_tol=rtol, abs_tol=atol)
    # strings (shape, tokens, skipped markers) compare exactly
    return cur == base


def compare_fingerprints(
    current : dict,
    baseline: dict,
    rtol    : float = _RTOL,
    atol    : float = _ATOL,
) -> list[str]:
    '''
    Return a list of human-readable difference strings (empty when equal).

    Detects value drift, missing keys and new keys so that both numerical
    changes and structural changes (a container vanishing) are caught.
    '''
    diffs: list[str] = []
    for key in sorted(set(current) | set(baseline)):
        if key not in baseline:
            diffs.append(f"+ NEW   {key} = {current[key]!r}")
        elif key not in current:
            diffs.append(f"- GONE  {key} (was {baseline[key]!r})")
        elif not _values_match(current[key], baseline[key], rtol, atol):
            diffs.append(f"~ DIFF  {key}: got {current[key]!r}, expected {baseline[key]!r}")
    return diffs


# ================================================================================
# Internal helper - baseline IO + assertion
# ================================================================================

def _baseline_path(name: str) -> Path:
    return _GOLDEN_DIR / f"{name}.json"


def _assert_golden(name: str, obj) -> None:
    '''Regenerate or compare the named baseline against ``obj``.'''
    fp   = fingerprint(obj)
    path = _baseline_path(name)

    if _REGEN:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fp, indent=2, sort_keys=True), encoding="utf-8")
        pytest.skip(f"[CL3O] Regenerated golden baseline '{name}' ({len(fp)} keys).")

    if not path.exists():
        pytest.fail(
            f"[CL3O] Missing golden baseline.\n"
            f"| name : {name}\n"
            f"| path : {path}\n"
            f"Generate it once with CL3O_REGEN_GOLDEN=1 python -m pytest "
            f"tests/test_regression_golden.py -m 'slow or integration'."
        )

    baseline = json.loads(path.read_text(encoding="utf-8"))
    diffs = compare_fingerprints(fp, baseline)
    if diffs:
        shown = "\n".join(diffs[:_MAX_REPORTED])
        extra = "" if len(diffs) <= _MAX_REPORTED else f"\n... and {len(diffs) - _MAX_REPORTED} more"
        pytest.fail(
            f"[CL3O] Golden regression '{name}': {len(diffs)} field(s) drifted "
            f"(rtol={_RTOL}, atol={_ATOL}).\n{shown}{extra}\n"
            f"If this change is intentional, re-pin with CL3O_REGEN_GOLDEN=1."
        )


# ================================================================================
# PUBLIC API - Regression tests
# ================================================================================

@pytest.mark.slow
@pytest.mark.integration
def test_golden_single_eval(runtime):
    '''Pin the full RuntimeData of one deterministic pipeline evaluation.'''
    _assert_golden("single_eval", runtime)


@pytest.mark.slow
@pytest.mark.integration
def test_golden_de_history(de_history):
    '''Pin the HistoryData of the tiny seeded end-to-end DE run.'''
    _assert_golden("de_history", de_history)


# ================================================================================
# PUBLIC API - Fast self-test (no database, proves the harness catches drift)
# ================================================================================

def test_fingerprint_is_deterministic():
    '''The same object fingerprints identically on repeated calls.'''
    import types

    obj = types.SimpleNamespace(
        a=np.array([1.0, 2.0, 3.0]),
        b={"x": 4, "y": [np.array([[1.0, 2.0], [3.0, 4.0]]), True]},
        c=float("inf"),
    )
    assert fingerprint(obj) == fingerprint(obj)


def test_compare_detects_drift():
    '''A small numeric perturbation must be reported by the comparator.'''
    base = {"k::sum": 10.0, "k::shape": "3", "flag": True}
    same = dict(base)
    assert compare_fingerprints(same, base) == []

    drifted = dict(base, **{"k::sum": 10.0 + 1e-3})
    assert compare_fingerprints(drifted, base), "numeric drift went undetected"

    reshaped = dict(base, **{"k::shape": "4"})
    assert compare_fingerprints(reshaped, base), "shape change went undetected"

    flipped = dict(base, **{"flag": False})
    assert compare_fingerprints(flipped, base), "bool flip went undetected"

    assert compare_fingerprints({"k::sum": 5.0}, base), "missing key went undetected"


def test_compare_respects_tolerance():
    '''Sub-tolerance noise passes; supra-tolerance drift fails.'''
    base = {"v": 1.0}
    assert compare_fingerprints({"v": 1.0 + 1e-12}, base) == []
    assert compare_fingerprints({"v": 1.0 + 1e-3}, base)
