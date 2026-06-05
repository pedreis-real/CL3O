'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Shared Utilities Unit Tests.

Fast, database-free unit tests for the previously-untested utility layer:
the bounded LRUCache eviction semantics, the pure math_utils geometry and
coordinate-transform helpers, and the io_utils JSON dataclass round-trip.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from dataclasses import dataclass, field

import numpy as np
import pytest

# ================ Module imports ================
from cl3o.utils.lru_cache import LRUCache
from cl3o.utils import math_utils as mu
from cl3o.utils import io_utils as io


# ================================================================================
# LRUCache - eviction + recency semantics
# ================================================================================

def test_lru_unbounded_keeps_all():
    c = LRUCache(maxsize=0)
    for i in range(5):
        c[i] = i
    assert len(c) == 5
    assert all(c.get(i) == i for i in range(5))


def test_lru_evicts_oldest_when_over_size():
    c = LRUCache(maxsize=3)
    for i in range(4):          # inserting 3 evicts 0 (the oldest)
        c[i] = i
    assert len(c) == 3
    assert c.get(0) is None
    assert c.get(3) == 3


def test_lru_get_marks_most_recent():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert c.get("a") == 1      # touch 'a' -> 'b' becomes the oldest
    c["c"] = 3                  # evicts 'b', not 'a'
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3


def test_lru_update_existing_marks_recent_without_growth():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10                 # update 'a' -> most recent; 'b' now oldest
    assert len(c) == 2
    c["c"] = 3                  # evicts 'b'
    assert c.get("b") is None
    assert c.get("a") == 10


def test_lru_get_returns_default_on_miss():
    c = LRUCache(maxsize=2)
    assert c.get("missing") is None
    assert c.get("missing", -1) == -1


# ================================================================================
# math_utils - polygon geometry
# ================================================================================

# Unit square, counter-clockwise, given open (auto-closed by the helpers).
_SQ_X = np.array([0.0, 1.0, 1.0, 0.0])
_SQ_Y = np.array([0.0, 0.0, 1.0, 1.0])


def test_polygon_area_unit_square():
    assert mu.polygon_area(_SQ_X, _SQ_Y) == pytest.approx(1.0)


def test_polygon_area_is_orientation_independent():
    assert mu.polygon_area(_SQ_X[::-1], _SQ_Y[::-1]) == pytest.approx(1.0)


def test_polygon_centroid_unit_square():
    cx, cy = mu.polygon_centroid(_SQ_X, _SQ_Y)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)


def test_polygon_centroid_zero_area_raises():
    line_x = np.array([0.0, 1.0, 2.0])
    line_y = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        mu.polygon_centroid(line_x, line_y)


def test_polygon_second_moments_unit_square_about_centroid():
    # A unit square about its own centroid: I = b*h^3/12 = 1/12; Ixy = 0.
    Ixx, Iyy, Ixy = mu.polygon_second_moments(_SQ_X, _SQ_Y, about_centroid=True)
    assert Ixx == pytest.approx(1.0 / 12.0)
    assert Iyy == pytest.approx(1.0 / 12.0)
    assert Ixy == pytest.approx(0.0, abs=1e-12)


def test_polygon_perimeter_closed_square():
    x = np.append(_SQ_X, _SQ_X[0])
    y = np.append(_SQ_Y, _SQ_Y[0])
    assert mu.polygon_perimeter(x, y) == pytest.approx(4.0)


def test_swept_double_area_unit_square_loop():
    x = np.append(_SQ_X, _SQ_X[0])
    y = np.append(_SQ_Y, _SQ_Y[0])
    assert mu.swept_double_area(x, y) == pytest.approx(2.0)   # 2 * enclosed area


# ================================================================================
# math_utils - coordinate transforms
# ================================================================================

def test_rotate_points_90deg_about_origin():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    xr, yr = mu.rotate_points(x, y, 90.0)
    # (1, 0) -> (0, 1) and (0, 1) -> (-1, 0)
    assert xr == pytest.approx([0.0, -1.0], abs=1e-12)
    assert yr == pytest.approx([1.0, 0.0], abs=1e-12)


def test_rotate_points_about_pivot_fixes_the_pivot():
    xr, yr = mu.rotate_points(np.array([2.0]), np.array([3.0]), 37.0, cx=2.0, cy=3.0)
    assert xr[0] == pytest.approx(2.0)
    assert yr[0] == pytest.approx(3.0)


def test_scale_points_uniform_and_per_axis():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    xs, ys = mu.scale_points(x, y, 2.0)            # uniform (sy defaults to sx)
    assert np.allclose(xs, [2.0, 4.0])
    assert np.allclose(ys, [6.0, 8.0])
    xs, ys = mu.scale_points(x, y, 2.0, 0.5)       # per-axis
    assert np.allclose(xs, [2.0, 4.0])
    assert np.allclose(ys, [1.5, 2.0])


# ================================================================================
# math_utils - interpolation
# ================================================================================

def test_interp1d_linear_midpoints():
    xk = np.array([0.0, 1.0, 2.0])
    yk = np.array([0.0, 10.0, 20.0])
    out = mu.interp1d_linear(np.array([0.5, 1.5]), xk, yk)
    assert np.allclose(out, [5.0, 15.0])


def test_interp1d_linear_out_of_bounds_is_nan_without_extrapolate():
    xk = np.array([0.0, 1.0])
    yk = np.array([0.0, 1.0])
    out = mu.interp1d_linear(np.array([2.0]), xk, yk, extrapolate=False)
    assert np.isnan(out).all()


def test_interp1d_linear_extrapolate_is_finite():
    xk = np.array([0.0, 1.0])
    yk = np.array([0.0, 1.0])
    out = mu.interp1d_linear(np.array([2.0]), xk, yk, extrapolate=True)
    assert np.isfinite(out).all()


# ================================================================================
# io_utils - JSON dataclass round-trip
# ================================================================================

@dataclass
class _RoundTrip:
    '''Minimal dataclass exercising the io_utils JSON round-trip.'''
    name   : str        = ""
    scalar : float      = 0.0
    vec    : np.ndarray = field(default_factory=lambda: np.zeros(0))


def test_io_json_roundtrip_dataclass(tmp_path):
    obj = _RoundTrip(name="da62", scalar=4.5, vec=np.array([1.0, 2.0, 3.0]))
    path = tmp_path / "round_trip.json"

    io.write_json(obj, path)
    assert path.is_file()

    back = io.read_json(path, _RoundTrip)
    assert back.name == "da62"
    assert back.scalar == pytest.approx(4.5)
    assert np.allclose(np.asarray(back.vec, dtype=float), [1.0, 2.0, 3.0])


def test_io_json_write_creates_parent_dirs(tmp_path):
    obj = _RoundTrip(name="nested")
    path = tmp_path / "a" / "b" / "c.json"
    io.write_json(obj, path)
    assert path.is_file()
