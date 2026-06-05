'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Airfoil Helper Unit Tests.

Fast, database-free unit tests for the pure AirfoilHelper geometry routines
extracted from the Airfoil builder: surface splitting (both Selig
orientations), the NACA 4-digit thickness polynomial, and the camber-normal
projection invariants.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.geometry.airfoil import AirfoilHelper


# ================================================================================
# AirfoilHelper.split_upper_lower - surface orientation
# ================================================================================

def test_split_upper_lower_le_to_te_first_branch():
    # Selig order: TE -> LE on top, then LE -> TE on bottom (y[le+1] < 0).
    x = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
    y = np.array([0.0, 0.1, 0.0, -0.1, 0.0])
    x_u, y_u, x_l, y_l = AirfoilHelper.split_upper_lower(x, y)

    # Both surfaces run leading-edge -> trailing-edge.
    assert x_u[0] == pytest.approx(0.0)
    assert x_u[-1] == pytest.approx(1.0)
    assert np.all(np.diff(x_u) > 0)
    assert x_l[0] == pytest.approx(0.0)
    assert x_l[-1] == pytest.approx(1.0)
    # Upper sits above the chord, lower below.
    assert np.all(y_u >= 0.0)
    assert np.all(y_l <= 0.0)


def test_split_upper_lower_le_to_te_second_branch():
    # Reversed order: LE -> TE on top first, then TE -> LE on bottom.
    x = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    y = np.array([0.0, 0.1, 0.0, -0.1, 0.0])
    x_u, y_u, x_l, y_l = AirfoilHelper.split_upper_lower(x, y)

    assert x_u[0] == pytest.approx(0.0)
    assert x_u[-1] == pytest.approx(1.0)
    assert x_l[0] == pytest.approx(0.0)
    assert x_l[-1] == pytest.approx(1.0)
    assert np.all(y_u >= 0.0)
    assert np.all(y_l <= 0.0)


# ================================================================================
# AirfoilHelper.naca_thickness - 4-digit thickness polynomial
# ================================================================================

def test_naca_thickness_zero_at_leading_edge():
    assert AirfoilHelper.naca_thickness(np.array([0.0]), 0.12)[0] == pytest.approx(0.0)


def test_naca_thickness_open_trailing_edge_value():
    # yt(1, t) = 5t * 0.0021 = 0.0105 t (the open-TE gap of the 4-digit form).
    yt_te = AirfoilHelper.naca_thickness(np.array([1.0]), 0.12)[0]
    assert yt_te == pytest.approx(0.00126, abs=1e-5)


def test_naca_thickness_is_linear_in_t():
    x = np.linspace(0.0, 1.0, 25)
    yt1 = AirfoilHelper.naca_thickness(x, 0.12)
    yt2 = AirfoilHelper.naca_thickness(x, 0.24)
    assert np.allclose(yt2, 2.0 * yt1)


def test_naca_thickness_max_is_half_thickness_ratio():
    # Peak half-thickness of a 12% foil is ~0.06 (full thickness ~= t).
    x = np.linspace(0.0, 1.0, 1001)
    assert AirfoilHelper.naca_thickness(x, 0.12).max() == pytest.approx(0.06, abs=1e-3)


# ================================================================================
# AirfoilHelper.apply_camber_normal - normal projection invariants
# ================================================================================

def test_apply_camber_normal_zero_camber_is_symmetric():
    x  = np.linspace(0.0, 1.0, 11)
    yt = AirfoilHelper.naca_thickness(x, 0.12)
    zero = np.zeros_like(x)
    x_u, y_u, x_l, y_l = AirfoilHelper.apply_camber_normal(x, zero, zero, yt)

    assert np.allclose(x_u, x)
    assert np.allclose(x_l, x)
    assert np.allclose(y_u, yt)
    assert np.allclose(y_l, -yt)


def test_apply_camber_normal_keeps_mean_line_and_offset():
    x    = np.linspace(0.0, 1.0, 11)
    y_c  = 0.05 * np.sin(np.pi * x)         # arbitrary smooth camber
    dy_c = 0.05 * np.pi * np.cos(np.pi * x) # its analytic slope
    yt   = AirfoilHelper.naca_thickness(x, 0.12)
    x_u, y_u, x_l, y_l = AirfoilHelper.apply_camber_normal(x, y_c, dy_c, yt)

    # Mean of the two surfaces recovers the camber line exactly.
    assert np.allclose(0.5 * (x_u + x_l), x)
    assert np.allclose(0.5 * (y_u + y_l), y_c)
    # Each surface point lies exactly yt away from its camber point.
    dist_u = np.hypot(x_u - x, y_u - y_c)
    assert np.allclose(dist_u, yt)
