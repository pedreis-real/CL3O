'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Math Utilities Module.

Domain-agnostic numerical and geometric operations on np.ndarray inputs,
ensuring maximum reusability across the geometry, structural idealization,
and stress recovery modules. Does not reference any CL3O concepts.

Functions organized by category:
  1. Interpolation         - tabulated curve evaluation and resampling
  2. Numerical integration - area under curve, closed-contour integrals
  3. Polygon geometry      - area, centroid, second moments, perimeter
  4. Coordinate transforms - rotation, translation, scaling
  5. Curve processing      - arc-length operations and surface splitting

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
from scipy import interpolate as sci_interp

# ================ Global variables ================
_TOL = 1e-12


def _is_close(a: float, b: float) -> bool:
    '''
    Scalar equivalent of np.isclose with its default tolerances.

    Replicates |a - b| <= atol + rtol * |b| (atol=1e-8, rtol=1e-5) using
    plain Python floats, avoiding the heavy array machinery np.isclose
    invokes on scalar inputs (a hot-path overhead in the geometry build).
    '''
    return abs(a - b) <= 1e-8 + 1e-5 * abs(b)


# ================================================================================
# 1. Interpolation
# ================================================================================

def interp1d_linear(
    x_query    : np.ndarray,
    x_known    : np.ndarray,
    y_known    : np.ndarray,
    extrapolate: bool = False,
) -> np.ndarray:
    '''
    Linear 1-D interpolation over tabulated data.

    Args:
        x_query:     Points at which to evaluate the function.
        x_known:     Known abscissae (must be monotonically increasing).
        y_known:     Corresponding ordinates.
        extrapolate: If True, linearly extrapolates outside the domain.
            If False, returns NaN at out-of-bounds points.

    Returns:
        Array with the same shape as x_query.
    '''
    fill = (y_known[0], y_known[-1]) if extrapolate else np.nan
    f = sci_interp.interp1d(
        x_known, y_known,
        kind       = 'linear',
        bounds_error = False,
        fill_value = fill,
    )
    return f(np.asarray(x_query))


def interp1d_cubic(
    x_query: np.ndarray,
    x_known: np.ndarray,
    y_known: np.ndarray,
) -> np.ndarray:
    '''
    Cubic spline interpolation.

    Preferred over linear when the curve is smooth (e.g., airfoil
    thickness or chord distribution). Does not extrapolate outside
    the domain.

    Args:
        x_query: Points at which to evaluate the spline.
        x_known: Known abscissae (must be monotonically increasing).
        y_known: Corresponding ordinates.

    Returns:
        Array with the same shape as x_query.
    '''
    return sci_interp.CubicSpline(x_known, y_known)(np.asarray(x_query))


def resample_curve(
    x       : np.ndarray,
    y       : np.ndarray,
    n_points: int,
    spacing : str = 'uniform',
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Resample a 2-D curve to n_points with the chosen arc-length spacing.

    Arc-length parameterisation ensures high-curvature regions receive
    more samples than straight regions, independent of the original
    point distribution.

    Args:
        x:        x-coordinates of the original curve.
        y:        y-coordinates of the original curve.
        n_points: Number of points in the resampled curve.
        spacing:  'uniform' for equal arc-length steps, 'cosine' for
            denser sampling near both ends (recommended for airfoils).

    Returns:
        Tuple (x_new, y_new) with resampled coordinates.
    '''
    ds     = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s      = np.concatenate([[0.0], np.cumsum(ds)])
    s_norm = s / s[-1]

    if spacing == 'cosine':
        t = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_points)))
    else:
        t = np.linspace(0.0, 1.0, n_points)

    return interp1d_cubic(t, s_norm, x), interp1d_cubic(t, s_norm, y)


# ================================================================================
# 2. Numerical integration
# ================================================================================

def integrate_piecewise_squared(
    f: np.ndarray,
    x: np.ndarray,
) -> float:
    '''
    Exact integral of a piecewise-linear function squared: int f(x)^2 dx.

    Over each segment [x_i, x_{i+1}] the exact formula is:

        (x_{i+1} - x_i) / 3 * (f_i^2 + f_i*f_{i+1} + f_{i+1}^2)

    Used in the calculation of the mean aerodynamic chord (CMA).

    Args:
        f: Function values at the nodes (e.g., chord distribution).
        x: Node positions (e.g., spanwise stations, must be monotonic).

    Returns:
        Scalar value of the integral over the full domain.
    '''
    dx  = np.diff(x)
    fi  = f[:-1]
    fi1 = f[1:]
    return float(np.sum(dx * (fi**2 + fi * fi1 + fi1**2) / 3.0))


def integrate_closed_contour(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    '''
    Integrates along a closed contour using the Gauss-Green formula.

    Computes the signed area: positive for counter-clockwise contours,
    negative for clockwise.

    Args:
        x: x-coordinates of the contour vertices.
        y: y-coordinates of the contour vertices.

    Returns:
        Signed area enclosed by the contour.
    '''
    if not (_is_close(x[0], x[-1]) and _is_close(y[0], y[-1])):
        x = np.concatenate((x, x[:1]))
        y = np.concatenate((y, y[:1]))
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


# ================================================================================
# 3. Polygon geometry (cross-sectional properties)
# ================================================================================

def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Area of an arbitrary polygon via the Shoelace (Gauss) formula.

    Args:
        x: x-coordinates of the polygon vertices.
        y: y-coordinates of the polygon vertices.

    Returns:
        Positive area value, regardless of contour orientation.
    '''
    return abs(integrate_closed_contour(x, y))


def polygon_centroid(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    '''
    Centroid of a planar polygon.

    Args:
        x: x-coordinates of the polygon vertices.
        y: y-coordinates of the polygon vertices.

    Returns:
        Tuple (x_c, y_c) with the centroid coordinates.
    '''
    if not (_is_close(x[0], x[-1]) and _is_close(y[0], y[-1])):
        x = np.concatenate((x, x[:1]))
        y = np.concatenate((y, y[:1]))

    A = integrate_closed_contour(x, y)
    if abs(A) < 1e-14:
        raise ValueError('Polygon area is zero; centroid is undefined.')

    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    cx    = np.sum((x[:-1] + x[1:]) * cross) / (6.0 * A)
    cy    = np.sum((y[:-1] + y[1:]) * cross) / (6.0 * A)
    return float(cx), float(cy)


def polygon_second_moments(
    x             : np.ndarray,
    y             : np.ndarray,
    about_centroid: bool = True,
) -> tuple[float, float, float]:
    '''
    Second moments of area of a polygon: (Ixx, Iyy, Ixy).

    Args:
        x:              x-coordinates of the polygon vertices.
        y:              y-coordinates of the polygon vertices.
        about_centroid: If True, applies the parallel-axis theorem to
            refer all moments to the centroid. Default is True.

    Returns:
        Tuple (Ixx, Iyy, Ixy) where:
            Ixx = int int y^2 dA  (bending about x-axis),
            Iyy = int int x^2 dA  (bending about y-axis),
            Ixy = int int x*y dA  (product of inertia).
    '''
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    Ixx   = np.sum(
        (y[:-1]**2 + y[:-1] * y[1:] + y[1:]**2) * cross
    ) / 12.0
    Iyy   = np.sum(
        (x[:-1]**2 + x[:-1] * x[1:] + x[1:]**2) * cross
    ) / 12.0
    Ixy   = np.sum(
        (x[:-1]*y[1:] + 2*x[:-1]*y[:-1]
         + 2*x[1:]*y[1:] + x[1:]*y[:-1]) * cross
    ) / 24.0

    if about_centroid:
        A       = integrate_closed_contour(x, y)
        cx, cy  = polygon_centroid(x, y)
        Ixx    -= A * cy**2
        Iyy    -= A * cx**2
        Ixy    -= A * cx * cy

    return float(Ixx), float(Iyy), float(Ixy)


def polygon_perimeter(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Perimeter length of an open or closed polygon.

    Args:
        x: x-coordinates of the vertices.
        y: y-coordinates of the vertices.

    Returns:
        Total perimeter length.
    '''
    return float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))


def swept_double_area(x: np.ndarray, z: np.ndarray) -> float:
    '''
    Twice the area swept by the radius vector from the origin as it
    traces the path (x, z). Used to compute shear-flow moment arms.

    The result equals |sum_i (x_i * z_{i+1} - x_{i+1} * z_i)|, which
    is 2 * (area swept by the line from origin to the path point).

    Args:
        x: x-coordinates, shifted so the moment centre is at the origin.
        z: z-coordinates, shifted so the moment centre is at the origin.

    Returns:
        2 * swept area (always non-negative), in the same units as x^2.
    '''
    return float(np.abs(np.sum(x[:-1] * z[1:] - x[1:] * z[:-1])))


# ================================================================================
# 4. Coordinate transforms
# ================================================================================

def rotate_points(
    x        : np.ndarray,
    y        : np.ndarray,
    angle_deg: float,
    cx       : float = 0.0,
    cy       : float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Rotate points (x, y) about pivot (cx, cy) by angle_deg degrees.

    Args:
        x:         x-coordinates to rotate.
        y:         y-coordinates to rotate.
        angle_deg: Rotation angle in degrees. Positive = counter-clockwise.
        cx:        x-coordinate of the rotation pivot. Default is 0.
        cy:        y-coordinate of the rotation pivot. Default is 0.

    Returns:
        Tuple (x_rot, y_rot) with the rotated coordinates.
    '''
    a      = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    xt, yt = x - cx, y - cy
    x_rot  = ca * xt - sa * yt + cx
    y_rot  = sa * xt + ca * yt + cy
    return x_rot, y_rot


def scale_points(
    x : np.ndarray,
    y : np.ndarray,
    sx: float,
    sy: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Scale coordinates by factors sx and sy.

    Args:
        x:  x-coordinates to scale.
        y:  y-coordinates to scale.
        sx: Scale factor along x.
        sy: Scale factor along y. If None, uses sx for both axes.

    Returns:
        Tuple (x_scaled, y_scaled).
    '''
    sy = sy if sy is not None else sx
    return x * sx, y * sy


def translate_points(
    x : np.ndarray,
    y : np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Translate coordinates by (dx, dy).

    Args:
        x:  x-coordinates to translate.
        y:  y-coordinates to translate.
        dx: Translation along x.
        dy: Translation along y.

    Returns:
        Tuple (x_translated, y_translated).
    '''
    return x + dx, y + dy


def cart2sph(C: np.ndarray) -> tuple[float, float, float]:
    '''
    Spherical-coordinate decomposition of a 3D vector written in the
    canonical cartesian coordinate system.

    Convention (matches the rotation sequence used in BeamElement,
    gamma = rot1(c) @ rot2(b) @ rot3(a), with X-Y as the reference
    plane and Z as the elevation axis):
        - a : azimuth in the X-Y plane, measured from +X toward +Y
        - b : elevation from the X-Y plane toward +Z

    Reference cases:
        C = (+L, 0, 0)  -> a = 0,      b = 0       (beam along +X)
        C = (0, +L, 0)  -> a = +pi/2,  b = 0       (beam along +Y, right wing)
        C = (0, -L, 0)  -> a = -pi/2,  b = 0       (beam along -Y, left wing)
        C = (0, 0, +L)  -> a = 0,      b = +pi/2   (beam along +Z)

    Args:
        C: The vector XYZ [mm].

    Returns:
        a: Azimuth angle    [rad]
        b: Elevation angle  [rad]
        L: Euclidian norm of the vector (3D)    [mm]
    '''
    Cx, Cy, Cz = float(C[0]), float(C[1]), float(C[2])
    L = float(np.sqrt(Cx**2 + Cy**2 + Cz**2))
    a = float(np.arctan2(Cy, Cx))
    b = float(np.arctan2(-Cz, np.sqrt(Cx**2 + Cy**2)))
    return a, b, L


def rot1(ang: float) -> np.ndarray:
    '''
    Rotation matrix along axis 1 (X)

    Args:
        ang : Rotation angle [rad]
    
    Returns:
        rot : Rotation matrix (3,3)
    '''
    c = np.cos(ang)
    s = np.sin(ang)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,   s],
        [0.0,  -s,   c],
    ])


def rot2(ang: float) -> np.ndarray:
    '''
    Rotation matrix along axis 2 (Y)

    Args:
        ang : Rotation angle [rad]
    
    Returns:
        rot : Rotation matrix (3,3)
    '''
    c = np.cos(ang)
    s = np.sin(ang)
    return np.array([
        [   c, 0.0,  -s],
        [ 0.0, 1.0, 0.0],
        [   s, 0.0,   c],
    ])


def rot3(ang: float) -> np.ndarray:
    '''
    Rotation matrix along axis 3 (Z)

    Args:
        ang : Rotation angle [rad]
    
    Returns:
        rot : Rotation matrix (3,3)
    '''
    c = np.cos(ang)
    s = np.sin(ang)
    return np.array([
        [  c,   s,  0.0],
        [ -s,   c,  0.0],
        [0.0, 0.0,  1.0],
    ])


# ================================================================================
# 5. Curve processing
# ================================================================================

def find_leading_edge_index(x: np.ndarray) -> int:
    '''
    Index of the leading edge (minimum x value) in a Selig-format array.

    Args:
        x: x-coordinate array of the full airfoil contour.

    Returns:
        Integer index of the leading-edge point.
    '''
    return int(np.argmin(x))


def split_upper_lower(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split Selig-format coordinates into upper and lower surfaces.

    In the Selig format the contour starts at the trailing edge, runs
    along the upper surface to the leading edge, then returns along the
    lower surface to the trailing edge. The split criterion is the point
    with the minimum x (leading edge).

    Args:
        x: x-coordinate array of the full Selig-format contour.
        y: y-coordinate array of the full Selig-format contour.

    Returns:
        Tuple (x_upper, y_upper, x_lower, y_lower), all oriented LE to
        TE (increasing x), ready for x/c-based interpolation.
    '''
    le      = find_leading_edge_index(x)
    x_upper = x[:le + 1][::-1]
    y_upper = y[:le + 1][::-1]
    x_lower = x[le:]
    y_lower = y[le:]
    return x_upper, y_upper, x_lower, y_lower


def split_curve_at_x(
    x    : np.ndarray,
    z    : np.ndarray,
    x_cut: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    '''
    Split a curve at x = x_cut (monotonically increasing x).

    The cut point is linearly interpolated and included in both halves,
    so no geometry is lost at the split.

    Args:
        x:     x-coordinates (monotonically increasing).
        z:     z-coordinates.
        x_cut: x-coordinate at which to cut.

    Returns:
        ((x_left, z_left), (x_right, z_right)) where left covers x[0]
        to x_cut and right covers x_cut to x[-1], with the cut point
        included as the last/first element of each half respectively.
    '''
    _eps = 1e-12
    idx   = min(max(int(np.searchsorted(x, x_cut)), 1), len(x) - 1)
    frac  = (x_cut - x[idx-1]) / (x[idx] - x[idx-1] + _eps)
    z_cut = z[idx-1] + frac * (z[idx] - z[idx-1])
    x_left  = np.concatenate((x[:idx], (x_cut,)))
    z_left  = np.concatenate((z[:idx], (z_cut,)))
    x_right = np.concatenate(((x_cut,), x[idx:]))
    z_right = np.concatenate(((z_cut,), z[idx:]))
    return (x_left, z_left), (x_right, z_right)


def arc_length(x: np.ndarray, z: np.ndarray) -> float:
    '''Polyline arc length (open path).'''
    dx = np.diff(x)
    dz = np.diff(z)
    return float(np.sum(np.sqrt(dx * dx + dz * dz)))


def arc_length_to_x(
    x       : np.ndarray,
    z       : np.ndarray,
    x_target: float,
) -> float:
    '''
    Arc length from the start of a curve to x = x_target.

    The curve must have monotonically increasing x-coordinates. Uses
    cumulative arc length with linear interpolation at the cut point.

    Args:
        x:        x-coordinates of the curve (monotonically increasing).
        z:        z-coordinates of the curve.
        x_target: Target x-position along the curve.

    Returns:
        Arc length from x[0] to x_target, in the same units as x and z.
    '''
    ds      = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
    s_cumul = np.concatenate([[0.0], np.cumsum(ds)])
    idx     = int(np.clip(np.searchsorted(x, x_target), 1, len(x) - 1))
    s_prv   = s_cumul[idx - 1]
    z_int   = float(np.interp(x_target, x, z))
    dseg    = np.sqrt(
        (x_target - x[idx-1])**2 + (z_int - z[idx-1])**2
    )
    return float(s_prv + dseg)


def camber_and_thickness(
    x_u       : np.ndarray,
    y_u       : np.ndarray,
    x_l       : np.ndarray,
    y_l       : np.ndarray,
    x_stations: np.ndarray | None = None,
    n         : int = 100,
) -> dict:
    '''
    Compute the camber line and thickness distribution of an airfoil.

    The camber line is the arithmetic mean of the upper and lower
    surfaces at each x/c station. Thickness is their difference.

    Args:
        x_u:        Upper surface x-coordinates (LE to TE).
        y_u:        Upper surface y-coordinates (LE to TE).
        x_l:        Lower surface x-coordinates (LE to TE).
        y_l:        Lower surface y-coordinates (LE to TE).
        x_stations: x/c stations for evaluation. If None, a cosine
            distribution with n points is used.
        n:          Number of stations when x_stations is None.

    Returns:
        Dictionary with keys: 'x', 'camber', 'thickness',
        'max_camber', 'x_max_camber', 'max_thickness', 'x_max_thickness'.
    '''
    if x_stations is None:
        x_stations = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))

    yu = interp1d_cubic(x_stations, x_u, y_u)
    yl = interp1d_cubic(x_stations, x_l, y_l)

    camber    = 0.5 * (yu + yl)
    thickness = yu - yl

    i_cam = int(np.argmax(np.abs(camber)))
    i_thk = int(np.argmax(thickness))

    return {
        'x'               : x_stations,
        'camber'          : camber,
        'thickness'       : thickness,
        'max_camber'      : float(camber[i_cam]),
        'x_max_camber'    : float(x_stations[i_cam]),
        'max_thickness'   : float(np.max(thickness)),
        'x_max_thickness' : float(x_stations[i_thk]),
    }


# ================================================================================
# 6. Tensor transformation
# ================================================================================

def mohr_circle(Ixx, Iyy, Ixy):
    C = 0.5 * (Ixx + Iyy)
    R = np.sqrt(0.25 * (Ixx - Iyy)**2 + Ixy**2)
    I1 = C + R
    I2 = C - R
    denom = Ixx - Iyy
    if abs(denom) < _TOL:
        theta_P = float(0.25 * np.pi * np.sign(Ixy))
    else:
        theta_P = float(0.5 * np.arctan2(2.0 * Ixy, denom))
    
    return (I1, I2, theta_P)


# ================================================================================
# 7. Cross-section integration helpers
# ================================================================================

def skin_panel_first_moments(
    x: np.ndarray,
    z: np.ndarray,
    t: float,
) -> tuple[float, float, float]:
    '''
    First moments of area for a single skin panel (piecewise-linear
    thickness t, constant over the panel).

    Args:
        x: x-coordinates of the panel polyline.
        z: z-coordinates of the panel polyline.
        t: Panel thickness [mm].

    Returns:
        Tuple (sum_xdA, sum_zdA, sum_dA) where dA = t * ds.
    '''
    x_mid = 0.5 * (x[:-1] + x[1:])
    z_mid = 0.5 * (z[:-1] + z[1:])
    ds    = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
    dA    = ds * t
    return float(np.dot(x_mid, dA)), float(np.dot(z_mid, dA)), float(np.sum(dA))


def skin_panel_second_moments(
    x  : np.ndarray,
    z  : np.ndarray,
    t  : float,
    Xc : float,
    Zc : float,
) -> tuple[float, float, float]:
    '''
    Second moments of area contribution from a single skin panel about
    the centroid (Xc, Zc).

    Args:
        x:  x-coordinates of the panel polyline.
        z:  z-coordinates of the panel polyline.
        t:  Panel thickness [mm].
        Xc: Centroid x-coordinate [mm].
        Zc: Centroid z-coordinate [mm].

    Returns:
        Tuple (I_XX, I_ZZ, I_XZ) contributions from this panel.
    '''
    x_mid = 0.5 * (x[:-1] + x[1:])
    z_mid = 0.5 * (z[:-1] + z[1:])
    ds    = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
    dA    = ds * t
    du    = x_mid - Xc
    dw    = z_mid - Zc
    return (
        float(np.dot(dw**2,   dA)),
        float(np.dot(du**2,   dA)),
        float(np.dot(du * dw, dA)),
    )


def discrete_section_properties(
    x: np.ndarray,
    z: np.ndarray,
    A: np.ndarray,
) -> tuple[float, float, float, float, float]:
    '''
    Centroid and second moments of area of a discrete set of area
    concentrations (boom idealization).

    Args:
        x: x-coordinates of the booms [mm].
        z: z-coordinates of the booms [mm].
        A: Boom areas [mm^2].

    Returns:
        Tuple (Xc, Zc, I_XX, I_ZZ, I_XZ).
    '''
    sum_A = float(np.sum(A)) + _TOL
    Xc    = float(np.dot(x, A)) / sum_A
    Zc    = float(np.dot(z, A)) / sum_A
    du    = x - Xc
    dw    = z - Zc
    I_XX  = float(np.dot(dw**2,   A))
    I_ZZ  = float(np.dot(du**2,   A))
    I_XZ  = float(np.dot(du * dw, A))
    return Xc, Zc, I_XX, I_ZZ, I_XZ


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
