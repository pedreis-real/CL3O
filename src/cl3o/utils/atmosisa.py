'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Atmosphere (ISA) Module.

Implements the ICAO International Standard Atmosphere (ISA) model,
computing temperature, pressure, density, speed of sound, kinematic
viscosity and dynamic viscosity for geopotential altitudes between
-5000 m and 84852 m (mesopause). Reference: U.S. Standard Atmosphere,
1976, U.S. Government Printing Office, Washington, D.C.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import warnings
import numpy as np


# ========================================================================
# PUBLIC API - International Standard Atmosphere
# ========================================================================

def atmosisa(
    h:      float | np.ndarray,
    action: str = 'warning',
) -> tuple:
    '''
    Compute ISA atmospheric properties for given geopotential altitudes.

    Args:
        h:      Geopotential altitude(s) in meters.
        action: Behaviour when h is outside [-5000, 84852] m.
                'warning' issues a UserWarning (default).
                'error'   raises a ValueError.
                'none'    silently ignores the violation.

    Returns:
        Tuple (T, a, P, rho, nu, mu) of ndarrays, or scalars for scalar
        input. T is temperature [K], a is speed of sound [m/s], P is
        pressure [Pa], rho is density [kg/m^3], nu is kinematic
        viscosity [m^2/s] and mu is dynamic viscosity [N*s/m^2].
    '''

    # -------- Physical constants --------
    g0    = 9.80665
    gamma = 1.4
    beta  = 1.458e-6
    S     = 110.4

    # -------- ISA layer definitions --------
    L  = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])
    Hb = np.array(
        [0, 11000, 20000, 32000, 47000, 51000, 71000, 84852],
        dtype=float,
    )
    T0 = np.array([
        288.1500, 216.6500, 216.6500, 228.6500,
        270.6500, 270.6500, 214.6500, 186.9460,
    ])
    P0 = np.array([
        101325.0,
        22632.0405969347,
        5474.8776606600,
        868.0158377494,
        110.9057845539,
        66.9385353730,
        3.9563927546,
        0.3733803710,
    ])

    rho0 = 1.225
    R    = P0[0] / (rho0 * T0[0])

    # -------- Input preparation --------
    scalar_input = np.ndim(h) == 0
    h = np.atleast_1d(np.array(h, dtype=float))

    _handle_action(h, action)
    h = np.clip(h, -5000.0, 84852.0)

    T = np.zeros_like(h)
    P = np.zeros_like(h)

    # -------- Sea-level --------
    mask_zero    = h == 0.0
    T[mask_zero] = T0[0]
    P[mask_zero] = P0[0]

    # -------- Below sea-level (layer 0 extrapolated) --------
    mask_neg = h < 0.0
    if np.any(mask_neg):
        T[mask_neg] = T0[0] + L[0] * (h[mask_neg] - Hb[0])
        theta        = T[mask_neg] / T0[0]
        P[mask_neg]  = P0[0] * theta ** (-g0 / (L[0] * R))

    # -------- Positive-altitude layers --------
    hvals = h.copy()
    hvals[mask_neg | mask_zero] = 0.0

    for i in range(len(Hb) - 1):
        if not np.any(hvals):
            break

        mask = (h > Hb[i]) & (h <= Hb[i + 1])
        if not np.any(mask):
            continue

        if L[i] == 0.0:
            T[mask] = T0[i]
            P[mask] = P0[i] * np.exp(
                -g0 * (h[mask] - Hb[i]) / (R * T0[i])
            )
        else:
            T[mask] = T0[i] + L[i] * (h[mask] - Hb[i])
            theta    = T[mask] / T0[i]
            P[mask]  = P0[i] * theta ** (-g0 / (L[i] * R))

        hvals[mask] = 0.0

    # -------- Derived quantities --------
    rho = P / (R * T)
    a   = np.sqrt(gamma * R * T)
    mu  = beta * (T ** 1.5) / (T + S)
    nu  = mu / rho

    if scalar_input:
        return T[0], a[0], P[0], rho[0], nu[0], mu[0]

    return T, a, P, rho, nu, mu


# ========================================================================
# PRIVATE API - Internal helpers
# ========================================================================

def _handle_action(h: np.ndarray, action: str) -> None:
    '''Validate altitude range and act according to the action flag.'''
    action_lower = action.strip().lower()
    out_of_range = (h < -5000.0) | (h > 84852.0)

    if action_lower == 'none' or not np.any(out_of_range):
        return

    msg = (
        'Altitude(s) outside the valid range [-5000 m, 84852 m]. '
        'Output values will be held at the limits.'
    )

    if action_lower == 'warning':
        warnings.warn(msg, UserWarning, stacklevel=3)
    elif action_lower == 'error':
        idx = int(np.argmax(out_of_range))
        raise ValueError(
            f'[CL3O] Altitude out of valid range.\n'
            f'| Index : {idx}\n'
            f'| Value : {h.flat[idx]:.4f} m\n'
            f'Valid range is -5000 m to 84852 m.'
        )
    else:
        raise ValueError(
            f"[CL3O] Invalid 'action' value: '{action}'.\n"
            f"Use 'warning', 'error', or 'none'."
        )


if __name__ == '__main__':
    _alts = [16000 * 0.3048]

    print(
        f"{'Alt (m)':>10} {'T (K)':>10} {'a (m/s)':>10} "
        f"{'P (Pa)':>14} {'rho (kg/m3)':>13} "
        f"{'nu (m2/s)':>13} {'mu (N*s/m2)':>14}"
    )
    print('-' * 90)

    for alt in _alts:
        T, a, P, rho, nu, mu = atmosisa(alt)
        print(
            f'{alt:>10.0f} {T:>10.4f} {a:>10.4f} '
            f'{P:>14.4f} {rho:>13.6f} '
            f'{nu:>13.6e} {mu:>14.6e}'
        )
