'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Project-Wide Constants Module.

Shared constants for all modules through used throughout CL3O. All modules
must import from here instead of redefining constants locally.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np


# ========================================================================
# Design parameters
# ========================================================================

# Displacement margin of safety
U_MAX_FACTOR = 1 / 15.0         # Maximum linear displ. = u_max_factor * b
THETA_MAX    = np.radians(2.0)  # Maximum rotation

LARGE_DISPL_MS = 1.0e6

# Weighting factor for the DE objective TotalScore.
WEIGHTING_FACTOR = 1.0     # e.g. wm = 1000 -> transforms mass in kg to g

# Analyzed lifting-surface side. Selects which half-span the load mapper
# slices and which sign the spanwise stations carry through the pipeline.
# "right" -> Y > 0 (root 0 -> tip +b/2);  "left" -> Y < 0 (root 0 -> tip -b/2).
WING_SIDE = "right"                                   # "right" | "left"
WING_SIDE_SIGN = +1.0 if WING_SIDE == "right" else -1.0

# Differential Evolution default hyper-parameters
DE_HYPERPAR: dict = {
    'NP'             : 16,
    'CR'             : 0.9,
    'F'              : 0.8,
    'lambda'         : 0.5,
    'k_max'          : 200,
    'seed'           : 42,
    'std_tol'        : 1.0e-6,     # std_tol * mean_f < std_f
    'stall_patience' : 50,         # gens of no best-f improvement -> stop
}

# Distinct-individual dedup tolerance (euclidean norm in design space)
DEDUP_TOL = 1.0e-6
# Relative tolerance used to detect best-f improvement for the stall counter
STALL_REL_TOL = 1.0e-9

# Optimization design-vector boundaries
OPT_LIMS = {
    'xw1'    : (0.10, 0.40),
    'xw2'    : (0.30, 0.60),    # may overlap xw1; swap enforced at decode
    'bfk'    : (0.02, 0.10),
    'layup'  : (1, 22),
    'fl_tpr' : (0.01, 1.0),
}

# Penalty paremeters
PENALTY_VARS = {
    "Pcap" : 1000,     # if mass in kg, Pcap means P(X) = ( {Pcap} kg ) at maximum
    "psi1" : 0.10,
    "psi2" : 0.95,
    "v1" : 0.05,
    "v2" : 0.20,
    "nv_test" : 100,
    "k" : 0.3427775704335106,
    "v0" : 11.410059370446527,
    "overflow" : 1e12,
}


# ========================================================================
# Structural topology
# ========================================================================

N_BOOMS   = 7           # structural booms  (B1..B7)
N_PANELS  = 10          # T2 sub-panels
N_SEG_T1  = 7           # T1 base segments
N_FLANGES = 4           # boom flanges (T4: F1..F4)

BOOM_LBLS = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')

# T2 sub-panel index (0..9) -> T1 segment index (0..6).
# Mirrors the (T1_seg) argument passed to _segment_T2 in geom_properties.
T2_TO_T1 = np.array([2, 1, 1, 0, 0, 4, 4, 3, 5, 6], dtype=int)

# Boom index (0..6) -> flange index in lam_T4, or -1 if no flange.
# T4 mapping: F1->B3 (idx 2), F2->B5 (idx 4), F3->B1 (idx 0), F4->B7 (idx 6).
BOOM_TO_T4 = np.array([2, -1, 0, -1, 1, -1, 3], dtype=int)

# Flange index (0..3) -> boom index (inverse of BOOM_TO_T4 for flanged booms).
FLANGE_BOOM_IDX = (2, 4, 0, 6)

# Stringer boom indices (zero-area placeholders for now): B2, B6.
STRINGER_BOOM_IDX = (1, 5)


# ========================================================================
# Beam element behaviour
# ========================================================================

# When True, release matrices model a spherical hinge instead of the
# default planar release. Tracked here so that beam_element.py does not
# carry module-level globals.
SPHERICAL_HINGE = False


# ========================================================================
# Unit conversions
# ========================================================================

T_TO_KG = 1000.0    # tonne -> kilogram (density stored in t/mm^3)

TO_METERS: dict[str, float] = {
    'ft':      0.3048,
    'm':       1.0,
    'km':      1000.0,
    'in':      0.0254,
    'mi':      1609.344,
    'naut mi': 1852.0,
}

TO_MPS: dict[str, float] = {
    'ft/s':   0.3048,
    'm/s':    1.0,
    'km/s':   1000.0,
    'in/s':   0.0254,
    'km/h':   1.0 / 3.6,
    'mph':    0.44704,
    'kts':    1852.0 / 3600.0,
    'ft/min': 0.3048 / 60.0,
}


# ========================================================================
# Numerical integration / float point tolerance
# ========================================================================

N_DEC = 4
TOL   = 1e-12
