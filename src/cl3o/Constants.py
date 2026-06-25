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
WING_SIDE = "left"                                   # "right" | "left"
WING_SIDE_SIGN = +1.0 if WING_SIDE == "right" else -1.0

# Differential Evolution default hyper-parameters
# Latin-Hypercube sample #14 from tune-de-3:
# NP = 23, CR = 0.9089, F = 0.7081, lambda = 0.3930 (stall - 50)
DE_HYPERPAR: dict = {
    'NP'             : 23,
    'CR'             : 0.9089,
    'F'              : 0.7081,
    'lambda'         : 0.3930,
    'k_max'          : 400,
    'seed'           : 67,
    'std_tol'        : 0.01,    # std_tol * mean_f < std_f ||| 0.01 -> 1 gram (mass) of std
    'stall_patience' : 80,      # gens of no best-f improvement -> stop
}

# Relative tolerance used to detect best-f improvement for the stall counter
STALL_REL_TOL = 0.01    # 1 gram order

# Optimization design-vector boundaries
OPT_LIMS = {
    'xw1'          : (0.10, 0.40),
    'xw2'          : (0.30, 0.65),  # may overlap xw1; swap enforced at decode
    # 'xw2'          : (0.30, 0.75),  # may overlap xw1; swap enforced at decode
    'bfk'          : (0.02, 0.10),
    'layup_skin'   : (0, 11),       # ls1, ls2  - AP/QI group (torsion/shear)
    'layup_web'    : (0, 11),       # lw1, lw2  - AP/QI group (torsion/shear)
    'layup_flange' : (0, 11),        # lf1..lf4  - UD/CRS group (normal stress)
    # 'layup_skin'   : (6, 11),       # ls1, ls2  - AP/QI group (torsion/shear)
    # 'layup_web'    : (6, 11),       # lw1, lw2  - AP/QI group (torsion/shear)
    # 'layup_flange' : (0, 5),        # lf1..lf4  - UD/CRS group (normal stress)
    'fl_tpr'       : (0.01, 1.0),
}

# Canonical 0-based ordering of the curated laminate catalogue.
# Index k in the DE design vector maps to LAYUP_ORDER[k] (i.e. MAT{k+1} in
# laminate_db). Groups: UD 0-2, CRS 3-5, AP 6-8, QI 9-11.
LAYUP_ORDER: list[str] = [
    "MAT_CFRP_AS4_UD24",    # 0  - UD   CFRP AS4
    "MAT_CFRP_AS4_CRS24",   # 1  - CRS  CFRP AS4
    "MAT_CFRP_AS4_AP24",    # 2  - AP   CFRP AS4
    "MAT_CFRP_AS4_QI24",    # 3  - QI   CFRP AS4
    "MAT_CFRP_IM7_UD24",    # 4  - UD   CFRP IM7
    "MAT_CFRP_IM7_CRS24",   # 5  - CRS  CFRP IM7
    "MAT_CFRP_IM7_AP24",    # 6  - AP   CFRP IM7
    "MAT_CFRP_IM7_QI24",    # 7  - QI   CFRP IM7
    "MAT_SAND-HC_UD",       # 8  - UD   sandwich HC
    "MAT_SAND-HC_CRS",      # 9  - CRS  sandwich HC
    "MAT_SAND-HC_AP",       # 10 - AP   sandwich HC
    "MAT_SAND-HC_QI",       # 11 - QI   sandwich HC
]

# LAYUP_ORDER: list[str] = [  # old     new 
#     "MAT_CFRP_AS4_UD24",    # 0       0   - UD   CFRP AS4
#     "MAT_CFRP_IM7_UD24",    # 3       1   - UD   CFRP IM7
#     "MAT_SAND-HC_UD",       # 6       2   - UD   sandwich HC
#     "MAT_CFRP_AS4_CRS24",   # 9       3   - CRS  CFRP AS4
#     "MAT_CFRP_IM7_CRS24",   # 1       4   - CRS  CFRP IM7
#     "MAT_SAND-HC_CRS",      # 4       5   - CRS  sandwich HC
#     "MAT_CFRP_AS4_AP24",    # 7       6   - AP   CFRP AS4
#     "MAT_CFRP_IM7_AP24",    # 10      7   - AP   CFRP IM7
#     "MAT_SAND-HC_AP",       # 2       8   - AP   sandwich HC
#     "MAT_CFRP_AS4_QI24",    # 5       9   - QI   CFRP AS4
#     "MAT_CFRP_IM7_QI24",    # 8       10  - QI   CFRP IM7
#     "MAT_SAND-HC_QI",       # 11      11  - QI   sandwich HC
# ]

# Penalty paremeters
PENALTY_VARS = {
    "Pcap" : 1000,     # if mass in kg, Pcap means P(X) = ( {Pcap} kg ) at maximum
    "psi1" : 0.30,
    "psi2" : 0.95,
    "v1" : 0.05,
    "v2" : 0.20,
    "nv_test" : 10,
    "k" : 2.5278245597024287,
    "v0" : 0.8351885545755384,
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
# Runtime memoization caches
# ========================================================================

# Entry ceilings for the shared geometry/beam memoization caches (LRUCache).
# The caches are keyed by per-candidate design variables, so each distinct
# DE trial adds entries that would otherwise never be evicted. Bounding them
# caps worst-case RAM while preserving the convergence-driven hit rate
# (recently used candidates - the ones that repeat - stay resident).
# Set to 0 to disable eviction (unbounded, legacy behaviour).
GEOM_CACHE_MAXSIZE = 20_000     # GeomData entries (one per station per candidate)
BEAM_CACHE_MAXSIZE = 20_000     # BeamData+T-matrix tuples (one per element)


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
