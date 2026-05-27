'''
================================================================================
CL3O - Composite Wing Structural Sizing.
Laminate Module.

Ply-by-ply calculations and laminate engineering constants for composite
materials. Follows the on-axis / off-axis convention from Tsai & Hahn (1980):
    xys --> on-axis, aligned with principal fiber direction
    126 --> off-axis, aligned with global composite axis (angle = 0)

@ CL3O Authors - MIT License
================================================================================
'''
# ================ Module imports ================
from laminate import Ply, Laminate


# ==============================================================================
# Public API - Bulk data for single ply materials
# ==============================================================================

def write_plies() -> None:
    # -------- Table A.4 - Unidirectional Composites (Daniel & Ishai) --------
    # rho: g/cm^3 * 1e-9 = ton/mm^3  |  E, strengths: GPa * 1000 = MPa

    # E-Glass/Epoxy, Vf = 0.55
    Ply(
        name  = "E-Glass--Epoxy",
        thick = 0.25,
        angle = 0,
        rho   = 1.97e-9,
        Ex    = 41000,
        Ey    = 10400,
        Es    = 4300,
        nux   = 0.28,
        Xt    = 1140,
        Xc    = 620,
        Yt    = 39,
        Yc    = 128,
        S     = 89,
    )

    # S-Glass/Epoxy, Vf = 0.50
    Ply(
        name  = "S-Glass--Epoxy",
        thick = 0.25,
        angle = 0,
        rho   = 2.00e-9,
        Ex    = 45000,
        Ey    = 11000,
        Es    = 4500,
        nux   = 0.29,
        Xt    = 1725,
        Xc    = 690,
        Yt    = 49,
        Yc    = 158,
        S     = 70,
    )

    # Kevlar/Epoxy (Aramid 49), Vf = 0.60
    Ply(
        name  = "Kevlar49--Epoxy",
        thick = 0.125,
        angle = 0,
        rho   = 1.38e-9,
        Ex    = 80000,
        Ey    = 5500,
        Es    = 2200,
        nux   = 0.34,
        Xt    = 1400,
        Xc    = 335,
        Yt    = 30,
        Yc    = 158,
        S     = 49,
    )

    # Carbon/Epoxy AS4/3501-6, Vf = 0.63
    Ply(
        name  = "CFRP-AS4--3501-6",
        thick = 0.125,
        angle = 0,
        rho   = 1.60e-9,
        Ex    = 147000,
        Ey    = 10300,
        Es    = 7000,
        nux   = 0.27,
        Xt    = 2280,
        Xc    = 1725,
        Yt    = 57,
        Yc    = 228,
        S     = 76,
    )

    # Carbon/Epoxy IM6G/3501-6, Vf = 0.66
    Ply(
        name  = "CFRP-IM6G--3501-6",
        thick = 0.125,
        angle = 0,
        rho   = 1.62e-9,
        Ex    = 169000,
        Ey    = 9000,
        Es    = 6500,
        nux   = 0.31,
        Xt    = 2240,
        Xc    = 1680,
        Yt    = 46,
        Yc    = 215,
        S     = 73,
    )

    # Carbon/Epoxy IM7/977-3, Vf = 0.65
    Ply(
        name  = "CFRP-IM7--977-3",
        thick = 0.125,
        angle = 0,
        rho   = 1.61e-9,
        Ex    = 190000,
        Ey    = 9900,
        Es    = 7800,
        nux   = 0.35,
        Xt    = 3250,
        Xc    = 1590,
        Yt    = 62,
        Yc    = 200,
        S     = 75,
    )

    # Carbon/PEEK AS4/APC2, Vf = 0.58
    Ply(
        name  = "CFRP-AS4--APC2-PEEK",
        thick = 0.125,
        angle = 0,
        rho   = 1.57e-9,
        Ex    = 138000,
        Ey    = 8700,
        Es    = 5000,
        nux   = 0.28,
        Xt    = 2060,
        Xc    = 1100,
        Yt    = 78,
        Yc    = 196,
        S     = 157,
    )

    # Carbon/Polyimide Mod I/WRD9371, Vf = 0.45
    Ply(
        name  = "CFRP-ModI--WRD9371-PI",
        thick = 0.125,
        angle = 0,
        rho   = 1.54e-9,
        Ex    = 216000,
        Ey    = 5000,
        Es    = 4500,
        nux   = 0.25,
        Xt    = 807,
        Xc    = 655,
        Yt    = 15,
        Yc    = 71,
        S     = 22,
    )

    # Graphite/Epoxy GY-70/934, Vf = 0.57
    Ply(
        name  = "GrFRP-GY70--934",
        thick = 0.125,
        angle = 0,
        rho   = 1.59e-9,
        Ex    = 294000,
        Ey    = 6400,
        Es    = 4900,
        nux   = 0.23,
        Xt    = 985,
        Xc    = 690,
        Yt    = 29,
        Yc    = 98,
        S     = 49,
    )

    # Boron/Epoxy B5.6/5505, Vf = 0.50
    Ply(
        name  = "Boron--Epoxy-B56--5505",
        thick = 0.125,
        angle = 0,
        rho   = 2.03e-9,
        Ex    = 201000,
        Ey    = 21700,
        Es    = 5400,
        nux   = 0.17,
        Xt    = 1380,
        Xc    = 1600,
        Yt    = 56,
        Yc    = 125,
        S     = 62,
    )

    # -------- Tables 1.7 / 7.1 - Unidirectional Composites (Tsai & Hahn) --------
    # rho: SG * 1e-9 = ton/mm^3 (SG = specific gravity in g/cm^3)
    # Unit ply thickness per book convention: 125e-6 m = 0.125 mm.

    # Graphite/Epoxy T300/5208, Vf = 0.70
    Ply(
        name  = "CFRP-T300--5208",
        thick = 0.125,
        angle = 0,
        rho   = 1.60e-9,
        Ex    = 181000,
        Ey    = 10300,
        Es    = 7170,
        nux   = 0.28,
        Xt    = 1500,
        Xc    = 1500,
        Yt    = 40,
        Yc    = 246,
        S     = 68,
    )

    # Boron/Epoxy B(4)/5505, Vf = 0.50
    Ply(
        name  = "Boron--Epoxy-B4--5505",
        thick = 0.125,
        angle = 0,
        rho   = 2.00e-9,
        Ex    = 204000,
        Ey    = 18500,
        Es    = 5590,
        nux   = 0.23,
        Xt    = 1260,
        Xc    = 2500,
        Yt    = 61,
        Yc    = 202,
        S     = 67,
    )

    # Graphite/Epoxy AS/3501, Vf = 0.66
    Ply(
        name  = "CFRP-AS--3501",
        thick = 0.125,
        angle = 0,
        rho   = 1.60e-9,
        Ex    = 138000,
        Ey    = 8960,
        Es    = 7100,
        nux   = 0.30,
        Xt    = 1447,
        Xc    = 1447,
        Yt    = 51.7,
        Yc    = 206,
        S     = 93,
    )

    # Glass/Epoxy Scotchply 1002, Vf = 0.45
    Ply(
        name  = "GFRP-Scotchply-1002",
        thick = 0.125,
        angle = 0,
        rho   = 1.80e-9,
        Ex    = 38600,
        Ey    = 8270,
        Es    = 4140,
        nux   = 0.26,
        Xt    = 1062,
        Xc    = 610,
        Yt    = 31,
        Yc    = 118,
        S     = 72,
    )

    # Aramid/Epoxy Kevlar 49/Epoxy, Vf = 0.60 (Tsai & Hahn data)
    Ply(
        name  = "Kevlar49--Epoxy-TH",
        thick = 0.125,
        angle = 0,
        rho   = 1.46e-9,
        Ex    = 76000,
        Ey    = 5500,
        Es    = 2300,
        nux   = 0.34,
        Xt    = 1400,
        Xc    = 235,
        Yt    = 12,
        Yc    = 53,
        S     = 34,
    )

    # -------- Table A.5 - Fabric Composites (Daniel & Ishai) --------

    # Woven Glass/Epoxy 7781/5245C, Vf = 0.45
    Ply(
        name  = "WGlass--Epoxy-7781--5245C",
        thick = 0.30,
        angle = 0,
        rho   = 2.20e-9,
        Ex    = 29700,
        Ey    = 29700,
        Es    = 5300,
        nux   = 0.17,
        Xt    = 367,
        Xc    = 549,
        Yt    = 367,
        Yc    = 549,
        S     = 97.1,
    )

    # Woven Glass/Epoxy 120/3501-6, Vf = 0.55  -- Xc/Yc not listed in table
    Ply(
        name  = "WGlass--Epoxy-120--3501-6",
        thick = 0.30,
        angle = 0,
        rho   = 1.97e-9,
        Ex    = 27500,
        Ey    = 26700,
        Es    = 5500,
        nux   = 0.14,
        Xt    = 435,
        Xc    = 435,
        Yt    = 386,
        Yc    = 386,
        S     = 55,
    )

    # Woven Glass/Epoxy M10E/3783, Vf = 0.50
    Ply(
        name  = "WGlass--Epoxy-M10E--3783",
        thick = 0.30,
        angle = 0,
        rho   = 1.90e-9,
        Ex    = 24500,
        Ey    = 23800,
        Es    = 4700,
        nux   = 0.11,
        Xt    = 433,
        Xc    = 377,
        Yt    = 386,
        Yc    = 335,
        S     = 84,
    )

    # Kevlar 49 Fabric/Epoxy K120/M10.2  -- rho not listed in table
    Ply(
        name  = "Kevlar49-Fabric--Epoxy-K120--M10.2",
        thick = 0.30,
        angle = 0,
        rho   = 1.38e-9,
        Ex    = 29000,
        Ey    = 29000,
        Es    = 18000,
        nux   = 0.05,
        Xt    = 369,
        Xc    = 129,
        Yt    = 369,
        Yc    = 129,
        S     = 113,
    )

    # Carbon Fabric/Epoxy AGP370-5H/3501-6S, Vf = 0.62
    Ply(
        name  = "CFabric--Epoxy-AGP370-5H--3501-6S",
        thick = 0.30,
        angle = 0,
        rho   = 1.60e-9,
        Ex    = 77000,
        Ey    = 75000,
        Es    = 6500,
        nux   = 0.06,
        Xt    = 963,
        Xc    = 900,
        Yt    = 856,
        Yc    = 900,
        S     = 71,
    )

    # -------- Table A.9 - Sandwich Core Materials (Daniel & Ishai) --------
    # rho: kg/m^3 * 1e-12 = ton/mm^3

    # Divinycell H80
    Ply(
        name  = "Divinycell-H80",
        thick = 10.0,
        angle = 0,
        core  = True,
        rho   = 80e-12,
    )

    # Divinycell H100
    Ply(
        name  = "Divinycell-H100",
        thick = 10.0,
        angle = 0,
        core  = True,
        rho   = 100e-12,
    )

    # Divinycell H160
    Ply(
        name  = "Divinycell-H160",
        thick = 10.0,
        angle = 0,
        core  = True,
        rho   = 160e-12,
    )

    # Divinycell H250
    Ply(
        name  = "Divinycell-H250",
        thick = 10.0,
        angle = 0,
        core  = True,
        rho   = 250e-12,
    )

    # Balsa Wood CK57
    Ply(
        name  = "Balsa-CK57",
        thick = 10.0,
        angle = 0,
        core  = True,
        rho   = 150e-12,
    )

    # Aluminum Honeycomb PAMG 5052
    Ply(
        name  = "Al-Honeycomb-PAMG-5052",
        thick = 10.0,
        angle = 0,
        core  = True,
        rho   = 130e-12,
    )

    # Foam-Filled Honeycomb Style 20
    Ply(
        name  = "Foam-Honeycomb-Style-20",
        thick = 10.0,
        angle = 0,
        core  = True,
        rho   = 128e-12,
    )

    # Polyurethane FR-3708
    Ply(
        name  = "Polyurethane-FR-3708",
        thick = 10.0,
        angle = 0,
        core  = True,
        rho   = 128e-12,
    )

    # -------- Standalone multi-angle variants (not in any laminate) --------
    # Defined explicitly so a clean regen preserves them. AS4--3501-6-UD is
    # an alternate AS4 set (distinct from CFRP-AS4--3501-6); Kevlar49--Epoxy
    # reuses the base data defined earlier (off-axis angles only).

    as4_ud = dict(
        name="AS4--3501-6-UD", thick=0.125, rho=1.58e-9,
        Ex=142000, Ey=10300, Es=5700, nux=0.27,
        Xt=2280, Xc=1440, Yt=57, Yc=228, S=71,
    )
    for ang in (0, 45, -45, 90):
        Ply(angle=ang, **as4_ud)

    kevlar49 = dict(
        name="Kevlar49--Epoxy", thick=0.125, rho=1.38e-9,
        Ex=80000, Ey=5500, Es=2200, nux=0.34,
        Xt=1400, Xc=335, Yt=30, Yc=158, S=49,
    )
    for ang in (45, -45, 90, -90):
        Ply(angle=ang, **kevlar49)


# ==============================================================================
# Public API - Define new laminate materials
# ==============================================================================

# --------------------------------------------------------
# Ply material shortcuts - matched to the ply DB on disk
# --------------------------------------------------------

_CFRP_AS4 = dict(
    name="CFRP-AS4--3501-6", thick=0.125, rho=1.60e-9,
    Ex=147000, Ey=10300, Es=7000, nux=0.27,
    Xt=2280, Xc=1725, Yt=57, Yc=228, S=76,
)

_CFRP_T300 = dict(
    name="CFRP-T300--5208", thick=0.125, rho=1.60e-9,
    Ex=181000, Ey=10300, Es=7170, nux=0.28,
    Xt=1500, Xc=1500, Yt=40, Yc=246, S=68,
)

_GFRP = dict(
    name="GFRP-Scotchply-1002", thick=0.125, rho=1.80e-9,
    Ex=38600, Ey=8270, Es=4140, nux=0.26,
    Xt=1062, Xc=610, Yt=31, Yc=118, S=72,
)

_CORE_H80     = dict(name="Divinycell-H80",         rho=80e-12)
_CORE_H100    = dict(name="Divinycell-H100",        rho=100e-12)
_CORE_H160    = dict(name="Divinycell-H160",        rho=160e-12)
_CORE_AL_HC   = dict(name="Al-Honeycomb-PAMG-5052", rho=130e-12)

_CORE_THICK = 5.0   # [mm] - sandwich core ply thickness


# --------------------------------------------------------
# Internal helpers - laminate stacking
# --------------------------------------------------------

def _stack(mat: Laminate, ply_spec: dict, angles: list[int]) -> None:
    '''Append plies (bottom to top) reusing one ply material spec.'''
    for ang in angles:
        mat.add_ply(angle=ang, **ply_spec)


def _add_core(mat: Laminate, core_spec: dict, thick: float = _CORE_THICK) -> None:
    '''Append a single core ply.'''
    mat.add_ply(angle=0, core=True, thick=thick, **core_spec)


def _build(name: str, builder) -> None:
    '''Instantiate a Laminate, run the builder closure, persist it.'''
    mat = Laminate(name=name)
    builder(mat)
    mat.define_laminate_data()


# ==============================================================================
# Public API - Curated laminate catalogue (22 entries)
# ==============================================================================

def write_laminates() -> None:
    '''
    Build the curated laminate catalogue used by the DE optimizer.

    Design rules
    ------------
    - Minimum total laminate thickness: 2.00 mm (>= 16 plies at 0.125 mm
      each for solid laminates; always satisfied for sandwiches).
    - Orientation priority: 0 / 90 plies (normal / bending stiffness) are
      placed before 45 / -45 (shear stiffness) in every layup sequence.
    - Sandwich variants are provided for both 0/90-dominant (Groups A/B)
      and 45/-45-dominant (Group C) structural roles.

    Group A - Boom / flange (axial / bending, 0-dominant): 4 entries
        Pure UD and hard layups (mostly 0 deg) at two thickness tiers.
    Group B - Skin (normal / bending, 0/90-dominant): 5 entries
        Cross-ply and quasi-isotropic solid skins at two thickness tiers.
    Group C - Web (shear-dominated, +/-45-dominant): 3 entries
        Angle-ply and angle-ply-hard solid laminates.
    Group D - Sandwich (buckling-critical): 8 entries
        0/90-dominant (UD, CRS, QI) and +/-45-dominant (AP) facings
        paired with Divinycell H80 / H100 / H160 and Al-honeycomb cores.
    '''
    # -------- Group A - Boom / flange laminates --------
    # 16 plies = 2.00 mm | 24 plies = 3.00 mm

    _build("MAT_CFRP_UD16",   lambda m: _stack(m, _CFRP_AS4, [0]*16))
    _build("MAT_CFRP_UD24",   lambda m: _stack(m, _CFRP_AS4, [0]*24))
    _build("MAT_CFRP_HARD16", lambda m: _stack(m, _CFRP_AS4,
        # 12 x 0 deg + 4 x +/-45 deg; symmetric & balanced
        [0, 0, 0, 0, 0, 0, 45, -45,
         -45, 45, 0, 0, 0, 0, 0, 0]))
    _build("MAT_CFRPHM_UD16", lambda m: _stack(m, _CFRP_T300, [0]*16))

    # -------- Group B - Skin laminates --------
    # 16 plies = 2.00 mm | 24 plies = 3.00 mm

    _build("MAT_CFRP_CRS16",  lambda m: _stack(m, _CFRP_AS4,
        # Cross-ply: alternating 0/90; symmetric & balanced
        [0, 90] * 8))
    _build("MAT_CFRP_CRS24",  lambda m: _stack(m, _CFRP_AS4,
        [0, 90] * 12))
    _build("MAT_CFRP_QI16",   lambda m: _stack(m, _CFRP_AS4,
        # Quasi-isotropic: 0/45/-45/90 repeated; symmetric & balanced
        [0, 45, -45, 90, 0, 45, -45, 90,
         90, -45, 45, 0, 90, -45, 45, 0]))
    _build("MAT_CFRP_QI24",   lambda m: _stack(m, _CFRP_AS4,
        [0, 45, -45, 90] * 6))
    _build("MAT_GFRP_CRS16",  lambda m: _stack(m, _GFRP,
        [0, 90] * 8))

    # -------- Group C - Web laminates --------
    # 16 plies = 2.00 mm | 24 plies = 3.00 mm

    _build("MAT_CFRP_AP16",     lambda m: _stack(m, _CFRP_AS4,
        # Pure angle-ply; symmetric & balanced
        [45, -45] * 8))
    _build("MAT_CFRP_AP24",     lambda m: _stack(m, _CFRP_AS4,
        [45, -45] * 12))
    _build("MAT_CFRP_APHARD16", lambda m: _stack(m, _CFRP_AS4,
        # Angle-ply hard: 8 x +/-45 + 8 x 0; symmetric & balanced
        [45, -45, 0, 0, 0, 0, -45, 45,
         45, -45, 0, 0, 0, 0, -45, 45]))

    # -------- Group D - Sandwich laminates --------
    # All facings use CFRP-AS4; core adds >= 5 mm --> total >= 6.0 mm.

    def _ud_sandwich(core_spec):
        '''UD (0-dominant) facings: 8 plies each side = 1.00 mm/face.'''
        def builder(m):
            _stack(m, _CFRP_AS4, [0] * 8)
            _add_core(m, core_spec)
            _stack(m, _CFRP_AS4, [0] * 8)
        return builder

    def _crs_sandwich(core_spec):
        '''Cross-ply facings: [0,90,90,0] each side = 0.50 mm/face.'''
        def builder(m):
            _stack(m, _CFRP_AS4, [0, 90, 90, 0])
            _add_core(m, core_spec)
            _stack(m, _CFRP_AS4, [0, 90, 90, 0])
        return builder

    def _qi_sandwich(core_spec):
        '''QI facings: [0,45,-45,90] / [90,-45,45,0] = 0.50 mm/face.'''
        def builder(m):
            _stack(m, _CFRP_AS4, [0, 45, -45, 90])
            _add_core(m, core_spec)
            _stack(m, _CFRP_AS4, [90, -45, 45, 0])
        return builder

    def _ap_sandwich(core_spec):
        '''Angle-ply facings: [45,-45,-45,45] each side = 0.50 mm/face.'''
        def builder(m):
            _stack(m, _CFRP_AS4, [45, -45, -45, 45])
            _add_core(m, core_spec)
            _stack(m, _CFRP_AS4, [45, -45, -45, 45])
        return builder

    # 0-dominant sandwiches (Groups A/B roles)
    _build("MAT_SAND_UD_H100",  _ud_sandwich(_CORE_H100))
    _build("MAT_SAND_CRS_H80",  _crs_sandwich(_CORE_H80))
    _build("MAT_SAND_CRS_H100", _crs_sandwich(_CORE_H100))
    _build("MAT_SAND_QI_H80",   _qi_sandwich(_CORE_H80))
    _build("MAT_SAND_QI_H100",  _qi_sandwich(_CORE_H100))
    _build("MAT_SAND_QI_H160",  _qi_sandwich(_CORE_H160))

    # +/-45-dominant sandwiches (Group C role, buckling-critical webs)
    _build("MAT_SAND_AP_H80",   _ap_sandwich(_CORE_H80))
    _build("MAT_SAND_AP_HC",    _ap_sandwich(_CORE_AL_HC))


# ============================================================================ #

if __name__ == '__main__':
    write_plies()
    write_laminates()