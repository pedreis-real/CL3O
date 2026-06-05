'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Wing-Surface Builder Module.

Reconstructs the 3-D analyzed-wing visualization scene from an archived
RuntimeData snapshot, mirroring the geometry recipe in
src/validation/validate_fea.py:

  - lofted outer skin surface  (airfoil loop scaled/twisted along the span)
  - two spar surfaces          (front spar @ xw1 = seg6, rear @ xw2 = seg7)
  - centroid line              (per-station GeomData.C)
  - shear-centre line          (per-station GeomData.S_XYZ)

Output is emitted as Plotly mesh3d-ready vertices + triangle indices and
polyline point lists. Optionally the surface is deformed by the nodal
solution (per-station rigid translation + small rotation about the LE) so
the same builder feeds the Mesh post-processing view.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from __future__ import annotations

import numpy as np

# ================ Pathing ================
from . import paths                                # project path constants

# ================ Module imports ================

# Geometry
from cl3o.geometry.wing import WingHelper

# ================ Module constants ================
_N_CHORD    = 81              # airfoil-loop points per rib (kept light for WebGL)

# Per-panel shear-flux fields lofted onto the stress surface. Closed-cell
# fluxes (qsX/qsZ/qT) scale with the shear-centre forces SX/SZ/T; open-cell
# fluxes (qbX/qbZ) reuse the SX/SZ forces. Input attr names on GeomData map
# to the output keys advertised to the frontend, index-aligned.
_FLUX_KEYS = ("qsX_star", "qsZ_star", "qT_star", "qbX_star", "qbZ_star")
_FLUX_OUT  = ("flux_qsX", "flux_qsZ", "flux_qT", "flux_qbX", "flux_qbZ")


# ========================================================================
# PRIVATE API - geometry primitives
# ========================================================================

def _skin_rib(gd) -> tuple[np.ndarray, int]:
    '''Outer skin contour for one section from T1 seg1..seg5, in global XZ.

    T1 outer segments in connectivity order (seg1 = LE cell, seg2..5 = mid/TE):
        seg1: B5 → (LE) → B3   lower-front skin wrapping around the nose  (ls1)
        seg2: B3 → B1           upper mid skin                              (ls2)
        seg3: B1 → TE           upper TE skin                               (ls2)
        seg4: TE → B7           lower TE skin                               (ls2)
        seg5: B7 → B5           lower mid skin                              (ls2)

    All pts are already in global XZ (LE_xz offset applied by section builder).
    Sub-sampled to at most _N_CHORD points for WebGL budget.

    Returns:
        Tuple (pts, n_ls1) where pts is an (N, 2) float array [x, z] and
        n_ls1 is the number of points in pts belonging to seg1 (ls1 region).
    '''
    segs = {s["label"]: np.asarray(s["pts"], float) for s in gd.T1}
    parts = []
    for k, lbl in enumerate(("seg1", "seg2", "seg3", "seg4", "seg5")):
        p = segs.get(lbl)
        if p is None:
            continue
        parts.append(p if k == 0 else p[1:])   # skip shared endpoint
    if not parts:
        return np.zeros((0, 2)), 0
    n_ls1_raw = len(parts[0])
    loop = np.vstack(parts)
    n_raw = loop.shape[0]
    if n_raw > _N_CHORD:
        idx = np.unique(np.linspace(0, n_raw - 1, _N_CHORD).round().astype(int))
        loop = loop[idx]
        n_ls1_in_loop = int(np.sum(idx < n_ls1_raw))
    else:
        n_ls1_in_loop = n_ls1_raw
    return loop, n_ls1_in_loop


def _span_stations(rt) -> list[int]:
    '''Section indices of the analyzed wing, ordered root -> tip (|Y| ascending).

    A snapshot holds only the analyzed half-span (Constants.WING_SIDE), so all
    sections belong to it regardless of sign; ordering by |Y| works for either
    the right wing (Y > 0) or the left wing (Y < 0).
    '''
    sec = rt.sections.sec_data
    idx = list(range(len(sec)))
    idx.sort(key=lambda i: abs(float(sec[i].C[1])))
    return idx


def _grid_faces(n_c: int, n_s: int) -> tuple[list, list, list]:
    '''Triangulate a structured (n_c chord x n_s span) grid; vertex = s*n_c+c.'''
    return _grid_faces_range(n_c, n_s, 0, n_c)


def _grid_faces_range(
    n_c: int, n_s: int, c_start: int, c_end: int
) -> tuple[list, list, list]:
    '''Triangulate chord strips [c_start, c_end) of a structured grid.'''
    i, j, k = [], [], []
    for s in range(n_s - 1):
        for c in range(c_start, c_end - 1):
            v00 = s * n_c + c
            v10 = s * n_c + c + 1
            v01 = (s + 1) * n_c + c
            v11 = (s + 1) * n_c + c + 1
            i += [v00, v00]
            j += [v10, v11]
            k += [v11, v01]
    return i, j, k


def _apply_disp(
    pts  : np.ndarray,
    le   : np.ndarray,
    disp : np.ndarray | None,
    scale: float,
) -> np.ndarray:
    '''Per-station rigid displacement (u,v,w + small rotation about LE).

    Args:
        pts  : (N, 3) rib points in global XYZ.
        le   : (3,) leading-edge reference for the rotation.
        disp : (6,) [u, v, w, rx, ry, rz] or None for the identity.
        scale: Displacement magnification.
    '''
    if disp is None:
        return pts
    u, v, w = disp[0], disp[1], disp[2]
    rx, ry, rz = disp[3], disp[4], disp[5]
    x_loc = pts[:, 0] - le[0]
    y_loc = np.zeros_like(x_loc)
    z_loc = pts[:, 2] - le[2]
    d_x = ry * z_loc - rz * y_loc
    d_y = rz * x_loc - rx * z_loc
    d_z = rx * y_loc - ry * x_loc
    out = pts.copy()
    out[:, 0] += scale * (u + d_x)
    out[:, 1] += scale * (v + d_y)
    out[:, 2] += scale * (w + d_z)
    return out


# ========================================================================
# PUBLIC API - scene builder
# ========================================================================

def _build_skin_verts(
    left: list, raw_ribs: list, n_c: int, y_left: np.ndarray,
    dmat, LE: np.ndarray, lrow: list, lc: int, scale: float,
) -> tuple[np.ndarray, dict]:
    '''Place each station's outer-skin rib in the wing frame, optionally
    displaced by the nodal solution.

    Args:
        left     : analyzed-wing station node indices (root -> tip).
        raw_ribs : per-station (m_i, 2) skin polylines in section global-XZ.
        n_c      : common chord-point count to resample every rib to.
        y_left   : per-station span coordinate (mm).
        dmat     : (6, n, nc) nodal displacement matrix, or None (no deform).
        LE       : per-lerp-row leading-edge coords for the deform anchor.
        lrow     : station -> lerp-row map for the displacement reference.
        lc       : load-case index (already clamped).
        scale    : deformation magnification.

    Returns:
        Tuple (verts, disp) - the (n_c, n_s, 3) vertex grid and a dict of
        per-station displacement component arrays (zeros when dmat is None).
    '''
    n_s = len(left)
    verts = np.zeros((n_c, n_s, 3))
    comp_keys = ("u", "v", "w", "t", "rx", "ry", "rz", "r")
    disp = {key: np.zeros(n_s) for key in comp_keys}
    for s, node_i in enumerate(left):
        raw = raw_ribs[s]
        if raw.shape[0] != n_c:
            idx = np.round(np.linspace(0, raw.shape[0] - 1, n_c)).astype(int)
            raw = raw[idx]
        y = y_left[s]
        rib = np.column_stack([raw[:, 0], np.full(n_c, y), raw[:, 1]])
        if dmat is not None:
            r = lrow[s]
            xle, zle = float(LE[r, 0]), float(LE[r, 2])
            d = dmat[:, node_i, lc]
            _deg = 180.0 / np.pi
            disp["u"][s], disp["v"][s], disp["w"][s] = d[0], d[1], d[2]
            disp["rx"][s] = float(d[3]) * _deg
            disp["ry"][s] = float(d[4]) * _deg
            disp["rz"][s] = float(d[5]) * _deg
            disp["t"][s]  = float(np.linalg.norm(d[0:3]))
            disp["r"][s]  = float(np.linalg.norm(d[3:6])) * _deg
            rib = _apply_disp(rib, np.array([xle, y, zle]), d, scale)
        verts[:, s] = rib
    return verts, disp


def build_scene(
    rt,
    wing,
    afl,
    lc    : int = 0,
    scale : float = 1.0,
    deform: bool = False,
) -> dict:
    '''
    Build the analyzed-wing 3-D scene for one snapshot.

    Args:
        rt     : RuntimeData snapshot.
        wing   : WingData (typed) for the run.
        afl    : AirfoilData for the run.
        lc     : Load case index for the deformation.
        scale  : Deformation magnification.
        deform : When True, displace ribs by the nodal solution and attach a
            vertical-deflection scalar (`w_global`) per vertex.

    Returns:
        Dict with mesh3d-ready `surface`, `front_spar`, `rear_spar` and the
        `centroid_line` / `shear_line` polylines, plus station metadata.
    '''
    sec = rt.sections.sec_data
    left = _span_stations(rt)
    n_s = len(left)
    y_left = np.array([float(sec[i].C[1]) for i in left])

    # Per-station LE / chord / twist via the production single-wing lerp.
    wing_side = "left" if len(y_left) > 0 and float(y_left[np.argmax(np.abs(y_left))]) < 0 else "right"
    lerp = WingHelper.lerp_from_data(wing, y_left, wing_side)
    ly = np.asarray(lerp.Y_sta, dtype=float)
    LE = np.asarray(lerp.LE, dtype=float)
    chord = np.asarray(lerp.chord, dtype=float)
    twist = np.asarray(lerp.twist, dtype=float)
    # Map each station Y -> lerp row (nearest) for displacement reference.
    lrow = [int(np.argmin(np.abs(ly - y))) for y in y_left]

    # Build per-station outer skin ribs from T1 pts (seg1..seg5 in global XZ).
    # _skin_rib returns (loop, n_ls1_in_loop): seg1 = ls1, seg2-5 = ls2.
    raw_ribs_full = [_skin_rib(sec[node_i]) for node_i in left]
    raw_ribs = [r for r, _ in raw_ribs_full]
    n_ls1s   = [n for _, n in raw_ribs_full]
    n_c = min(r.shape[0] for r in raw_ribs if r.shape[0] > 0) or _N_CHORD

    # Chord split index (last ls1 point in the n_c resampled array), root station.
    raw0   = raw_ribs[0]
    n_ls10 = n_ls1s[0]
    if raw0.shape[0] != n_c and raw0.shape[0] > 1:
        c_split = int(round((n_ls10 - 1) / (raw0.shape[0] - 1) * (n_c - 1)))
    else:
        c_split = max(0, n_ls10 - 1)
    c_split = max(1, min(c_split, n_c - 2))

    dmat = getattr(getattr(rt, "fea_rts", None), "dmatrix", None)
    if deform and dmat is not None:
        dmat = np.asarray(dmat, float)
        nc_lc = dmat.shape[2]
        lc = max(0, min(int(lc), nc_lc - 1))
    else:
        dmat = None

    verts, disp = _build_skin_verts(
        left, raw_ribs, n_c, y_left, dmat, LE, lrow, lc, scale
    )

    flat = verts.reshape(-1, 3, order="F")          # vertex idx = s*n_c + c
    disp_payload = dict(disp) if dmat is not None else None

    i, j, k = _grid_faces(n_c, n_s)
    surface = {"vertices": flat, "i": i, "j": j, "k": k, "n_chord": n_c, "n_span": n_s}
    if disp_payload is not None:
        surface["station_disp"] = disp_payload

    # ls1 sub-mesh: seg1 (LE → front spar), own vertex slice [0, c_split].
    # Slicing verts avoids Plotly rendering unused vertices with the wrong color.
    n_c1 = c_split + 1
    flat_ls1 = verts[:n_c1, :, :].reshape(-1, 3, order="F")
    i1, j1, k1 = _grid_faces(n_c1, n_s)
    surface_ls1 = {"vertices": flat_ls1, "i": i1, "j": j1, "k": k1, "n_chord": n_c1, "n_span": n_s}
    if disp_payload is not None:
        surface_ls1["station_disp"] = disp_payload

    # ls2 sub-mesh: seg2-5 (front spar → TE), own vertex slice [c_split, n_c).
    n_c2 = n_c - c_split
    flat_ls2 = verts[c_split:, :, :].reshape(-1, 3, order="F")
    i2, j2, k2 = _grid_faces(n_c2, n_s)
    surface_ls2 = {"vertices": flat_ls2, "i": i2, "j": j2, "k": k2, "n_chord": n_c2, "n_span": n_s}
    if disp_payload is not None:
        surface_ls2["station_disp"] = disp_payload

    # Spar strips and structural lines. Each section point is expressed as a
    # chord fraction of its own profile, then re-placed on the per-station
    # wing frame (LE + chord + twist) used by the skin, so the overlays track
    # sweep / dihedral / taper / twist even when sec_data is a single profile
    # replicated across stations.
    front = _spar_strip(rt, left, "seg6", dmat, LE, chord, twist, lrow, scale, lc)
    rear = _spar_strip(rt, left, "seg7", dmat, LE, chord, twist, lrow, scale, lc)
    cline = _line(rt, left, "C", dmat, LE, chord, twist, lrow, scale, lc)
    sline = _line(rt, left, "S_XYZ", dmat, LE, chord, twist, lrow, scale, lc)
    flanges = _flange_strips(
        rt, left, dmat, LE, chord, lrow, scale, lc, getattr(rt, "optvars", None),
    )

    out = {
        "surface": surface,
        "surface_ls1": surface_ls1,
        "surface_ls2": surface_ls2,
        "front_spar": front,
        "rear_spar": rear,
        "centroid_line": cline,
        "shear_line": sline,
        "n_stations": n_s,
        "y_span": y_left,
        "deformed": dmat is not None,
    }

    # Layups per panel/web/flange (per-cpt index arrays from OptVars).
    ov = getattr(rt, "optvars", None)
    if ov is not None:
        out["layups"] = {
            key: getattr(ov, key, None)
            for key in ("ls1", "ls2", "lw1", "lw2",
                        "lf1", "lf2", "lf3", "lf4")
        }
    out["flanges"] = flanges
    return out


# ========================================================================
# PUBLIC API - stress surface
# ========================================================================

def _panel_force_ends(Qsc_arr: np.ndarray, e: int, lc_q: int) -> dict:
    '''Per-element end-A / end-B applied forces that scale each flux field.

    Rows of Q_sc: 0=N, 1=Sy, 2=Sz, 3=T, 4=My, 5=Mz (then +6 for end-B with
    sign -1), following the _extract_internal_forces convention. Returns a
    map keyed by the GeomData flux attr name to its (force_A, force_B) pair.
    '''
    SX_a = float(Qsc_arr[1, e, lc_q]);   SX_b = -float(Qsc_arr[7, e, lc_q])
    SZ_a = float(Qsc_arr[2, e, lc_q]);   SZ_b = -float(Qsc_arr[8, e, lc_q])
    T_a  = float(Qsc_arr[3, e, lc_q]);   T_b  = -float(Qsc_arr[9, e, lc_q])
    return {
        "qsX_star": (SX_a, SX_b),
        "qsZ_star": (SZ_a, SZ_b),
        "qT_star":  (T_a,  T_b),
        "qbX_star": (SX_a, SX_b),
        "qbZ_star": (SZ_a, SZ_b),
    }


def _loft_stress_panels(
    sec, conn: np.ndarray, left_e: list, tau: np.ndarray,
    lc: int, end: str, Qsc_arr: np.ndarray, lc_q: int,
) -> tuple[list, list, list, list, list, dict]:
    '''Loft each T2 panel polyline between adjacent stations into triangles.

    For every analyzed-wing element the two end-station panel polylines are
    resampled to a common length and lofted into a ruled strip that hugs the
    real skin contour, coloured by the panel shear stress tau and carrying
    the per-face shear-flux values (each influence coefficient scaled by the
    actual applied force).

    Returns:
        Tuple (verts, fi, fj, fk, fc, flux_cols) - vertex list, the three
        mesh3d triangle-index lists, the per-face tau intensity, and the
        dict of per-face flux columns keyed by _FLUX_OUT.
    '''
    verts: list = []
    fi, fj, fk, fc = [], [], [], []
    flux_cols: dict[str, list] = {k: [] for k in _FLUX_OUT}

    for e in left_e:
        node_a, node_b = int(conn[e, 0]), int(conn[e, 1])
        y_a = float(sec[node_a].C[1])
        y_b = float(sec[node_b].C[1])
        n_panels_a = len(sec[node_a].T2)
        n_panels_b = len(sec[node_b].T2)

        # Per-element internal forces: end-A (sign +1, off 0) and
        # end-B (sign -1, off 6) following _extract_internal_forces convention.
        _force_ends = _panel_force_ends(Qsc_arr, e, lc_q)

        for jp in range(min(n_panels_a, n_panels_b)):
            pts_a = np.asarray(sec[node_a].T2[jp]["pts"], float)  # (na, 2)
            pts_b = np.asarray(sec[node_b].T2[jp]["pts"], float)  # (nb, 2)
            t = float(tau[e, jp, lc])

            # Multiply each influence coefficient (1/mm) by the actual
            # applied force (N) to obtain the true shear flow (N/mm).
            flux_face: dict[str, float] = {}
            for fk_in, fk_out in zip(_FLUX_KEYS, _FLUX_OUT):
                ic_a = float(np.asarray(getattr(sec[node_a], fk_in, np.zeros(10)))[jp])
                ic_b = float(np.asarray(getattr(sec[node_b], fk_in, np.zeros(10)))[jp])
                fa, fb = _force_ends[fk_in]
                va = ic_a * fa
                vb = ic_b * fb
                if end == "A":
                    flux_face[fk_out] = va
                elif end == "B":
                    flux_face[fk_out] = vb
                else:
                    flux_face[fk_out] = 0.5 * (va + vb)

            # Resample to the shorter length so index arrays align.
            na, nb_pts = pts_a.shape[0], pts_b.shape[0]
            n = min(na, nb_pts)
            if na != n:
                idx = np.round(np.linspace(0, na - 1, n)).astype(int)
                pts_a = pts_a[idx]
            if nb_pts != n:
                idx = np.round(np.linspace(0, nb_pts - 1, n)).astype(int)
                pts_b = pts_b[idx]

            # Station A vertices: base + 0 .. base + n-1
            # Station B vertices: base + n .. base + 2n-1
            base = len(verts)
            for i in range(n):
                verts.append([pts_a[i, 0], y_a, pts_a[i, 1]])
            for i in range(n):
                verts.append([pts_b[i, 0], y_b, pts_b[i, 1]])

            for i in range(n - 1):
                ia0 = base + i
                ia1 = base + i + 1
                ib0 = base + n + i
                ib1 = base + n + i + 1
                fi += [ia0, ia0]
                fj += [ia1, ib1]
                fk += [ib1, ib0]
                fc += [t, t]
                for fk_out in _FLUX_OUT:
                    flux_cols[fk_out] += [flux_face[fk_out], flux_face[fk_out]]

    return verts, fi, fj, fk, fc, flux_cols


def build_stress_surface(rt, wing, lc: int = 0, end: str = "avg") -> dict:
    '''
    Build the analyzed-wing stress surface for one snapshot.

    Each T2 panel stores a full curved polyline (pts) following the actual
    airfoil skin geometry. This function lofts that polyline between adjacent
    spanwise stations (one per beam element), producing a ruled surface that
    hugs the real wing contour. Each strip is coloured by the panel shear
    stress tau. Boom rods coloured by normal stress sigma are included too.

    Args:
        rt   : RuntimeData snapshot.
        wing : WingData (typed) for the run.
        lc   : Load case index.

    Returns:
        Dict with mesh3d-ready curved panel surface, per-face tau intensity,
        and boom rods with per-node sigma values.
    '''
    sec = rt.sections.sec_data
    coord = np.asarray(rt.mesh.coord, float)
    conn = np.asarray(rt.mesh.conn, int)[:, :2]
    tauA_arr = np.asarray(rt.stress.tauA, float)        # (m, n_panels, nc)
    tauB_arr = np.asarray(rt.stress.tauB, float)
    sigA_arr = np.asarray(rt.stress.sigmaA, float)      # (m, n_booms, nc)
    sigB_arr = np.asarray(rt.stress.sigmaB, float)
    nc = tauA_arr.shape[2] if tauA_arr.ndim == 3 else 1
    lc = max(0, min(int(lc), nc - 1))

    # Select element-end value for stress: A-end, B-end, or average.
    def _pick_stress(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if end == "A":
            return A
        if end == "B":
            return B
        return 0.5 * (A + B)

    tau     = _pick_stress(tauA_arr, tauB_arr)
    sig_arr = _pick_stress(sigA_arr, sigB_arr)

    # Analyzed-wing elements, root -> tip (|Y| ascending; side-agnostic).
    mid_y = 0.5 * (coord[conn[:, 0], 1] + coord[conn[:, 1], 1])
    left_e = list(range(conn.shape[0]))
    left_e.sort(key=lambda e: abs(mid_y[e]))

    # -------- T2 panel surfaces coloured by shear stress --------
    # Each T2 panel stores a full curved polyline in the section global-XZ
    # frame (pts[:, 0] = x, pts[:, 1] = z). Lofting these polylines between
    # the two element end-stations produces a ruled surface that follows the
    # actual airfoil skin geometry, not a flat quad between boom endpoints.
    #
    # Pre-load local shear-centre internal forces (12, m, nc) so that each
    # influence coefficient can be scaled by the actual applied force.
    Qsc_arr = np.asarray(rt.fea_rts.Q_sc, float)
    nc_q    = Qsc_arr.shape[2] if Qsc_arr.ndim == 3 else 1
    lc_q    = max(0, min(int(lc), nc_q - 1))

    verts, fi, fj, fk, fc, flux_cols = _loft_stress_panels(
        sec, conn, left_e, tau, lc, end, Qsc_arr, lc_q
    )

    # Tau abs spans both A and B ends so the colorbar is stable across A/B/avg.
    tau_both = np.concatenate([tauA_arr[:, :, lc], tauB_arr[:, :, lc]])
    tau_fin  = tau_both[np.isfinite(tau_both)]
    tau_abs  = float(np.nanmax(np.abs(tau_fin))) if tau_fin.size else 1.0

    # Flux abs derived from the SAME accumulated per-face values used for display,
    # so the colorbar range always matches what is actually plotted.
    flux_abs: dict[str, float] = {}
    for fk_out in _FLUX_OUT:
        vals = np.array(flux_cols[fk_out], dtype=float)
        fin  = vals[np.isfinite(vals)]
        flux_abs[fk_out] = float(np.nanmax(np.abs(fin))) if fin.size else 1.0

    # -------- Boom rods coloured by normal stress sigma --------
    sig_both   = np.concatenate([sigA_arr[:, :, lc], sigB_arr[:, :, lc]])
    sig_fin    = sig_both[np.isfinite(sig_both)]
    sigma_abs_global = float(np.nanmax(np.abs(sig_fin))) if sig_fin.size else 1.0
    boom_rods, _ = _build_boom_rods(sig_arr, left_e, conn, sec, lc, nc)
    sigma_abs    = sigma_abs_global

    return {
        "vertices": np.asarray(verts, float) if verts else np.zeros((0, 3)),
        "i": fi, "j": fj, "k": fk,
        "intensity": fc,
        "tau_abs": tau_abs or 1.0,
        **{fk_out: flux_cols[fk_out] for fk_out in _FLUX_OUT},
        **{fk_out + "_abs": flux_abs[fk_out] for fk_out in _FLUX_OUT},
        "boom_rods": boom_rods,
        "sigma_abs": sigma_abs or 1.0,
        "n_elements": len(left_e),
        "n_loadcases": nc,
    }


def build_tsw_surface(rt, wing, lc: int = 0, end: str = "avg") -> dict:
    '''
    Build the Tsai-Wu strength-ratio (R) surface for one snapshot.

    Identical lofting geometry to build_stress_surface, but coloured by the
    Tsai-Wu strength ratio R (failure at R >= 1) instead of shear stress.

    Args:
        rt   : RuntimeData snapshot.
        wing : WingData (typed) for the run.
        lc   : Load case index.
        end  : "A" / "B" / "avg" -- element end selector.

    Returns:
        Dict with mesh3d vertices + triangle indices, per-face R intensity,
        boom rods coloured by boom R, and scalar limits.
    '''
    sec     = rt.sections.sec_data
    coord   = np.asarray(rt.mesh.coord, float)
    conn    = np.asarray(rt.mesh.conn, int)[:, :2]

    tsw = rt.tsw
    R_panels_arr = np.asarray(tsw.R_panels, float)  # (m, 10, 2, nc)
    R_booms_arr  = np.asarray(tsw.R_booms,  float)  # (m,  7, 2, nc)

    nc = R_panels_arr.shape[3] if R_panels_arr.ndim == 4 else 1
    lc = max(0, min(int(lc), nc - 1))

    end_idx = 0 if end == "A" else (1 if end == "B" else None)

    def _pick(arr: np.ndarray) -> np.ndarray:
        # arr shape: (m, n_items, 2, nc)
        if end_idx is not None:
            return arr[:, :, end_idx, lc]
        return 0.5 * (arr[:, :, 0, lc] + arr[:, :, 1, lc])

    R_p = _pick(R_panels_arr)   # (m, 10)
    R_b = _pick(R_booms_arr)    # (m,  7)

    mid_y  = 0.5 * (coord[conn[:, 0], 1] + coord[conn[:, 1], 1])
    left_e = sorted(range(conn.shape[0]), key=lambda e: abs(mid_y[e]))

    verts: list = []
    fi, fj, fk, fc = [], [], [], []

    for e in left_e:
        node_a, node_b = int(conn[e, 0]), int(conn[e, 1])
        y_a = float(sec[node_a].C[1])
        y_b = float(sec[node_b].C[1])
        n_panels = min(len(sec[node_a].T2), len(sec[node_b].T2))

        for jp in range(n_panels):
            pts_a = np.asarray(sec[node_a].T2[jp]["pts"], float)
            pts_b = np.asarray(sec[node_b].T2[jp]["pts"], float)
            r_val = float(R_p[e, jp]) if jp < R_p.shape[1] else np.nan

            na, nb_pts = pts_a.shape[0], pts_b.shape[0]
            n = min(na, nb_pts)
            if na != n:
                idx = np.round(np.linspace(0, na - 1, n)).astype(int)
                pts_a = pts_a[idx]
            if nb_pts != n:
                idx = np.round(np.linspace(0, nb_pts - 1, n)).astype(int)
                pts_b = pts_b[idx]

            base = len(verts)
            for i in range(n):
                verts.append([pts_a[i, 0], y_a, pts_a[i, 1]])
            for i in range(n):
                verts.append([pts_b[i, 0], y_b, pts_b[i, 1]])

            for i in range(n - 1):
                ia0, ia1 = base + i, base + i + 1
                ib0, ib1 = base + n + i, base + n + i + 1
                fi += [ia0, ia0]
                fj += [ia1, ib1]
                fk += [ib1, ib0]
                fc += [r_val, r_val]

    r_fin = np.array([v for v in fc if np.isfinite(v)], dtype=float)
    r_max = float(np.nanmax(r_fin)) if r_fin.size else 2.0
    r_min = float(np.nanmin(r_fin)) if r_fin.size else 0.0

    # Boom rods coloured by R (reuse _build_boom_rods with R_b as the "sigma" arg,
    # but add an extra nc-dim so the helper's indexing works).
    R_b_nc = R_b[:, :, np.newaxis]   # (m, 7, 1)  -- fake nc=1 dimension
    boom_rods, r_abs_booms = _build_boom_rods(R_b_nc, left_e, conn, sec, 0, 1)

    return {
        "vertices": np.asarray(verts, float) if verts else np.zeros((0, 3)),
        "i": fi, "j": fj, "k": fk,
        "intensity": fc,
        "r_max":     r_max,
        "r_min":     r_min,
        "boom_rods": boom_rods,
        "r_abs_booms": r_abs_booms,
        "n_elements":  len(left_e),
        "n_loadcases": nc,
    }


def _build_boom_rods(
    sig_arr: np.ndarray,
    left_e: list,
    conn: np.ndarray,
    sec: list,
    lc: int,
    nc: int,
) -> tuple[list[dict], float]:
    '''Per-boom polylines coloured by normal stress sigma along the span.

    sig_arr : pre-picked (m, nb, nc) array (A-end, B-end, or avg).
    Returns (boom_rods, sigma_abs) where boom_rods is a list of dicts:
        {"xyz": [[x,y,z], ...], "sigma": [s0, s1, ...], "label": "B1"}
    and sigma_abs is max |sigma| across all booms (for the colourbar).
    '''
    sig_arr = np.asarray(sig_arr, float)
    if sig_arr.ndim < 3:
        return [], 1.0
    n_lc = sig_arr.shape[2]
    lc_s = max(0, min(int(lc), n_lc - 1))
    nb = sig_arr.shape[1]

    # Accumulate sigma contributions per node.
    node_sig_sum: dict[int, np.ndarray] = {}
    node_sig_cnt: dict[int, int] = {}
    for e in left_e:
        na, nb_e = int(conn[e, 0]), int(conn[e, 1])
        for n in (na, nb_e):
            if n not in node_sig_sum:
                node_sig_sum[n] = np.zeros(nb)
                node_sig_cnt[n] = 0
            node_sig_sum[n] += sig_arr[e, :, lc_s]
            node_sig_cnt[n] += 1

    used_nodes = sorted(node_sig_sum.keys())
    if not used_nodes:
        return [], 1.0

    node_sigma = {n: node_sig_sum[n] / node_sig_cnt[n] for n in used_nodes}

    boom_lbls = list(getattr(sec[used_nodes[0]], "boom_lbls",
                             [f"B{i+1}" for i in range(nb)]))

    boom_rods = []
    for b in range(nb):
        xyz, sigs = [], []
        for n in used_nodes:
            gd = sec[n]
            bx = float(gd.boom_Xc) + float(gd.boom_u[b])
            bz = float(gd.boom_Zc) + float(gd.boom_w[b])
            y = float(gd.C[1])
            xyz.append([bx, y, bz])
            sigs.append(float(node_sigma[n][b]))
        label = str(boom_lbls[b]) if b < len(boom_lbls) else f"B{b+1}"
        boom_rods.append({"xyz": xyz, "sigma": sigs, "label": label})

    all_sig = np.array([s for rod in boom_rods for s in rod["sigma"]])
    fin = all_sig[np.isfinite(all_sig)]
    sigma_abs = float(np.nanmax(np.abs(fin))) if fin.size else 1.0
    return boom_rods, sigma_abs


# ========================================================================
# PRIVATE API - structural overlays
# ========================================================================

def _place_uw(u: float, w: float, xle: float, zle: float,
              chord: float, twist: float) -> tuple[float, float]:
    '''Map an adimensional (u, w) chord-fraction point to global (x, z).'''
    xs, zs = u * chord, w * chord
    if abs(twist) > 1e-12:
        ca, sa = np.cos(twist), np.sin(twist)
        xs, zs = ca * xs - sa * zs, sa * xs + ca * zs
    return xs + xle, zs + zle


def _sec_le_chord(gd) -> tuple[float, float, float]:
    '''Leading-edge (x, z) and chord of a section, in its own stored frame.'''
    pts = np.vstack([np.asarray(s["pts"], float) for s in gd.T1])
    le = pts[int(np.argmin(pts[:, 0]))]
    return float(le[0]), float(le[1]), float(gd.chord)


def _spar_strip(rt, left, seg_label, dmat, LE, chord, twist, lrow, scale, lc) -> dict:
    '''2 x n_s quad strip for a spar web (seg6 front / seg7 rear).

    T1 pts are already in the global XZ frame (twist + chord scale applied by
    the section builder), so they are used directly without re-normalization.
    Normalizing and then calling _place_uw would double-apply the twist.
    '''
    sec = rt.sections.sec_data
    n_s = len(left)
    strip = np.zeros((2, n_s, 3))
    for s, node_i in enumerate(left):
        gd = sec[node_i]
        seg = next((p for p in gd.T1 if p["label"] == seg_label), None)
        if seg is None:
            raise ValueError(
                f"[CL3O] Spar web segment not found in section T1 topology.\n"
                f"| Label   : {seg_label}\n"
                f"| Station : {node_i}\n"
                f"| Present : {[p.get('label') for p in gd.T1]}\n"
                f"Check the seg_label argument or rebuild the section geometry."
            )
        pts = np.asarray(seg["pts"], float)          # (2, 2): col 0 = X, col 1 = Z
        r = lrow[s]
        xle, zle, y = float(LE[r, 0]), float(LE[r, 2]), float(gd.C[1])
        row = np.zeros((2, 3))
        for e, pi in enumerate((0, -1)):
            row[e] = (pts[pi, 0], y, pts[pi, 1])    # global XZ -> 3-D XYZ
        if dmat is not None:
            row = _apply_disp(row, np.array([xle, y, zle]), dmat[:, node_i, lc], scale)
        strip[:, s] = row
    flat = strip.reshape(-1, 3, order="F")          # vertex idx = s*2 + r
    i, j, k = _grid_faces(2, n_s)
    return {"vertices": flat, "i": i, "j": j, "k": k}


def _flange_strips(rt, left, dmat, LE, chord, lrow, scale, lc, optvars) -> list[dict]:
    '''
    Build one mesh3d-ready strip per structural flange (F1..F4).

    Each flange is a thin ribbon centered on its anchor boom (per
    Constants.FLANGE_BOOM_IDX), extending +/- bf/2 in the chord
    direction with bf taken from the OptVars root width. Layup index
    is the root value of lfK so the frontend can color-code by family.
    '''
    from cl3o.Constants import FLANGE_BOOM_IDX
    sec = rt.sections.sec_data
    n_s = len(left)
    if optvars is None or n_s == 0:
        return []


    def _root(arr, default=0.0):
        a = np.asarray(arr, float).ravel()
        return float(a[0]) if a.size else float(default)

    bf_roots = [
        _root(getattr(optvars, "bf1_root", 0.0)),
        _root(getattr(optvars, "bf2_root", 0.0)),
        _root(getattr(optvars, "bf3_root", 0.0)),
        _root(getattr(optvars, "bf4_root", 0.0)),
    ]
    lf_roots = [
        int(round(_root(getattr(optvars, f"lf{k}", [0]))))
        for k in range(1, 5)
    ]
    out: list[dict] = []
    for fk, boom_idx in enumerate(FLANGE_BOOM_IDX):
        strip = np.zeros((2, n_s, 3))
        for s, node_i in enumerate(left):
            gd = sec[node_i]
            # Canonical boom position: boom_u/boom_w are stored relative to
            # the section centroid (boom_Xc/boom_Zc). T2 panel endpoints are
            # unreliable here because some panels collect TE/LE intermediate
            # waypoints under the same boomA/boomB tag.
            try:
                xb = float(np.asarray(gd.boom_u, float)[boom_idx]
                           + float(gd.boom_Xc))
                zb = float(np.asarray(gd.boom_w, float)[boom_idx]
                           + float(gd.boom_Zc))
            except Exception:
                continue
            r = lrow[s]
            cw = float(chord[r])
            xle, zle, y = float(LE[r, 0]), float(LE[r, 2]), float(gd.C[1])
            half = 0.5 * bf_roots[fk] * cw
            # Symmetric extension: ±half along the chord around the boom.
            x0, x1 = xb - half, xb + half
            row = np.array([[x0, y, zb], [x1, y, zb]])
            if dmat is not None:
                row = _apply_disp(
                    row, np.array([xle, y, zle]), dmat[:, node_i, lc], scale,
                )
            strip[:, s] = row
        flat = strip.reshape(-1, 3, order="F")
        i, j, k_ = _grid_faces(2, n_s)
        out.append({
            "vertices" : flat,
            "i"        : i, "j": j, "k": k_,
            "layup_idx": lf_roots[fk],
            "label"    : f"F{fk + 1}",
        })
    return out


def _line(rt, left, attr, dmat, LE, chord, twist, lrow, scale, lc) -> np.ndarray:
    '''Per-station polyline of a structural point (C or S_XYZ), expressed as a
    chord fraction of its section and re-placed on the wing frame.'''
    sec = rt.sections.sec_data
    n_s = len(left)
    out = np.zeros((n_s, 3))
    for s, node_i in enumerate(left):
        gd = sec[node_i]
        p = np.asarray(getattr(gd, attr), float)
        xle_s, zle_s, c_s = _sec_le_chord(gd)
        r = lrow[s]
        cw, tw = float(chord[r]), float(twist[r])
        xle, zle, y = float(LE[r, 0]), float(LE[r, 2]), float(p[1])
        u = (p[0] - xle_s) / c_s
        w = (p[2] - zle_s) / c_s
        X, Z = _place_uw(u, w, xle, zle, cw, tw)
        out[s] = (X, y, Z)
        if dmat is not None:
            out[s] = _apply_disp(out[s][None, :], np.array([xle, y, zle]),
                                 dmat[:, node_i, lc], scale)[0]
    return out
