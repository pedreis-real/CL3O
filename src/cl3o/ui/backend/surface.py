'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Wing-Surface Builder Module.

Reconstructs the 3-D LEFT-wing visualization scene from an archived
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
_Y_LEFT_TOL = 1.0e-6          # stations with C[1] <= tol belong to the left wing
_N_CHORD    = 81              # airfoil-loop points per rib (kept light for WebGL)


# ========================================================================
# PRIVATE API - geometry primitives
# ========================================================================

def _skin_rib(gd) -> np.ndarray:
    '''Outer skin contour for one section from T1 seg1..seg5, in global XZ.

    T1 outer segments in connectivity order (seg1 = LE cell, seg2..5 = mid/TE):
        seg1: B5 → (LE) → B3   lower-front skin wrapping around the nose
        seg2: B3 → B1           upper mid skin
        seg3: B1 → TE           upper TE skin
        seg4: TE → B7           lower TE skin
        seg5: B7 → B5           lower mid skin

    All pts are already in global XZ (LE_xz offset applied by section builder).
    Sub-sampled to at most _N_CHORD points for WebGL budget.

    Returns:
        (N, 2) float array, columns [x, z].
    '''
    segs = {s["label"]: np.asarray(s["pts"], float) for s in gd.T1}
    parts = []
    for k, lbl in enumerate(("seg1", "seg2", "seg3", "seg4", "seg5")):
        p = segs.get(lbl)
        if p is None:
            continue
        parts.append(p if k == 0 else p[1:])   # skip shared endpoint
    if not parts:
        return np.zeros((0, 2))
    loop = np.vstack(parts)
    if loop.shape[0] > _N_CHORD:
        idx = np.unique(np.linspace(0, loop.shape[0] - 1, _N_CHORD).round().astype(int))
        loop = loop[idx]
    return loop


def _left_stations(rt) -> list[int]:
    '''Section indices of the left wing, ordered root -> tip (|Y| ascending).'''
    sec = rt.sections.sec_data
    left = [i for i, g in enumerate(sec) if float(g.C[1]) <= _Y_LEFT_TOL]
    left.sort(key=lambda i: abs(float(sec[i].C[1])))
    return left


def _grid_faces(n_c: int, n_s: int) -> tuple[list, list, list]:
    '''Triangulate a structured (n_c chord x n_s span) grid; vertex = s*n_c+c.'''
    i, j, k = [], [], []
    for s in range(n_s - 1):
        for c in range(n_c - 1):
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

def build_scene(
    rt,
    wing,
    afl,
    lc    : int = 0,
    scale : float = 1.0,
    deform: bool = False,
) -> dict:
    '''
    Build the left-wing 3-D scene for one snapshot.

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
    left = _left_stations(rt)
    n_s = len(left)
    y_left = np.array([float(sec[i].C[1]) for i in left])

    # Per-station LE / chord / twist via the production left-wing lerp.
    lerp = WingHelper.lerp_from_data(wing, y_left)
    ly = np.asarray(lerp.Y_sta, dtype=float)
    LE = np.asarray(lerp.LE, dtype=float)
    chord = np.asarray(lerp.chord, dtype=float)
    twist = np.asarray(lerp.twist, dtype=float)
    # Map each left station Y -> lerp row (nearest) for displacement reference.
    lrow = [int(np.argmin(np.abs(ly - y))) for y in y_left]

    # Build per-station outer skin ribs from T1 pts (seg1..seg5 in global XZ).
    # These use the actual blended airfoil so they align with section panel data.
    raw_ribs = [_skin_rib(sec[node_i]) for node_i in left]
    n_c = min(r.shape[0] for r in raw_ribs if r.shape[0] > 0) or _N_CHORD

    dmat = getattr(getattr(rt, "fea_rts", None), "dmatrix", None)
    if deform and dmat is not None:
        dmat = np.asarray(dmat, float)
        nc_lc = dmat.shape[2]
        lc = max(0, min(int(lc), nc_lc - 1))
    else:
        dmat = None

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
            disp["u"][s], disp["v"][s], disp["w"][s] = d[0], d[1], d[2]
            disp["rx"][s], disp["ry"][s], disp["rz"][s] = d[3], d[4], d[5]
            disp["t"][s] = float(np.linalg.norm(d[0:3]))
            disp["r"][s] = float(np.linalg.norm(d[3:6]))
            rib = _apply_disp(rib, np.array([xle, y, zle]), d, scale)
        verts[:, s] = rib

    flat = verts.reshape(-1, 3, order="F")          # vertex idx = s*n_c + c
    i, j, k = _grid_faces(n_c, n_s)
    surface = {"vertices": flat, "i": i, "j": j, "k": k, "n_chord": n_c, "n_span": n_s}
    if dmat is not None:
        surface["station_disp"] = {key: disp[key] for key in comp_keys}

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

def build_stress_surface(rt, wing, lc: int = 0, end: str = "avg") -> dict:
    '''
    Build the left-wing stress surface for one snapshot.

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

    # Left-wing elements, root -> tip.
    mid_y = 0.5 * (coord[conn[:, 0], 1] + coord[conn[:, 1], 1])
    left_e = [e for e in range(conn.shape[0]) if mid_y[e] <= _Y_LEFT_TOL]
    left_e.sort(key=lambda e: abs(mid_y[e]))

    # -------- T2 panel surfaces coloured by shear stress --------
    # Each T2 panel stores a full curved polyline in the section global-XZ
    # frame (pts[:, 0] = x, pts[:, 1] = z). Lofting these polylines between
    # the two element end-stations produces a ruled surface that follows the
    # actual airfoil skin geometry, not a flat quad between boom endpoints.
    _FLUX_KEYS = ("qsX_star", "qsZ_star", "qT_star", "qbX_star", "qbZ_star")
    _FLUX_OUT  = ("flux_qsX", "flux_qsZ", "flux_qT", "flux_qbX", "flux_qbZ")

    verts: list = []
    fi, fj, fk, fc = [], [], [], []
    flux_cols: dict[str, list] = {k: [] for k in _FLUX_OUT}

    for e in left_e:
        node_a, node_b = int(conn[e, 0]), int(conn[e, 1])
        y_a = float(sec[node_a].C[1])
        y_b = float(sec[node_b].C[1])
        n_panels_a = len(sec[node_a].T2)
        n_panels_b = len(sec[node_b].T2)
        for jp in range(min(n_panels_a, n_panels_b)):
            pts_a = np.asarray(sec[node_a].T2[jp]["pts"], float)  # (na, 2)
            pts_b = np.asarray(sec[node_b].T2[jp]["pts"], float)  # (nb, 2)
            t = float(tau[e, jp, lc])

            # Select flux over the two end-stations.
            flux_face: dict[str, float] = {}
            for fk_in, fk_out in zip(_FLUX_KEYS, _FLUX_OUT):
                va = float(np.asarray(getattr(sec[node_a], fk_in, np.zeros(10)))[jp])
                vb = float(np.asarray(getattr(sec[node_b], fk_in, np.zeros(10)))[jp])
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

    # Limits always span both ends so the colorbar stays fixed across A/B/avg.
    tau_both = np.concatenate([tauA_arr[:, :, lc], tauB_arr[:, :, lc]])
    tau_fin  = tau_both[np.isfinite(tau_both)]
    tau_abs  = float(np.nanmax(np.abs(tau_fin))) if tau_fin.size else 1.0

    flux_abs: dict[str, float] = {}
    for fk_in, fk_out in zip(_FLUX_KEYS, _FLUX_OUT):
        vals_a = np.array([
            float(np.asarray(getattr(sec[int(conn[e, 0])], fk_in, np.zeros(10)))[jp])
            for e in left_e
            for jp in range(min(len(sec[int(conn[e, 0])].T2), len(sec[int(conn[e, 1])].T2)))
        ], dtype=float)
        vals_b = np.array([
            float(np.asarray(getattr(sec[int(conn[e, 1])], fk_in, np.zeros(10)))[jp])
            for e in left_e
            for jp in range(min(len(sec[int(conn[e, 0])].T2), len(sec[int(conn[e, 1])].T2)))
        ], dtype=float)
        combined = np.concatenate([vals_a, vals_b])
        flux_abs[fk_out] = float(np.nanmax(np.abs(combined))) if combined.size else 1.0

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
        seg = next(p for p in gd.T1 if p["label"] == seg_label)
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
