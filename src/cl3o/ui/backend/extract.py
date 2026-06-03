'''
================================================================================
CL3O Visualization UI - View extractors.

Pure functions turning archived data into JSON-ready dicts, one per UI view:
    planform(wing)             -> wing outer geometry (from the wing DB cpts)
    section(rt, i)             -> 3-cell idealized cross-section at station i
    mesh(rt, lc, deformed)     -> beam node/element mesh (+ deformed nodes)
    stress(rt, lc)             -> per-panel shear stress field
    info(rt)                   -> scalar summary for the sidebar

Numpy arrays are returned as-is; the app layer runs them through
serialize.to_jsonable (numpy -> list, NaN/Inf -> null).

@ CL3O Authors - MIT License
================================================================================
'''

import math

import numpy as np


def _f(v):
    '''float or None (for non-finite / missing).'''
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _g(obj, name, default=None):
    return getattr(obj, name, default)


# ------------------------------------------------------------------
# Misc - search space trajectory
# ------------------------------------------------------------------

def search_space(distinct_snapshots: list, manifest: dict) -> dict:
    '''
    Build a 2-D trajectory of distinct individuals through the design
    space, plus per-distinct statistical metrics.

    Approach: stack each distinct individual's flat design vector (read
    via cl3o.optimization.fobjective.BuildEvaluator.encode_optvars from
    its RuntimeData.optvars), centre, and project on the first two
    principal components via SVD. The resulting (x, y) coordinates trace
    the path the DE walker took through the search space, and the
    explained-variance ratios advertise how meaningful the projection is.

    Args:
        distinct_snapshots: list of (record_dict, RuntimeData) for each
            distinct individual, in manifest order.
        manifest: parsed manifest (best_f_hist, mean_f_hist, std_f_hist).

    Returns dict ready for serialize.to_jsonable.
    '''
    from cl3o.optimization.fobjective import BuildEvaluator

    Xs: list[np.ndarray] = []
    f:  list[float] = []
    gen: list[int]  = []
    feas: list[bool] = []
    for rec, rt in distinct_snapshots:
        ov = getattr(rt, "optvars", None)
        if ov is None:
            continue
        try:
            Xs.append(BuildEvaluator.encode_optvars(ov))
        except Exception:
            continue
        f.append(_f(rec.get("best_f")) or float("nan"))
        gen.append(int(rec.get("first_seen_gen", rec.get("k", 0))))
        feas.append(bool(rec.get("is_feasible", False)))

    if len(Xs) < 2:
        return {
            "x": [], "y": [], "z": [], "f": f, "gen": gen, "feasible": feas,
            "explained_variance": [0.0, 0.0, 0.0], "n_distinct": len(Xs),
            "metrics": _global_metrics(manifest),
        }

    X = np.vstack(Xs).astype(float)
    mu = X.mean(axis=0)
    Xc = X - mu
    # Compact SVD for PCA. U·S gives the projected coordinates.
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    n_pcs = min(3, U.shape[1])
    coords = U[:, :n_pcs] * S[:n_pcs]
    total = float((S ** 2).sum()) or 1.0
    ev = [float((S[i] ** 2) / total) if i < S.size else 0.0 for i in range(3)]

    # Cumulative L2 path length along the trajectory: how far DE travelled.
    dpath = np.linalg.norm(np.diff(X, axis=0), axis=1)
    cum_path = np.concatenate(([0.0], np.cumsum(dpath))).tolist()

    return {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist() if n_pcs > 1 else [0.0] * len(Xs),
        "z": coords[:, 2].tolist() if n_pcs > 2 else [0.0] * len(Xs),
        "f": f,
        "gen": gen,
        "feasible": feas,
        "cum_path": cum_path,
        "explained_variance": ev,
        "n_distinct": len(Xs),
        "metrics": _global_metrics(manifest),
    }


def _global_metrics(manifest: dict) -> dict:
    '''Manifest-level statistics: convergence rate, mean improvement, etc.'''
    best = np.asarray(manifest.get("best_f_hist") or [], dtype=float)
    mean = np.asarray(manifest.get("mean_f_hist") or [], dtype=float)
    std  = np.asarray(manifest.get("std_f_hist")  or [], dtype=float)
    out = {
        "n_gens"    : int(manifest.get("n_gens", 0)),
        "best_f"    : _f(best[-1] if best.size else None),
        "best_gen"  : int(np.argmin(best)) if best.size else 0,
        "mean_f"    : _f(mean[-1] if mean.size else None),
        "std_f"     : _f(std[-1]  if std.size  else None),
    }
    if best.size > 1:
        finite = best[np.isfinite(best)]
        if finite.size > 1:
            out["total_improvement"] = _f(float(finite[0] - finite.min()))
            out["mean_improvement"]  = _f(
                float((finite[0] - finite.min()) / max(1, finite.size - 1))
            )
    return out


# ------------------------------------------------------------------
# Geometry - wing planform (from the wing database control points)
# ------------------------------------------------------------------

def planform(wing: dict) -> dict:
    '''Full-span LE/TE polylines built from the half-wing control points.'''
    pos    = np.asarray(wing["pos"],    float)   # spanwise cpt positions (>=0)
    x_le   = np.asarray(wing["x_le"],   float)
    z_le   = np.asarray(wing["z_le"],   float)
    x_te   = np.asarray(wing["x_te"],   float)
    z_te   = np.asarray(wing["z_te"],   float)
    chords = np.asarray(wing["chords"], float)

    le_half = np.column_stack([x_le, pos, z_le])     # (ncpt, 3): x, y, z
    te_half = np.column_stack([x_te, pos, z_te])

    def mirror(arr):
        m = arr.copy()
        m[:, 1] *= -1.0
        return m[::-1]                                # left tip -> root

    le = np.vstack([mirror(le_half), le_half])
    te = np.vstack([mirror(te_half), te_half])

    return {
        "le": le, "te": te,
        "pos": pos, "chords": chords,
        "span":       _f(wing.get("b")),
        "area":       _f(wing.get("area")),
        "AR":         _f(wing.get("AR")),
        "mac":        _f(wing.get("mac")),
        "root_chord": _f(chords[0])  if chords.size else None,
        "tip_chord":  _f(chords[-1]) if chords.size else None,
    }


# ------------------------------------------------------------------
# Cross-section - 3-cell idealization at a station
# ------------------------------------------------------------------

def section(rt, i: int) -> dict:
    '''Panels (T2), closed cells (T3), skin segments (T1) and booms at station i.

    All `pts` are 2-D in the section's local chord frame (LE at origin), so
    no global transform is needed to draw them.
    '''
    secs = rt.sections.sec_data
    n = len(secs)
    i = max(0, min(int(i), n - 1))
    sec = secs[i]

    panels = [{
        "label": p.get("label"),
        "pts":   np.asarray(p["pts"], float),
        "t":     _f(p.get("t")),
        "boomA": int(p.get("boomA", -1)),
        "boomB": int(p.get("boomB", -1)),
    } for p in _g(sec, "T2", [])]

    cells = [{"label": c.get("label"), "pts": np.asarray(c["pts"], float)}
             for c in _g(sec, "T3", [])]

    skin = [{"label": s.get("label"), "pts": np.asarray(s["pts"], float),
             "t": _f(s.get("t"))} for s in _g(sec, "T1", [])]

    # Boom (x,z) in the same local frame: each T2 panel runs boomA -> boomB,
    # so panel endpoints land exactly on the booms they connect.
    boom_A = np.asarray(_g(sec, "boom_A", np.zeros(0)), float)
    nb = int(boom_A.size) or 7
    boom_xy = np.full((nb, 2), np.nan)
    for p in _g(sec, "T2", []):
        pts = np.asarray(p["pts"], float)
        a, b = int(p.get("boomA", -1)), int(p.get("boomB", -1))
        if 0 <= a < nb:
            boom_xy[a] = pts[0]
        if 0 <= b < nb:
            boom_xy[b] = pts[-1]

    s_xyz = np.asarray(_g(sec, "S_XYZ", np.zeros(3)), float)
    return {
        "station":      i,
        "n_stations":   n,
        "chord":        _f(_g(sec, "chord")),
        "y":            _f(_g(sec, "C", [0, 0, 0])[1]),
        "panels":       panels,
        "cells":        cells,
        "skin":         skin,
        "booms":        {"xy": boom_xy, "A": boom_A,
                         "labels": list(_g(sec, "boom_lbls", []))},
        "centroid":     [_f(_g(sec, "boom_Xc", 0.0)), _f(_g(sec, "boom_Zc", 0.0))],
        "shear_centre": [_f(s_xyz[0]) if s_xyz.size > 0 else None,
                         _f(s_xyz[2]) if s_xyz.size > 2 else None],
        "props": {
            "area":    _f(_g(sec, "A")),
            "I_XX":    _f(_g(sec, "I_XX")),
            "I_ZZ":    _f(_g(sec, "I_ZZ")),
            "I_XZ":    _f(_g(sec, "I_XZ")),
            "c_rad":   _f(_g(sec, "c_rad")),
            "J":       _f(_g(sec, "J")),
            "A_cells": np.asarray(_g(sec, "A_cells", np.zeros(0)), float),
            "xw1":     _f(_g(sec, "xw1")),
            "xw2":     _f(_g(sec, "xw2")),
        },
        "fluxes": {
            "qsX":  np.asarray(_g(sec, "qsX_star",  np.zeros(10)), float),
            "qsZ":  np.asarray(_g(sec, "qsZ_star",  np.zeros(10)), float),
            "qT":   np.asarray(_g(sec, "qT_star",   np.zeros(10)), float),
            "qbX":  np.asarray(_g(sec, "qbX_star",  np.zeros(10)), float),
            "qbZ":  np.asarray(_g(sec, "qbZ_star",  np.zeros(10)), float),
            "qs0X": np.asarray(_g(sec, "qs0X_star", np.zeros(3)),  float),
            "qs0Z": np.asarray(_g(sec, "qs0Z_star", np.zeros(3)),  float),
            "qs0T": np.asarray(_g(sec, "qs0T_star", np.zeros(3)),  float),
        },
    }


# ------------------------------------------------------------------
# Mesh - beam nodes / elements (+ deformed)
# ------------------------------------------------------------------

def mesh(rt, lc: int = 0, deformed: bool = False, scale: float = 1.0) -> dict:
    '''Beam mesh: node coords, element node-pairs, and (optionally) the
    deformed node positions for load case `lc` scaled by `scale`.'''
    coord = np.asarray(rt.mesh.coord, float)            # (n, 3)
    conn  = np.asarray(rt.mesh.conn, int)[:, :2]        # (m, 2)
    nc    = int(_g(rt.mesh, "nc", 1) or 1)
    lc    = max(0, min(int(lc), nc - 1))

    out = {
        "nodes":       coord,
        "elements":    conn,
        "n_loadcases": nc,
        "n_nodes":     int(coord.shape[0]),
        "n_elements":  int(conn.shape[0]),
    }

    dmat = _g(_g(rt, "fea_rts"), "dmatrix")
    if dmat is not None:
        dmat = np.asarray(dmat, float)                  # (6, n, nc)
        defl = dmat[0:3, :, lc].T                        # (n, 3)
        mag  = np.linalg.norm(defl, axis=1)
        out["displacement"] = defl
        out["deformed"]     = coord + float(scale) * defl
        out["max_disp"]     = _f(np.nanmax(mag)) if mag.size else 0.0
    return out


# ------------------------------------------------------------------
# Stress - per-panel shear field
# ------------------------------------------------------------------

def stress(rt, lc: int = 0, end: str = "avg") -> dict:
    '''Per-element x per-panel shear stress (MPa) for load case `lc`.

    `end` in {"A", "B", "avg"} selects the element start/end section or
    their mean. Also returns a per-element scalar (max |tau| across panels)
    for colouring the beam, and element midpoints for placement.
    '''
    s = rt.stress
    tauA = np.asarray(s.tauA, float)                    # (m, 10, nc)
    tauB = np.asarray(s.tauB, float)
    nc   = tauA.shape[2] if tauA.ndim == 3 else 1
    lc   = max(0, min(int(lc), nc - 1))

    A, B = tauA[:, :, lc], tauB[:, :, lc]
    tau  = A if end == "A" else B if end == "B" else 0.5 * (A + B)

    q = None
    qA = _g(s, "qA")
    if qA is not None:
        qA = np.asarray(qA, float)[:, :, lc]
        qB = np.asarray(s.qB, float)[:, :, lc]
        q = 0.5 * (qA + qB)

    coord    = np.asarray(rt.mesh.coord, float)
    conn     = np.asarray(rt.mesh.conn, int)[:, :2]
    elem_mid = 0.5 * (coord[conn[:, 0]] + coord[conn[:, 1]])
    labels   = [p.get("label", f"p{j}")
                for j, p in enumerate(_g(rt.sections.sec_data[0], "T2", []))]

    return {
        "tau":          tau,
        "q":            q,
        "elem_scalar":  np.nanmax(np.abs(tau), axis=1),
        "elem_mid":     elem_mid,
        "panel_labels": labels,
        "end":          end,
        "min":          _f(np.nanmin(tau)) if tau.size else 0.0,
        "max":          _f(np.nanmax(tau)) if tau.size else 0.0,
        "n_loadcases":  nc,
        "n_elements":   int(tau.shape[0]),
        "n_panels":     int(tau.shape[1]),
    }


# ------------------------------------------------------------------
# Internal forces - end-A resultants along the analyzed wing
# ------------------------------------------------------------------

def forces(rt, lc: int = 0) -> dict:
    '''Per-element end-A internal forces along the analyzed wing (root -> tip).

    Returns local (Q_sc) and global (Q_gl) resultants per component as
    beam-diagram series against spanwise distance |Y| from the root.

    Shear centre internal loads (Qsc_l and Qsc_g) are used to extract
    loads that produces shear stresses (Sy, Sz, T and SX, SZ);
    Centroidal internal loads (Qc_l and Qc_g) are used to extract
    loads that produces normal stresses (N, My, Mz and MX, MZ);
    '''
    fr = rt.fea_rts
    Qsc_l = np.asarray(fr.Q_sc,    float)            # (12, m, nc) shear-centre local
    Qsc_g = np.asarray(fr.Q_sc_gl, float)            # shear-centre global
    Qc_l  = np.asarray(fr.Q_c,     float)            # centroidal local
    Qc_g  = np.asarray(fr.Q_c_gl,  float)            # centroidal global
    nc = Qsc_l.shape[2] if Qsc_l.ndim == 3 else 1
    lc = max(0, min(int(lc), nc - 1))

    coord = np.asarray(rt.mesh.coord, float)
    conn  = np.asarray(rt.mesh.conn,  int)[:, :2]
    mid_y = 0.5 * (coord[conn[:, 0], 1] + coord[conn[:, 1], 1])
    # A snapshot holds only the analyzed half-span, so order all elements by
    # |Y| (root -> tip); side-agnostic for the right (Y>0) or left (Y<0) wing.
    left  = list(range(conn.shape[0]))
    left.sort(key=lambda e: abs(mid_y[e]))

    def s(Q, row):
        return [_f(Q[row, e, lc]) for e in left]

    local = {
        "N":  s(Qc_l,  0),
        "Sy": s(Qsc_l, 1),
        "Sz": s(Qsc_l, 2),
        "T":  s(Qsc_l, 3),
        "My": s(Qc_l,  4),
        "Mz": s(Qc_l,  5),
    }
    global_ = {
        "N":  s(Qc_g,  1),
        "SX": s(Qsc_g, 0),
        "SZ": s(Qsc_g, 2),
        "T":  s(Qsc_g, 4),
        "MX": s(Qc_g,  3),
        "MZ": s(Qc_g,  5),
    }
    return {
        "span":        [abs(float(mid_y[e])) for e in left],
        "local":       local,
        "global":      global_,
        "components":  list(local) + [k for k in global_ if k not in local],
        "n_loadcases": nc,
        "units": {
            "N": "N", "Sy": "N", "Sz": "N", "T": "N*mm",
            "My": "N*mm", "Mz": "N*mm",
            "SX": "N", "SZ": "N", "MX": "N*mm", "MZ": "N*mm",
        },
    }


# ------------------------------------------------------------------
# Sidebar - scalar summary
# ------------------------------------------------------------------

def info(rt) -> dict:
    '''Scalar fitness / failure / displacement / mass summary.'''
    fit = _g(rt, "fitness")
    tsw = _g(rt, "tsw")
    dsp = _g(rt, "displ")
    sco = _g(rt, "score")

    X_vec = None
    ov = _g(rt, "optvars")
    if ov is not None:
        try:
            from cl3o.optimization.fobjective import BuildEvaluator
            X_vec = BuildEvaluator.encode_optvars(ov).tolist()
        except Exception:
            pass

    return {
        "fitness": {
            "score":       _f(_g(fit, "score")),
            "penalty":     _f(_g(fit, "penalty")),
            "total":       _f(_g(fit, "total")),
            "is_feasible": bool(_g(fit, "is_feasible", False)),
        },
        "tsw": {
            "MS_min":       _f(_g(tsw, "MS_min")),
            "R_min":        _f(_g(tsw, "R_min")),
            "n_violations": int(_g(tsw, "nv", 0) or 0),
        },
        "displacement": {
            "MS_min":       _f(_g(dsp, "MS_min")),
            "n_violations": int(_g(dsp, "nv", 0) or 0),
        },
        "mass": {
            "total":   _f(_g(sco, "total")),
            "panels":  _f(np.nansum(np.asarray(_g(sco, "panels", [])))) if _g(sco, "panels") is not None else None,
            "flanges": _f(np.nansum(np.asarray(_g(sco, "flanges", [])))) if _g(sco, "flanges") is not None else None,
        },
        "optvars": X_vec,
    }
