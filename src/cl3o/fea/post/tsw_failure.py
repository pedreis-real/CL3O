'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Tsai-Wu Failure Module.

Evaluates the Tsai-Wu failure criterion ply-by-ply for every laminated
component (panel or flange) in every element/station and stores in
FailureData.

The computed margin of safety drives the optimization procedure, giving
a "penalty" value attached to the fitness of individuals which R, for any
ply, is less than one unity.

Pipeline
----------------
    1. Extract the stress state of the component:
        * booms  .... sigma_off_axis = {sigma, 0, 0}
        * panels .... sigma_off_axis = {0, 0, tau}
    2. Loop for every ply:
            >>> for k in rt.sections.lam_idx: # <- lam_T1 or lam_T4
            >>>     for ply in st.laminate_db[f"MAT{k}"].plies:
            >>>         ply_data = st.ply_db[f"{ply}"]
        a) Transform stress state to local laminate axis:
                stress_on_axis = ply_data.Ts_p * stress_off_axis
        b) Compute strength ratio
            a = Fxx*sigma_x^2 + Fyy*sigma_y^2
              + Fss*sigma_xy^2 + 2*Fxy*sigma_x*sigma_y
            b = Fx*sigma_x + Fy*sigma_y
            R = -b/(2*a)
              + sqrt( (b/(2*a))**2 + 1/a )
        c) Compute margin of safety
            M.S. = R - 1
    3. Store


@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# ================ Pathing ================


# ================ Module imports ================

# Constants
from cl3o.Constants import N_BOOMS, N_PANELS, T2_TO_T1, BOOM_TO_T4

# Utilities
from cl3o.utils import io_utils as io

# Materials
from cl3o.materials.laminate import PlyData, LaminateData


# ================================================================================
# Data persistence - Tsai-Wu failure results
# ================================================================================

@dataclass
class FailureData:
    '''
    Container for the Tsai-Wu ply-by-ply failure assessment.

    Property            Size                Description                  Units
    ----------------    --------------      -------------------------    -------
    n                   (1,)                Number of nodes              -
    m                   (1,)                Number of elements           -
    nc                  (1,)                Number of load conditions    -
    R_panels            (m, 10, 2, nc)      Min strength ratio (T2)      -
    R_booms             (m, 7,  2, nc)      Min strength ratio (booms)   -
    MS_panels           (m, 10, 2, nc)      Margin of safety (panels)    -
    MS_booms            (m, 7,  2, nc)      Margin of safety (booms)     -
    MS_min_component    (n_comp,)           Flattened component MS       -
    R_min               (1,)                Global min strength ratio    -
    MS_min              (1,)                Global min margin            -
    nv                  (1,)                Plies with MS < 0            -
    '''
    n  : int = 0
    m  : int = 0
    nc : int = 0

    R_panels  : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_PANELS, 2, 0))
    )
    R_booms   : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_BOOMS,  2, 0))
    )
    MS_panels : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_PANELS, 2, 0))
    )
    MS_booms  : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_BOOMS,  2, 0))
    )

    MS_min_component : np.ndarray = field(
        default_factory=lambda: np.zeros((0,))
    )

    R_min  : float = 0.0
    MS_min : float = 0.0
    nv     : int   = 0



# ================================================================================
# PUBLIC API - Tsai-Wu failure evaluator
# ================================================================================

class TsaiWuFailure:
    '''
    Evaluate Tsai-Wu failure ply-by-ply for every laminate panel (T2)
    and flange/boom (T4) in every element, at both ends and for every
    load condition.

    Use:
        tsw = TsaiWuFailure(data=(static_data, runtime_data))
        fd  = tsw.data
    '''

    def __init__(
        self,
        data : tuple,
        enable_logging : bool = True,
    ) -> None:
        '''
        Args:
            data: Tuple (static_data, runtime_data). Required fields:
                static.laminate_db : dict[str, LaminateData]
                static.ply_db      : dict[str, PlyData]
                runtime.sections   : SectionData with sec_data, lam_T1, lam_T4
                runtime.stress     : StressData with sigma and tau tuples
                runtime.mesh.conn  : (m, >=2) element node connectivity
        '''
        self.logger = io.setup_logger(self, enable_logging)

        st, rt = data
        self.lam_db   = st.laminate_db
        self.ply_db   = st.ply_db
        self.lam_T1   = np.asarray(rt.sections.lam_T1, dtype=int)
        self.lam_T4   = np.asarray(rt.sections.lam_T4, dtype=int)
        self.stress   = rt.stress
        self.conn     = np.asarray(rt.mesh.conn, dtype=int)

        self.n, self.m, self.nc = rt.mesh.n, rt.mesh.m, rt.mesh.nc

        # Per-laminate cache of (Ts_p_stack, F_stack) so the inner ply
        # loop runs once per (lam_idx) across the whole evaluation.
        self._lam_cache : dict[int, tuple[np.ndarray, np.ndarray]] = {}

        self._evaluate()

    # ----------------------------------------
    # Private - Laminate retrieval and caching
    # ----------------------------------------

    def _get_laminate(self, lam_idx: int) -> LaminateData:
        '''Retrieve LaminateData from the static database by integer key.'''
        return self.lam_db[f"MAT{int(lam_idx)}"]

    def _laminate_arrays(
        self,
        lam_idx : int,
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Return cached (Ts_p_stack, F_stack) for the load-carrying plies
        of one laminate.

        Ts_p_stack has shape (N, 3, 3) and F_stack has shape (N, 6),
        with rows [Fxx, Fyy, Fss, Fxy, Fx, Fy]. Core plies are skipped.
        Both arrays are empty when the laminate has no load-carrying ply.
        '''
        key = int(lam_idx)
        cached = self._lam_cache.get(key)
        if cached is not None:
            return cached

        lam = self._get_laminate(key)
        Ts_p_list : list[np.ndarray] = []
        F_list    : list[list[float]] = []
        for ply_name in lam.plies:
            ply : PlyData = self.ply_db[ply_name]
            if ply.core or ply.Ts_p is None:
                continue
            Ts_p_list.append(np.asarray(ply.Ts_p, dtype=float))
            F_list.append([
                float(ply.Fxx), float(ply.Fyy), float(ply.Fss),
                float(ply.Fxy), float(ply.Fx),  float(ply.Fy),
            ])

        if Ts_p_list:
            arrays = (np.stack(Ts_p_list, axis=0), np.array(F_list, dtype=float))
        else:
            arrays = (np.zeros((0, 3, 3)), np.zeros((0, 6)))

        self._lam_cache[key] = arrays
        return arrays

    # ----------------------------------------
    # Private - Tsai-Wu math
    # ----------------------------------------

    def _min_strength_ratio(
        self,
        lam_idx : int,
        s_off   : np.ndarray,
    ) -> tuple[float, int]:
        '''
        Minimum Tsai-Wu strength ratio across every load-carrying ply
        of one laminate, evaluated in a single vectorised pass.

        Solves a R^2 + b R - 1 = 0 for the positive root R, with

            a = Fxx s1^2 + Fyy s2^2 + Fss s6^2 + 2 Fxy s1 s2
            b = Fx s1 + Fy s2

        Plies whose quadratic coefficient `a` is non-positive (degenerate
        stress state) get R = +inf.

        Returns:
            Tuple (R_min, n_below) - the minimum R across plies and the
            count of plies with R < 1 in this laminate.
        '''
        Ts_stack, F_stack = self._laminate_arrays(lam_idx)
        if Ts_stack.size == 0:
            return float('inf'), 0

        s_on = Ts_stack @ s_off                                # (N, 3)
        s1, s2, s6 = s_on[:, 0], s_on[:, 1], s_on[:, 2]
        Fxx, Fyy, Fss, Fxy, Fx, Fy = (F_stack[:, k] for k in range(6))

        a = Fxx * s1 * s1 + Fyy * s2 * s2 + Fss * s6 * s6 + 2.0 * Fxy * s1 * s2
        b = Fx * s1 + Fy * s2

        safe    = a > 0.0
        a_safe  = np.where(safe, a, 1.0)
        half_ba = b / (2.0 * a_safe)
        R_safe  = -half_ba + np.sqrt(half_ba * half_ba + 1.0 / a_safe)
        R       = np.where(safe, R_safe, np.inf)

        return float(np.min(R)), int(np.sum(R < 1.0))

    # ----------------------------------------
    # Private - Evaluation pipeline
    # ----------------------------------------

    def _evaluate(self) -> None:
        '''
        Full Tsai-Wu evaluation pipeline.

        Steps
        -----
        1. Extract stress arrays from rt.stress (sigma, tau).
        2. For every (element, end, condition):
            a) Panels: build s_off = [0, 0, tau] and evaluate the
               laminate referenced by lam_T1[sta_idx, T1_seg_idx].
            b) Booms : build s_off = [sigma, 0, 0] and evaluate the
               laminate referenced by lam_T4[sta_idx, flange_idx].
               Booms without a flange (B2, B4, B6) are skipped.
        3. Pack per-component R, MS and global aggregates.
        '''
        sigma_A, sigma_B = self.stress.sigma   # (m, 7,  nc) each
        tau_A,   tau_B   = self.stress.tau     # (m, 10, nc) each

        m  = self.m
        nc = self.nc

        sigma_ends = (sigma_A, sigma_B)
        tau_ends   = (tau_A,   tau_B)
        sta_ends   = (self.conn[:, 0], self.conn[:, 1])

        R_panels  = np.full((m, N_PANELS, 2, nc), np.inf, dtype=float)
        R_booms   = np.full((m, N_BOOMS,  2, nc), np.inf, dtype=float)
        MS_panels = np.full((m, N_PANELS, 2, nc), np.inf, dtype=float)
        MS_booms  = np.full((m, N_BOOMS,  2, nc), np.inf, dtype=float)

        nv = 0

        for e in range(2):
            sigma_e = sigma_ends[e]
            tau_e   = tau_ends[e]
            sta_e   = sta_ends[e]

            for i in range(m):
                sta_idx = int(sta_e[i])
                lamT1_sta = self.lam_T1[sta_idx]
                lamT4_sta = self.lam_T4[sta_idx]

                for j in range(nc):

                    # -------- Panels (T2) --------
                    for p in range(N_PANELS):
                        seg_idx = int(T2_TO_T1[p])
                        s_off = np.array(
                            [0.0, 0.0, float(tau_e[i, p, j])],
                            dtype=float,
                        )
                        R, n_below = self._min_strength_ratio(
                            lam_idx = int(lamT1_sta[seg_idx]),
                            s_off   = s_off,
                        )
                        R_panels[i, p, e, j]  = R
                        MS_panels[i, p, e, j] = R - 1.0
                        nv         += n_below

                    # -------- Booms (T4 flanges) --------
                    for b in range(N_BOOMS):
                        f_idx = int(BOOM_TO_T4[b])
                        if f_idx < 0:
                            continue
                        s_off = np.array(
                            [float(sigma_e[i, b, j]), 0.0, 0.0],
                            dtype=float,
                        )
                        R, n_below = self._min_strength_ratio(
                            lam_idx = int(lamT4_sta[f_idx]),
                            s_off   = s_off,
                        )
                        R_booms[i, b, e, j]  = R
                        MS_booms[i, b, e, j] = R - 1.0
                        nv        += n_below

        MS_min_component = np.concatenate([
            MS_panels.ravel(),
            MS_booms[:, [0, 2, 4, 6], :, :].ravel(),
        ])

        R_min  = float(np.nanmin([np.nanmin(R_panels), np.nanmin(R_booms)]))
        MS_min = R_min - 1.0

        self.logger.info("Tsai-Wu evaluation complete.")
        self.logger.debug(f"Failure summary: \n"
            f"| R_min  : {R_min:.4f}\n"
            f"| MS_min : {MS_min:.4f}\n"
            f"| nv     : {nv}"
        )

        self.data = FailureData(
            n  = self.n,
            m  = m,
            nc = nc,
            R_panels  = R_panels,
            R_booms   = R_booms,
            MS_panels = MS_panels,
            MS_booms  = MS_booms,
            MS_min_component = MS_min_component.astype(float),
            R_min  = float(R_min),
            MS_min = float(MS_min),
            nv = int(nv),
        )



# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
