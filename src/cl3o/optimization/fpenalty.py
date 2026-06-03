'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Penalty Function Module.

Maps a FailureData (Tsai-Wu ply violations) and a DisplacementData
(displacement margin violations) into the scalar penalty term used by
the DE objective.

Implements a logistic (sigmoid) penalty curve:

    P(X) = [ g(k(v - v0)) - g(-k*v0) ] / [ 1 - g(-k*v0) ] * L

where
    v        = max(0, -MS_worst * nv_total)
    g(z)     = 1 / (1 + exp(-z))   (sigmoid)
    MS_worst = min over all components of (MS_tsw, MS_disp)

The shape of P(v) is governed by three real positive constants:

    L       Maximum penalty value
    psi_1   Fraction P(v_1)/L at v_1 = 0.05
    psi_2   Fraction P(v_2)/L at v_2 = 0.20

with k and v0 derived from (psi_1, psi_2) by solving the linear system:

    k  = [logit(psi_2) - logit(psi_1)] / (v_2 - v_1)
    v0 = [v_1*logit(psi_2) - v_2*logit(psi_1)]
         / [logit(psi_2) - logit(psi_1)]

Feasible designs (nv_total == 0) return P = 0. A hard cap PENALTY_CAP
prevents numeric overflow for pathological candidates.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ================ Pathing ================


# ================ Module imports ================

# Constants
from cl3o.Constants import (
    TOL,
    PENALTY_VARS,
)

# Utilities
from cl3o.utils import io_utils as io

# FEA
from cl3o.fea.post.tsw_failure     import FailureData
from cl3o.fea.post.displacement_ms import DisplacementData

# ================ Global variables ================
PENALTY_OVERFLOW_CAP = PENALTY_VARS["overflow"]


# ================================================================================
# Data persistence - Penalty breakdown
# ================================================================================

@dataclass
class PenaltyData:
    '''
    Scalar penalty term and its driving quantities.

    Property        Size        Description                             Units
    ------------    --------    ----------------------------------    --------
    Pcap            (1,)        Max penalty (penalty CAP)             -
    k               (1,)        Logistic slope                        -
    v0              (1,)        Inflection violation                  -
    n_tsw           (1,)        Ply-level Tsai-Wu violations          -
    n_disp          (1,)        Displacement margin violations        -
    nv_total        (1,)        Total number of violations            -
    MS_tsw          (1,)        Global min Tsai-Wu ply MS             -
    MS_disp         (1,)        Global min displacement MS            -
    MS_worst        (1,)        Global min MS                         -
    v_value         (1,)        Scalar value of violation function    -
    total           (1,)        Scalar value of P(X)                  -
    is_feasible     -           True when n_violations == 0           bool
    '''
    Pcap        : float = 0.0
    k           : float = 0.0
    v0          : float = 0.0
    n_tsw       : int   = 0
    n_disp      : int   = 0
    nv_total    : int   = 0
    MS_tsw      : float = 0.0
    MS_disp     : float = 0.0
    MS_worst    : float = 0.0
    v_value     : float = 0.0
    total       : float = 0.0
    is_feasible : bool  = False


# ================================================================================
# PUBLIC API - Penalty evaluator
# ================================================================================

class Penalty:
    '''
    Evaluate the logistic DE penalty P(X) from a FailureData and a
    DisplacementData container.

    Use:
        p = Penalty(data=(failure_data, disp_data)).data
    '''

    def __init__(
        self,
        data : tuple[FailureData, DisplacementData],
        Pcap : float = PENALTY_VARS["Pcap"],
        v1 : Optional[float] = PENALTY_VARS["v1"],
        v2 : Optional[float] = PENALTY_VARS["v2"],
        nv_test : Optional[float] = PENALTY_VARS["nv_test"],
        psi1 : Optional[float] = PENALTY_VARS["psi1"],
        psi2 : Optional[float] = PENALTY_VARS["psi2"],
        k : Optional[float] = PENALTY_VARS["k"],
        v0 : Optional[float] = PENALTY_VARS["v0"],
        enable_logging : bool  = True,
        verbose        : bool  = False,
    ) -> None:
        '''
        Args:
            data: Tuple (failure_data, disp_data) carrying the Tsai-Wu
                and displacement margin containers.
            Pcap          : Max penalty (upper asymptote of P).
            psi1          : P(v1)/L fraction at v1.
            psi2          : P(v2)/L fraction at v2.
            enable_logging: Toggle logger.
            verbose       : When True, log at DEBUG level.
        '''
        self.logger = io.setup_logger(self, enable_logging, verbose)

        failure_data, disp_data = data

        # -------- 1. Derive logistic shape constants --------
        if k is None and v0 is None:
            k, v0 = self._derive_k_v0(psi1, psi2, v1, v2, nv_test)

        # -------- 2. Extract margins and violation counts --------
        n_tsw    = int(failure_data.nv)
        n_disp   = int(disp_data.nv)
        nv_total = n_tsw + n_disp

        MS_tsw   = self._min_tsw_margin(failure_data)
        MS_disp  = float(disp_data.MS_min)
        MS_worst = min(MS_tsw, MS_disp)
        v        = max(0.0, -MS_worst * nv_total)

        # -------- 3. Evaluate penalty --------
        total    = self._penalty_value(v, Pcap, k, v0, nv_total)
        feasible = nv_total == 0

        self.logger.debug(
            f"Penalty evaluated.\n"
            f"| Pcap         : {float(Pcap)}\n"
            f"| k            : {k:.4f}\n"
            f"| v0           : {v0:.4f}\n"
            f"| n_tsw        : {n_tsw}\n"
            f"| n_disp       : {n_disp}\n"
            f"| nv_total     : {nv_total}\n"
            f"| MS_tsw       : {MS_tsw:.4f}\n"
            f"| MS_disp      : {MS_disp:.4f}\n"
            f"| MS_worst     : {MS_worst:.4f}\n"
            f"| v_value      : {v:.4f}\n"
            f"| total        : {total:.4e}\n"
            f"| is_feasible  : {feasible}"
        )

        # -------- 4. Pack results --------
        self.data = PenaltyData(
            Pcap        = float(Pcap),
            k           = float(k),
            v0          = float(v0),
            n_tsw       = n_tsw,
            n_disp      = n_disp,
            nv_total    = nv_total,
            MS_tsw      = float(MS_tsw),
            MS_disp     = float(MS_disp),
            MS_worst    = float(MS_worst),
            v_value     = float(v),
            total       = float(total),
            is_feasible = bool(feasible),
        )

    # --------------------------------------------------------
    # Private methods - Penalty math
    # --------------------------------------------------------

    def _sigmoid(self, z: float) -> float:
        '''Standard logistic function g(z) = 1 / (1 + exp(-z)).'''
        # Split on sign to avoid overflow in exp() for large |z|.
        if z >= 0.0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def _logit(self, p: float) -> float:
        '''Inverse sigmoid g^(-1)(p) = ln(p / (1 - p)).'''
        if not (0.0 < p < 1.0):
            raise ValueError(
                f"[CL3O] logit(p) requires 0 < p < 1, got p = {p}.\n"
                f"Check psi_1, psi_2 ranges ([0.05, 0.95])."
            )
        return math.log(p / (1.0 - p))

    def _derive_k_v0(
        self,
        psi1 : float,
        psi2 : float,
        v1   : float,
        v2   : float,
        nv_test : int,
    ) -> tuple[float, float]:
        '''
        Solve the 2x2 linear system:

            g(k*(v1 - v0)) = psi1       k*(v1 - v0) = logit(psi1)
            g(k*(v2 - v0)) = psi2       k*(v2 - v0) = logit(psi2)

        Returns (k, v0).
        '''
        v1 = v1 * nv_test
        v2 = v2 * nv_test

        l1 = self._logit(psi1)
        l2 = self._logit(psi2)
        denom = l2 - l1
        if abs(denom) < TOL:
            raise ValueError(
                f"[CL3O] psi_1 and psi_2 produce a degenerate logit "
                f"difference.\n| psi_1 : {psi1}\n| psi_2 : {psi2}\n"
                f"Pick psi_2 strictly greater than psi_1."
            )
        k  = denom / (v2 - v1)
        v0 = (v1 * l2 - v2 * l1) / denom
        return k, v0

    def _penalty_value(
        self,
        v   : float,
        Pcap: float,
        k   : float,
        v0  : float,
        n   : int,
    ) -> float:
        '''
        Logistic penalty:

            P(v) = [g(k(v - v0)) - g(-k*v0)] / [1 - g(-k*v0)] * L

        Returns 0.0 when n <= 0 (feasible design, no violations).
        Clipped to PENALTY_CAP to prevent overflow.
        '''
        if n <= 0:
            return 0.0
        try:
            g_shift = self._sigmoid(-k * v0)
            g_v     = self._sigmoid(k * (v - v0))
            denom   = 1.0 - g_shift
            if abs(denom) < 1.0e-15:
                return float(Pcap)
            value = (g_v - g_shift) / denom * float(Pcap)
        except OverflowError:
            return PENALTY_OVERFLOW_CAP
        return float(min(max(value, 0.0), PENALTY_OVERFLOW_CAP))

    def _min_tsw_margin(self, failure_data: FailureData) -> float:
        '''
        Global minimum Tsai-Wu ply margin across all elements, stations,
        and components. Returns PENALTY_CAP when no elements are present
        (all plies nominally safe).
        '''
        ms = failure_data.MS_min_component
        if ms.size == 0:
            return float(PENALTY_OVERFLOW_CAP)
        val = float(np.nanmin(ms))
        return val if np.isfinite(val) else float(PENALTY_OVERFLOW_CAP)


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
