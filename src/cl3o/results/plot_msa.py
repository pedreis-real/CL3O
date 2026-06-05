'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
MSA Diagnostics Plot Module.

Sanity-check plots for the linear static FEM solve: sparsity pattern of
the global stiffness matrix [K], eigenvalue spectrum of the free-DOF
partition, and a deformed-vs-undeformed overlay of the nodal mesh.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                         # noqa: F401

# ================ Pathing ================


# ================ Module imports ================
from cl3o.utils import io_utils as io
from cl3o.fea.solver.mesh_builder import MeshData
from cl3o.fea.solver.static_analysis   import FeaResults

# ================ Global variables ================
_COLOR_UNDEF = "#555555"
_COLOR_DEF   = "#3350b7"
_TOL         = 1e-12


# ================================================================================
# Internal Helpers
# ================================================================================

class PlotMSAHelper:
    def __init__(self):
        pass

    @staticmethod
    def free_partition(
        K           : np.ndarray,
        restraints  : np.ndarray,
        settlements : np.ndarray,
    ) -> np.ndarray:
        '''Extract [K_ff] using the same convention as MSAHelper.'''
        r = np.asarray(restraints, dtype=int).ravel().astype(bool)
        s = np.asarray(settlements, dtype=float).ravel()
        is_constrained = r | (s != 0.0)
        free_idx = np.where(~is_constrained)[0]
        return K[np.ix_(free_idx, free_idx)]


# ================================================================================
# PUBLIC API - [K] sparsity pattern
# ================================================================================

def plot_K_sparsity(
    K              : np.ndarray,
    title          : str | None        = None,
    save_path      : str | Path | None = None,
    show           : bool                 = True,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    Spy plot of the non-zero structure of the global stiffness matrix.

    Args:
        K        : Square stiffness matrix (6n, 6n).
        title    : Optional figure title.
        save_path: Optional PNG/PDF output path.
        show     : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_K_sparsity", enable_logging)

    M = np.asarray(K, dtype=float)
    nnz     = int(np.count_nonzero(np.abs(M) > _TOL))
    density = float(nnz) / float(M.size) if M.size > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.spy(M, markersize=2.0, color=_COLOR_DEF)

    ax.set_xlabel("Column DOF")
    ax.set_ylabel("Row DOF")
    ax.set_title(
        title
        if title is not None
        else (
            f"K sparsity  |  shape={M.shape}  "
            f"|  nnz={nnz}  |  density={density:.3%}"
        )
    )

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"K sparsity figure saved to {target}")

    if show:
        plt.show()

    return fig


# ================================================================================
# PUBLIC API - Eigenvalue spectrum of [K_ff]
# ================================================================================

def plot_K_spectrum(
    fem_arrays     : MeshData,
    title          : str | None        = None,
    save_path      : str | Path | None = None,
    show           : bool                 = True,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    Sorted magnitude spectrum of the free-DOF stiffness partition [K_ff].

    The plot uses a symlog scale because physically [K_ff] is positive
    definite but round-off can leak near-zero modes.

    Args:
        fem_arrays: Global FEM arrays container.
        title     : Optional figure title.
        save_path : Optional PNG/PDF output path.
        show      : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_K_spectrum", enable_logging)

    st_flat = getattr(
        fem_arrays, "st_flat",
        np.zeros_like(np.asarray(fem_arrays.re_flat, dtype=float)),
    )
    K_ff = PlotMSAHelper.free_partition(
        K           = fem_arrays.K,
        restraints  = fem_arrays.re_flat,
        settlements = st_flat,
    )
    if K_ff.size == 0:
        raise ValueError(
            "[CL3O] plot_K_spectrum: free partition K_ff is empty."
        )

    # Symmetrize to tame numerical asymmetry, then take real eigenvalues
    K_sym = 0.5 * (K_ff + K_ff.T)
    eig   = np.linalg.eigvalsh(K_sym)
    mag   = np.sort(np.abs(eig))[::-1]

    idx      = np.arange(1, mag.size + 1)
    cond_num = (
        float(mag[0] / mag[-1]) if mag[-1] > _TOL else float("inf")
    )
    n_neg    = int(np.sum(eig < -_TOL))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(idx, mag, marker=".", color=_COLOR_DEF, linewidth=1.0)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("|lambda(K_ff)|")
    ax.grid(which="both", linestyle=":", alpha=0.6)

    ax.set_title(
        title
        if title is not None
        else (
            f"K_ff spectrum  |  n={K_ff.shape[0]}  "
            f"|  cond={cond_num:.2e}  |  n_neg={n_neg}"
        )
    )

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"K spectrum figure saved to {target}")

    if show:
        plt.show()

    return fig


# ================================================================================
# PUBLIC API - Deformed mesh overlay
# ================================================================================

def plot_deformed_mesh(
    coord          : np.ndarray,
    conn           : np.ndarray,
    results        : FeaResults,
    scale          : float                = 1.0,
    title          : str | None        = None,
    save_path      : str | Path | None = None,
    show           : bool                 = True,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    Overlay undeformed and (scaled) deformed nodal polylines in 3-D.

    Args:
        coord   : Nodal coordinates (n, 3) [mm].
        conn    : Connectivity (m, 2) of beginning/end node indices.
        results : MSAResults from the linear static solve.
        scale   : Multiplier applied to nodal translations for the
            overlay. Use > 1 to exaggerate small displacements.
        title    : Optional figure title.
        save_path: Optional PNG/PDF output path.
        show     : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_deformed_mesh", enable_logging)

    coord = np.asarray(coord, dtype=float).reshape(-1, 3)
    conn  = np.asarray(conn,  dtype=int)
    if conn.ndim == 2 and conn.shape[1] > 2:
        conn = conn[:, :2]
    conn  = conn.reshape(-1, 2)

    d_mat = np.asarray(results.dmatrix, dtype=float)
    if d_mat.ndim == 3:
        d_mat = d_mat[:, :, 0]
    if d_mat.shape[0] != 6 or d_mat.shape[1] != coord.shape[0]:
        raise ValueError(
            f"[CL3O] plot_deformed_mesh: d_mat shape mismatch.\n"
            f"| d_mat.shape : {d_mat.shape}\n"
            f"| n_nodes     : {coord.shape[0]}"
        )
    disp = d_mat[:3, :].T                    # (n, 3) translations
    coord_def = coord + float(scale) * disp

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")

    for (nb, ne) in conn:
        ax.plot(
            [coord[nb, 0], coord[ne, 0]],
            [coord[nb, 1], coord[ne, 1]],
            [coord[nb, 2], coord[ne, 2]],
            color=_COLOR_UNDEF, linewidth=1.0, alpha=0.6,
        )
        ax.plot(
            [coord_def[nb, 0], coord_def[ne, 0]],
            [coord_def[nb, 1], coord_def[ne, 1]],
            [coord_def[nb, 2], coord_def[ne, 2]],
            color=_COLOR_DEF, linewidth=1.4,
        )

    ax.scatter(
        coord[:, 0], coord[:, 1], coord[:, 2],
        color=_COLOR_UNDEF, s=12, label="Undeformed",
    )
    ax.scatter(
        coord_def[:, 0], coord_def[:, 1], coord_def[:, 2],
        color=_COLOR_DEF, s=14, label=f"Deformed (x{scale:g})",
    )

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.legend(loc="best")
    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"Deformed-mesh figure saved to {target}")

    if show:
        plt.show()

    return fig


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
