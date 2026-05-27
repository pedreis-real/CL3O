'''
================================================================================
CWSS - Composite Wing Structural Sizing.
MSA Diagnostics Plot Tests Module.

Phase I regression tests for results.plot_msa:

  - plot_K_sparsity renders a spy figure and writes a PNG.
  - plot_K_spectrum extracts [K_ff] from FemArraysData.restraints, keeps
    only the free-free partition, and produces finite eigenvalues for a
    positive-definite stiffness block.
  - plot_deformed_mesh catches an MSAResults.d_mat with a wrong shape.
  - PlotMSAHelper.free_partition matches the MSAHelper convention:
    DOFs marked restrained OR with non-zero settlement are excluded.

@ CWSS Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path

import matplotlib
# matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ================ Module imports ================
from fem.solver.mesh_builder import FemArraysData
from fem.solver.msa_solver   import MSAResults
from results.plot_msa        import (
    plot_K_sparsity, plot_K_spectrum, plot_deformed_mesh, PlotMSAHelper,
)


# ================================================================================
# PRIVATE API - Synthetic mini-mesh
# ================================================================================

def _make_fem_arrays(n: int = 4) -> tuple[FemArraysData, np.ndarray, np.ndarray]:
    '''
    Build a trivial SPD global stiffness for an n-node chain along +Y.

    Node 0 fixed; all other DOFs free. Returns (fem, coord, conn).
    '''
    dof = 6 * n
    K   = np.eye(dof) * 1000.0
    for i in range(n - 1):
        a = 6 * i
        K[a     : a + 6,  a + 6: a + 12] = -300.0
        K[a + 6 : a + 12, a     : a + 6] = -300.0
        K[a + 6 : a + 12, a + 6 : a + 12] += 600.0
    K = 0.5 * (K + K.T)

    restraints = np.zeros(dof, dtype=int)
    restraints[:6] = 1
    settlements = np.zeros(dof)

    fem = FemArraysData(
        n_nodes     = n,
        n_elements  = n - 1,
        K           = K,
        Pf          = np.zeros(dof),
        F_nodal     = np.zeros(dof),
        Ni          = np.zeros((12, 12, n - 1)),
        R_be        = np.zeros((3,  3,  n - 1)),
        Ei          = np.zeros((12, n - 1), dtype=int),
        Qfi         = np.zeros((12, n - 1)),
        restraints  = restraints,
        settlements = settlements,
    )

    coord = np.stack([
        np.zeros(n),
        np.linspace(0.0, 1000.0, n),
        np.zeros(n),
    ], axis=1)
    conn = np.column_stack([np.arange(n - 1), np.arange(1, n)])

    return fem, coord, conn


# ================================================================================
# PUBLIC API - Helper unit tests
# ================================================================================

def test_free_partition_excludes_restraints() -> None:
    '''Restrained and settled DOFs are dropped from the free partition.'''
    K = np.arange(36, dtype=float).reshape(6, 6)
    restraints  = np.array([1, 0, 0, 0, 0, 0])
    settlements = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    K_ff = PlotMSAHelper.free_partition(K, restraints, settlements)

    # Expected kept indices: [1, 3, 4, 5] (4 free)
    assert K_ff.shape == (4, 4)
    assert np.allclose(K_ff, K[np.ix_([1, 3, 4, 5], [1, 3, 4, 5])])


# ================================================================================
# PUBLIC API - Rendering tests
# ================================================================================

def test_plot_K_sparsity_renders(tmp_path: Path) -> None:
    '''Spy plot writes a PNG larger than 1 kB.'''
    fem, _, _ = _make_fem_arrays(n=5)
    out = tmp_path / "K_spy.png"

    fig = plot_K_sparsity(
        K              = fem.K,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    plt.close(fig)


def test_plot_K_spectrum_positive_definite(tmp_path: Path) -> None:
    '''All free-partition eigenvalues are strictly positive for SPD [K_ff].'''
    fem, _, _ = _make_fem_arrays(n=4)

    # Independently recompute eigenvalues to cross-check the plot math
    K_ff = PlotMSAHelper.free_partition(
        fem.K, fem.restraints, fem.settlements,
    )
    eig = np.linalg.eigvalsh(0.5 * (K_ff + K_ff.T))
    assert np.all(eig > 0.0)
    assert np.all(np.isfinite(eig))

    out = tmp_path / "K_spec.png"
    fig = plot_K_spectrum(
        fem_arrays     = fem,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    plt.close(fig)


def test_plot_deformed_mesh_renders(tmp_path: Path) -> None:
    '''plot_deformed_mesh overlays undeformed/deformed polylines.'''
    fem, coord, conn = _make_fem_arrays(n=4)
    d_mat = np.zeros((6, fem.n_nodes))
    d_mat[2, 1:] = [5.0, 15.0, 35.0]        # cantilever-like deflection
    res = MSAResults(
        d         = np.zeros(6 * fem.n_nodes),
        d_mat     = d_mat,
        R         = np.zeros(6 * fem.n_nodes),
        R_mat     = np.zeros((6, fem.n_nodes)),
        Q_cc      = np.zeros((12, fem.n_elements)),
        Q_geom    = np.zeros((12, fem.n_elements)),
        free_dofs = np.arange(6, 6 * fem.n_nodes),
    )

    out = tmp_path / "def.png"
    fig = plot_deformed_mesh(
        coord          = coord,
        conn           = conn,
        results        = res,
        scale          = 10.0,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    plt.close(fig)


def test_plot_deformed_mesh_rejects_bad_shape() -> None:
    '''A d_mat whose shape does not match (6, n_nodes) must raise.'''
    fem, coord, conn = _make_fem_arrays(n=4)
    bad = MSAResults(d_mat=np.zeros((6, 99)))
    with pytest.raises(ValueError):
        plot_deformed_mesh(
            coord          = coord,
            conn           = conn,
            results        = bad,
            show           = False,
            enable_logging = False,
        )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    # pytest.main([__file__, "-q"])
    fem, coord, conn = _make_fem_arrays(n=5)
    d_mat = np.zeros((6, fem.n_nodes))
    d_mat[2, 1:] = np.linspace(0.0, 35.0, fem.n_nodes - 1)
    res = MSAResults(
        d         = np.zeros(6 * fem.n_nodes),
        d_mat     = d_mat,
        R         = np.zeros(6 * fem.n_nodes),
        R_mat     = np.zeros((6, fem.n_nodes)),
        Q_cc      = np.zeros((12, fem.n_elements)),
        Q_geom    = np.zeros((12, fem.n_elements)),
        free_dofs = np.arange(6, 6 * fem.n_nodes),
    )
    plot_K_sparsity(K=fem.K, enable_logging=False)
    plot_K_spectrum(fem_arrays=fem, enable_logging=False)
    plot_deformed_mesh(coord=coord, conn=conn, results=res, scale=10.0, enable_logging=False)
    plt.show()
