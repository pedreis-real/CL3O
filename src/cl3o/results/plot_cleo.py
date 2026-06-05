'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
DE Convergence Plot Module.

Visualizes the outcome of a Differential Evolution run: best-so-far
fitness, population mean + standard deviation envelope, and a per-
generation trajectory of each design variable (normalized to its
bounds).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ================ Pathing ================


# ================ Module imports ================
from cl3o.utils import io_utils as io
from cl3o.optimization.de_opt import OptData, HistoryData

# ================ Global variables ================
_COLOR_BEST = "#3350b7"
_COLOR_MEAN = "#2da84a"
_COLOR_BAND = "#c9d7f0"
_COLOR_FEAS = "#d63b3b"


# ================================================================================
# Internal Helpers
# ================================================================================

class PlotCL3OHelper:
    def __init__(self):
        pass

    @staticmethod
    def running_best(best_f: np.ndarray) -> np.ndarray:
        '''Return np.minimum.accumulate, coercing to float array first.'''
        arr = np.asarray(best_f, dtype=float).ravel()
        if arr.size == 0:
            return arr
        return np.minimum.accumulate(arr)

    @staticmethod
    def normalized_trajectories(
        best_X    : np.ndarray,
        bounds_lo : np.ndarray,
        bounds_hi : np.ndarray,
    ) -> np.ndarray:
        '''
        Return best_X in [0, 1] per design variable, using the DE bounds.

        Args:
            best_X   : (n_gen+1, n_dim) best-design trajectory.
            bounds_lo: (n_dim,) lower bounds.
            bounds_hi: (n_dim,) upper bounds.

        Returns:
            (n_gen+1, n_dim) normalized trajectory.
        '''
        X  = np.asarray(best_X,    dtype=float)
        lo = np.asarray(bounds_lo, dtype=float)
        hi = np.asarray(bounds_hi, dtype=float)
        span = np.where(hi - lo > 0.0, hi - lo, 1.0)
        return (X - lo[None, :]) / span[None, :]


# ================================================================================
# PUBLIC API - Fitness convergence
# ================================================================================

def plot_convergence(
    history        : HistoryData,
    title          : str | None        = None,
    save_path      : str | Path | None = None,
    show           : bool                 = True,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    Best-so-far and per-generation fitness plot with a +/-1 std band.

    Args:
        history : HistoryData from RunOpt.
        title   : Optional figure title.
        save_path: Optional PNG/PDF output path.
        show    : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_convergence", enable_logging)

    best_f = np.asarray(history.best_f, dtype=float).ravel()
    mean_f = np.asarray(history.mean_f, dtype=float).ravel()
    std_f  = np.asarray(history.std_f,  dtype=float).ravel()

    if best_f.size == 0:
        raise ValueError(
            "[CL3O] plot_convergence: empty history (best_f has size 0)."
        )

    gens = np.arange(best_f.size)
    run_best = PlotCL3OHelper.running_best(best_f)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.fill_between(
        gens,
        mean_f - std_f,
        mean_f + std_f,
        color = _COLOR_BAND,
        alpha = 0.6,
        label = "mean +/- std",
    )
    ax.plot(
        gens, mean_f,
        color     = _COLOR_MEAN,
        linewidth = 1.2,
        label     = "population mean",
    )
    ax.plot(
        gens, best_f,
        color     = _COLOR_BEST,
        linewidth = 1.4,
        label     = "per-gen best",
    )
    ax.plot(
        gens, run_best,
        color     = _COLOR_BEST,
        linewidth = 2.0,
        linestyle = "--",
        label     = "best so far",
    )

    feasible_f = float(history.feasible_f)
    if np.isfinite(feasible_f):
        ax.axhline(
            feasible_f,
            color     = _COLOR_FEAS,
            linewidth = 1.0,
            linestyle = ":",
            label     = f"best feasible = {feasible_f:.4g}",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid(which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(
            f"DE convergence  |  n_gen={history.ng}"
            f"  |  n_dim={history.D}"
        )

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"Convergence figure saved to {target}")

    if show:
        plt.show()

    return fig


# ================================================================================
# PUBLIC API - Normalized design-variable trajectories
# ================================================================================

def plot_design_trajectories(
    history        : HistoryData,
    opt_data       : OptData,
    title          : str | None        = None,
    save_path      : str | Path | None = None,
    show           : bool                 = True,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    Plot each design variable of best_X normalized to its DE bounds.

    Args:
        history  : HistoryData from RunOpt.
        opt_data : OptData from SetupOpt (used for bounds).
        title    : Optional figure title.
        save_path: Optional PNG/PDF output path.
        show     : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_design_trajectories", enable_logging)

    best_X = np.asarray(history.best_X, dtype=float)
    if best_X.ndim != 2 or best_X.size == 0:
        raise ValueError(
            "[CL3O] plot_design_trajectories: best_X must be (n_gen+1, n_dim)."
        )

    X_norm = PlotCL3OHelper.normalized_trajectories(
        best_X    = best_X,
        bounds_lo = opt_data.lo,
        bounds_hi = opt_data.hi,
    )

    gens  = np.arange(best_X.shape[0])
    n_dim = best_X.shape[1]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = plt.get_cmap("viridis")

    for j in range(n_dim):
        ax.plot(
            gens, X_norm[:, j],
            color     = cmap(j / max(n_dim - 1, 1)),
            linewidth = 1.0,
            alpha     = 0.85,
            label     = f"x[{j}]",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized design value  (0 = lo, 1 = hi)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(which="both", linestyle=":", alpha=0.6)
    if n_dim <= 20:
        ax.legend(
            loc            = "center left",
            bbox_to_anchor = (1.0, 0.5),
            fontsize       = 8,
            ncol           = 1,
        )

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Best-design trajectories  |  n_dim={n_dim}"
            f"  |  n_gen={history.ng}"
        )

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"Design-trajectory figure saved to {target}")

    if show:
        plt.show()

    return fig


# ================================================================================
# PUBLIC API - Live DE Monitor (on_generation callback)
# ================================================================================

class LiveDEPlotter:
    '''
    Live DE monitor that refreshes two matplotlib windows in place on
    every generation: a convergence panel and a design-trajectory panel.

    Mirrors plot_convergence / plot_design_trajectories visuals but runs
    in real time as an on_generation callback.

    Use as the on_generation callback of RunCLEO.run_optimization:
        plotter = LiveDEPlotter(opt_data=setup.data)
        runner.run_optimization(..., on_generation=plotter)
        plotter.close()
    '''

    def __init__(
        self,
        opt_data      : OptData,
        refresh_every : int = 1,
    ) -> None:
        self.opt_data      = opt_data
        self.refresh_every = max(int(refresh_every), 1)

        plt.ion()

        # -------- Convergence figure --------
        self.fig_c, self.ax_c = plt.subplots(figsize=(9, 4.5))
        self.ax_c.set_xlabel("Generation")
        self.ax_c.set_ylabel("Fitness")
        self.ax_c.grid(which="both", linestyle=":", alpha=0.6)
        self.fig_c.canvas.manager.set_window_title("CL3O - DE convergence")

        (self._ln_mean,) = self.ax_c.plot(
            [], [], color=_COLOR_MEAN, linewidth=1.2, label="population mean",
        )
        (self._ln_best,) = self.ax_c.plot(
            [], [], color=_COLOR_BEST, linewidth=1.4, label="per-gen best",
        )
        (self._ln_rbest,) = self.ax_c.plot(
            [], [], color=_COLOR_BEST, linewidth=2.0, linestyle="--",
            label="best so far",
        )
        self._band_c = None
        self.ax_c.legend(loc="best", fontsize=9)

        # -------- Design-trajectory figure --------
        self.fig_t, self.ax_t = plt.subplots(figsize=(9, 4.5))
        self.ax_t.set_xlabel("Generation")
        self.ax_t.set_ylabel("Normalized design value  (0 = lo, 1 = hi)")
        self.ax_t.set_ylim(-0.05, 1.05)
        self.ax_t.grid(which="both", linestyle=":", alpha=0.6)
        self.fig_t.canvas.manager.set_window_title(
            "CL3O - Design trajectories"
        )

        n_dim = int(opt_data.D)
        cmap  = plt.get_cmap("viridis")
        self._ln_traj = []
        for j in range(n_dim):
            (ln,) = self.ax_t.plot(
                [], [],
                color     = cmap(j / max(n_dim - 1, 1)),
                linewidth = 1.0,
                alpha     = 0.85,
                label     = f"x[{j}]",
            )
            self._ln_traj.append(ln)
        if n_dim <= 20:
            self.ax_t.legend(
                loc            = "center left",
                bbox_to_anchor = (1.0, 0.5),
                fontsize       = 8,
                ncol           = 1,
            )

        plt.show(block=False)

    def __call__(self, k: int, history: HistoryData) -> None:
        if (k % self.refresh_every) != 0:
            return

        # -------- Convergence update --------
        gens   = np.arange(history.best_f.size)
        best_f = np.asarray(history.best_f, dtype=float).ravel()
        mean_f = np.asarray(history.mean_f, dtype=float).ravel()
        std_f  = np.asarray(history.std_f,  dtype=float).ravel()
        rbest  = PlotCL3OHelper.running_best(best_f)

        self._ln_mean .set_data(gens, mean_f)
        self._ln_best .set_data(gens, best_f)
        self._ln_rbest.set_data(gens, rbest)

        if self._band_c is not None:
            self._band_c.remove()
        self._band_c = self.ax_c.fill_between(
            gens, mean_f - std_f, mean_f + std_f,
            color=_COLOR_BAND, alpha=0.6,
        )

        self.ax_c.set_title(
            f"DE convergence  |  gen {k}/{self.opt_data.k_max}  "
            f"|  best={best_f[-1]:.4g}"
        )
        self.ax_c.relim()
        self.ax_c.autoscale_view()

        # -------- Trajectory update --------
        X_norm = PlotCL3OHelper.normalized_trajectories(
            best_X    = history.best_X,
            bounds_lo = self.opt_data.lo,
            bounds_hi = self.opt_data.hi,
        )
        for j, ln in enumerate(self._ln_traj):
            ln.set_data(gens, X_norm[:, j])
        self.ax_t.set_title(
            f"Best-design trajectories  |  gen {k}/{self.opt_data.k_max}"
        )
        self.ax_t.set_xlim(0, max(gens[-1], 1))

        for fig in (self.fig_c, self.fig_t):
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        plt.ioff()


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
