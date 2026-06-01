'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Results Statistics & Visualization Module.

Post-hoc, plots-only consumer of CL3O result artifacts. Reads the Latin-
Hypercube DE sweep (results.csv + per-sample rate.csv), the ANOVA sensitivity
tables (anova_results.csv + anova_summary.csv), and the last-generation
RuntimeData pickle of a run, and renders publication-style vector-PDF figures
into tools/output/stats/. Writes nothing back into the producing tools.

Requires the optional 'analysis' extra:  pip install -e ".[analysis]"

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# ================ Module imports ================

# Utilities
from cl3o.paths import ROOT_DIR, OUTPUTS_DIR
from cl3o.utils import io_utils as io


# ================================================================================
# Data container
# ================================================================================

@dataclass
class StatsData:
    '''
    Resolved configuration for a stats visualization session.

    Property        Description
    ------------    --------------------------------------------------------
    aircraft        Aircraft id (lower-cased on init)
    sweep           LHS sweep name (tools/output/<sweep>/)
    run_name        outputs/ run folder for the best-design pkl
    tools_out       tools/output root (for results.csv / anova_*.csv)
    outputs_root    repo outputs/ root (for rate.csv + run pkl)
    out_dir         figure output directory (default tools/output/stats)
    dpi             raster DPI for any embedded bitmap elements
    fmt             figure format, fixed to 'pdf' (vector, for LaTeX)
    lhs_results     derived: tools_out/<sweep>/results.csv
    anova_results   derived: tools_out/sensitivity/anova_results.csv
    anova_summary   derived: tools_out/sensitivity/anova_summary.csv
    rate_pattern    derived: glob for per-sample LHS folders
    '''
    aircraft     : str = "da62"
    sweep        : str = "tune-de-3"
    run_name     : str | None = None
    tools_out    : Path = field(default_factory=lambda: ROOT_DIR / "tools" / "output")
    outputs_root : Path = field(default_factory=lambda: OUTPUTS_DIR)
    out_dir      : Path | None = None
    dpi          : int = 150
    fmt          : str = "pdf"

    lhs_results   : Path = field(init=False)
    anova_results : Path = field(init=False)
    anova_summary : Path = field(init=False)
    rate_pattern  : str  = field(init=False)

    def __post_init__(self) -> None:
        self.aircraft = self.aircraft.lower()
        if self.out_dir is None:
            self.out_dir = self.tools_out / "stats"
        self.lhs_results   = self.tools_out / self.sweep / "results.csv"
        self.anova_results = self.tools_out / "sensitivity" / "anova_results.csv"
        self.anova_summary = self.tools_out / "sensitivity" / "anova_summary.csv"
        self.rate_pattern  = f"{self.aircraft}_{self.sweep}_LHS-*"
        if self.run_name is None:
            self.run_name = f"{self.aircraft}_{self.sweep}_LHS-0"
        self.out_dir.mkdir(parents=True, exist_ok=True)


# Hyper-parameter and outcome columns of the LHS results.csv.
_LHS_PARAMS   = ["NP", "CR", "F", "lambda"]
_LHS_OUTCOMES = ["best_f_final", "feasible_f", "n_gens", "gen_to_half", "elapsed_s"]


# ================================================================================
# Helper - loaders and pure stats
# ================================================================================

class StatsHelper:
    '''Static loaders for CL3O result artifacts and small stats utilities.'''

    # Sub-directory names a run may use for its per-generation pickles.
    _PKL_SUBDIRS = ("generations", "opt_files", "")

    @staticmethod
    def load_lhs_results(path: str | Path) -> pd.DataFrame:
        '''Read the LHS sweep results.csv into a DataFrame.'''
        return pd.read_csv(path)

    @staticmethod
    def load_rate_curves(
        outputs_root : str | Path,
        pattern      : str,
    ) -> dict[int, pd.DataFrame]:
        '''
        Read every per-sample rate.csv under outputs_root matching `pattern`.

        Args:
            outputs_root: Directory holding the per-sample LHS run folders.
            pattern:      Glob for the run folders (e.g. "da62_sw_LHS-*").

        Returns:
            Dict mapping LHS sample index (parsed from the folder suffix) to
            its rate DataFrame, ordered by index. Folders without a rate.csv
            or an unparsable suffix are skipped.
        '''
        root = Path(outputs_root)
        curves: dict[int, pd.DataFrame] = {}
        for d in sorted(root.glob(pattern)):
            csv = d / "rate.csv"
            if not csv.is_file():
                continue
            try:
                idx = int(d.name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            curves[idx] = pd.read_csv(csv)
        return dict(sorted(curves.items()))

    @staticmethod
    def load_anova(
        results_path : str | Path,
        summary_path : str | Path,
    ) -> tuple[pd.DataFrame, dict | None]:
        '''
        Read the per-group ANOVA table and its optional sibling summary.

        Args:
            results_path: anova_results.csv (per-group rows, variant A).
            summary_path: anova_summary.csv (single summary row); optional.

        Returns:
            Tuple (per_group_df, summary_dict_or_None). summary_dict carries
            the float-cast values for the keys present among grand_mean,
            SS_total, df_between, df_within, F_stat, p_value.
        '''
        df = pd.read_csv(results_path)
        summary: dict | None = None
        sp = Path(summary_path)
        if sp.is_file():
            row  = pd.read_csv(sp).iloc[0]
            keys = ("grand_mean", "SS_total", "df_between",
                    "df_within", "F_stat", "p_value")
            summary = {k: float(row[k]) for k in keys if k in row.index}
        return df, summary

    @staticmethod
    def load_last_pkl(run_dir: str | Path) -> object:
        '''
        Load the highest-numbered gen_*.pkl from a run directory.

        Searches `run_dir` for the sub-directory holding the per-generation
        pickles (in order: generations/, opt_files/, the root itself).

        Args:
            run_dir: Path to the run folder under outputs/.

        Returns:
            The deserialized last-generation RuntimeData.

        Raises:
            FileNotFoundError: if the directory or any gen_*.pkl is missing.
        '''
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"[CL3O] Run directory not found.\n"
                f"| path : {run_dir}\n"
                f"Run a DE sweep first to produce the gen_*.pkl snapshots."
            )
        pkl_files: list[Path] = []
        for sub in StatsHelper._PKL_SUBDIRS:
            candidate = run_dir / sub if sub else run_dir
            if candidate.is_dir():
                pkl_files = sorted(candidate.glob("gen_*.pkl"))
                if pkl_files:
                    break
        if not pkl_files:
            raise FileNotFoundError(
                f"[CL3O] No gen_*.pkl files found in run directory.\n"
                f"| path : {run_dir}\n"
                f"Run a DE sweep first to produce the gen_*.pkl snapshots."
            )
        with open(pkl_files[-1], "rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def spearman_matrix(
        df       : pd.DataFrame,
        params   : list[str],
        outcomes : list[str],
    ) -> pd.DataFrame:
        '''
        Build a Spearman rank-correlation block (params x outcomes).

        Args:
            df:       Source DataFrame.
            params:   Row labels (column names present in df).
            outcomes: Column labels (column names present in df).

        Returns:
            DataFrame indexed by params, columns outcomes, of Spearman rho.
        '''
        rho = pd.DataFrame(index=params, columns=outcomes, dtype=float)
        for p in params:
            for o in outcomes:
                r, _ = spearmanr(df[p], df[o])
                rho.loc[p, o] = float(r)
        return rho


# ================================================================================
# Main - figure orchestrator
# ================================================================================

class RunStats:
    '''Render post-hoc CL3O result figures from archived artifacts.'''

    def __init__(self, data: StatsData, enable_logging: bool = True) -> None:
        self.data   = data
        self.logger = io.setup_logger(self, enable_logging)
        sns.set_theme(context="paper", style="whitegrid")

    # ---------------------------------------------------------------- helpers
    def _save(self, fig, name: str) -> None:
        '''Save and close a figure as <out_dir>/<name>.<fmt>.'''
        path = self.data.out_dir / f"{name}.{self.data.fmt}"
        fig.savefig(path, dpi=self.data.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"  saved -> {path}")

    # ------------------------------------------------------------------- LHS
    def plot_lhs(self) -> None:
        '''Render the four LHS-sweep figures (skips gracefully if missing).'''
        p = self.data.lhs_results
        if not p.is_file():
            self.logger.warning(
                f"[CL3O] LHS results.csv not found -- skipping LHS figures.\n"
                f"| path : {p}"
            )
            return
        df = StatsHelper.load_lhs_results(p)
        self._lhs_corr_heatmap(df)
        self._lhs_param_scatter(df)
        curves = StatsHelper.load_rate_curves(
            self.data.outputs_root, self.data.rate_pattern
        )
        if curves:
            self._lhs_convergence(curves, df)
        else:
            self.logger.warning(
                "[CL3O] No per-sample rate.csv found -- skipping convergence figure."
            )
        self._lhs_speed_ecdf(df)

    def _lhs_corr_heatmap(self, df: pd.DataFrame) -> None:
        params   = [c for c in _LHS_PARAMS   if c in df.columns]
        outcomes = [c for c in _LHS_OUTCOMES if c in df.columns]
        rho = StatsHelper.spearman_matrix(df, params, outcomes)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(rho, annot=True, fmt=".2f", cmap="vlag", center=0.0,
                    vmin=-1.0, vmax=1.0, ax=ax,
                    cbar_kws={"label": "Spearman rho"})
        ax.set_title("LHS hyper-parameter / outcome rank correlation")
        fig.tight_layout()
        self._save(fig, "lhs_corr_heatmap")

    def _lhs_param_scatter(self, df: pd.DataFrame) -> None:
        params = [c for c in _LHS_PARAMS if c in df.columns]
        hue    = "converged" if "converged" in df.columns else None
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        flat = axes.ravel()
        for ax, par in zip(flat, params):
            sns.scatterplot(data=df, x=par, y="best_f_final", hue=hue,
                            palette="Set1", ax=ax, legend=False)
            x = df[par].to_numpy(float)
            y = df["best_f_final"].to_numpy(float)
            if np.ptp(x) > 0:
                b, a = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, a + b * xs, color="black", lw=1.0, ls="--")
            ax.set_ylabel("best_f_final")
        for ax in flat[len(params):]:
            ax.set_visible(False)
        fig.suptitle("Hyper-parameter vs final fitness (colour = converged)")
        fig.tight_layout()
        self._save(fig, "lhs_param_scatter")

    def _lhs_convergence(
        self, curves: dict[int, pd.DataFrame], df: pd.DataFrame
    ) -> None:
        best_idx = None
        if {"best_f_final", "sample_idx"}.issubset(df.columns):
            best_idx = int(df.loc[df["best_f_final"].idxmin(), "sample_idx"])
        fig, ax = plt.subplots(figsize=(9, 6))
        for idx, rc in curves.items():
            is_best = idx == best_idx
            ax.semilogy(
                rc["k"], rc["best_f"],
                color  = "crimson" if is_best else "0.7",
                lw     = 1.8 if is_best else 0.8,
                zorder = 3 if is_best else 1,
                label  = f"#{idx} (best)" if is_best else None,
            )
            conv_rows = rc[rc["conv"] == "Y"]
            if not conv_rows.empty:
                kk = conv_rows.iloc[0]
                ax.scatter([kk["k"]], [kk["best_f"]], s=18,
                           color="black", zorder=4)
        ax.set_xlabel("Generation  k")
        ax.set_ylabel("best_f  [log scale]")
        ax.set_title(f"DE convergence -- {len(curves)} LHS samples")
        if best_idx is not None:
            ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "lhs_convergence")

    def _lhs_speed_ecdf(self, df: pd.DataFrame) -> None:
        if "n_gens" not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.ecdfplot(data=df, x="n_gens", ax=ax)
        ax.set_xlabel("Generations to stop  n_gens")
        ax.set_ylabel("ECDF")
        ax.set_title("Convergence-speed distribution across LHS samples")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, "lhs_speed_ecdf")
