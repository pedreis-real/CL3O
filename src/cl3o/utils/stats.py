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
        self.aircraft     = self.aircraft.lower()
        self.tools_out    = Path(self.tools_out)
        self.outputs_root = Path(self.outputs_root)
        if self.out_dir is None:
            self.out_dir = self.tools_out / "stats"
        self.out_dir = Path(self.out_dir)
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
    def _save(self, fig: plt.Figure, name: str) -> None:
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
            # One legend total (on the first panel) so the "colour = converged"
            # caption is decodable without cluttering every subplot.
            show_legend = hue is not None and ax is flat[0]
            sns.scatterplot(data=df, x=par, y="best_f_final", hue=hue,
                            palette="Set1", ax=ax, legend=show_legend)
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

    # ----------------------------------------------------------- sensitivity
    def plot_sensitivity(self) -> None:
        '''Render the two ANOVA figures (skips gracefully if missing).'''
        p = self.data.anova_results
        if not p.is_file():
            self.logger.warning(
                f"[CL3O] anova_results.csv not found -- skipping sensitivity figures.\n"
                f"| path : {p}\n"
                f"Run tools.sensitivity_analysis (variant-A schema) first."
            )
            return
        df, summary = StatsHelper.load_anova(p, self.data.anova_summary)
        # The legacy producer appends a trailing "ANOVA" summary row (with a
        # packed "F=.. p=.." eta_sq string and "nan" cells). Drop it and coerce
        # the numeric columns so it never renders as a phantom structural group;
        # this is a no-op for the clean variant-A schema.
        df = df[df["group"].astype(str).str.upper() != "ANOVA"].copy()
        for col in ("eta_sq", "mean_f", "std_f"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        self._anova_eta_sq(df, summary)
        self._anova_group_means(df)

    def _anova_eta_sq(self, df: pd.DataFrame, summary: dict | None) -> None:
        d = df.dropna(subset=["eta_sq"]).sort_values("eta_sq")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=d, y="group", x="eta_sq", hue="group",
                    palette="viridis", legend=False, ax=ax)
        ax.set_xlabel("eta^2  (ANOVA effect size)")
        ax.set_ylabel("")
        title = "Structural group sensitivity -- one-way ANOVA"
        if summary is not None and "F_stat" in summary and "p_value" in summary:
            title += f"\nF = {summary['F_stat']:.3f},  p = {summary['p_value']:.3g}"
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "anova_eta_sq")

    def _anova_group_means(self, df: pd.DataFrame) -> None:
        d = df.dropna(subset=["mean_f"])
        colors = sns.color_palette("muted", len(d))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(d["group"], d["mean_f"],
               yerr=d["std_f"] if "std_f" in d.columns else None,
               capsize=4, color=colors)
        ax.set_ylabel("Fitness  mean_f  (+/- std_f)")
        ax.set_xlabel("")
        ax.set_title("Per-group fitness mean and dispersion")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "anova_group_means")

    # ----------------------------------------------------------- best design
    def plot_best_design(self) -> None:
        '''Render the four best-design figures from a run's last pkl.'''
        run_dir = self.data.outputs_root / self.data.run_name
        try:
            rt = StatsHelper.load_last_pkl(run_dir)
        except FileNotFoundError as exc:
            self.logger.warning(f"{exc}\n-- skipping best-design figures.")
            return
        self._design_mass(rt)
        self._design_margins(rt)
        self._design_panel_stress(rt)
        self._design_forces(rt)

    def _design_mass(self, rt: object) -> None:
        sco     = rt.score
        panels  = float(np.nansum(np.asarray(getattr(sco, "panels", []), float)))
        flanges = float(np.nansum(np.asarray(getattr(sco, "flanges", []), float)))
        total   = float(getattr(sco, "total", panels + flanges))
        labels  = ["Panels (skin+webs)", "Flanges", "Total"]
        vals    = [panels, flanges, total]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, vals, color=sns.color_palette("crest", 3))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Mass  [kg]")
        ax.set_title("Best-design structural mass breakdown")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "design_mass")

    def _design_margins(self, rt: object) -> None:
        tsw, dsp = rt.tsw, rt.displ
        tsw_ms = np.concatenate([
            np.asarray(getattr(tsw, "MS_panels", []), float).ravel(),
            np.asarray(getattr(tsw, "MS_booms",  []), float).ravel(),
        ])
        dsp_ms = np.concatenate([
            np.asarray(getattr(dsp, "MS_u",  []), float).ravel(),
            np.asarray(getattr(dsp, "MS_th", []), float).ravel(),
        ])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        panels = (
            (axes[0], tsw_ms, "Tsai-Wu margins of safety",
             getattr(tsw, "MS_min", None)),
            (axes[1], dsp_ms, "Displacement margins of safety",
             getattr(dsp, "MS_min", None)),
        )
        for ax, ms, title, mmin in panels:
            ms = ms[np.isfinite(ms)]
            if ms.size:
                sns.histplot(ms, ax=ax, bins=30, color="steelblue")
            if mmin is not None and np.isfinite(float(mmin)):
                ax.axvline(float(mmin), color="crimson", ls="--", lw=1.5,
                           label=f"MS_min = {float(mmin):.3f}")
                ax.legend(fontsize=8)
            ax.axvline(0.0, color="black", lw=0.8)
            ax.set_xlabel("Margin of safety")
            ax.set_title(title)
        fig.suptitle("Best-design margins of safety  (MS < 0 = failure)")
        fig.tight_layout()
        self._save(fig, "design_margins")

    def _design_panel_stress(self, rt: object) -> None:
        tauA = np.asarray(rt.stress.tauA, float)
        tauB = np.asarray(rt.stress.tauB, float)
        lc   = 0
        if tauA.ndim == 3:
            tau = np.maximum(np.abs(tauA[:, :, lc]), np.abs(tauB[:, :, lc]))
        else:
            tau = np.maximum(np.abs(tauA), np.abs(tauB))
        coord = np.asarray(rt.mesh.coord, float)
        conn  = np.asarray(rt.mesh.conn, int)[:, :2]
        mid_y = np.abs(0.5 * (coord[conn[:, 0], 1] + coord[conn[:, 1], 1]))
        order = np.argsort(mid_y)
        t2     = getattr(rt.sections.sec_data[0], "T2", [])
        labels = [p.get("label", f"P{j}") for j, p in enumerate(t2)]
        fig, ax = plt.subplots(figsize=(9, 5))
        for j in range(tau.shape[1]):
            lbl = labels[j] if j < len(labels) else f"P{j}"
            ax.plot(mid_y[order], tau[order, j], lw=1.0, label=lbl)
        ax.set_xlabel("Spanwise  |Y|  [mm]")
        ax.set_ylabel("max|tau|  [MPa]")
        ax.set_title("Per-panel shear stress along span (load case 0)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, "design_panel_stress")

    def _design_forces(self, rt: object) -> None:
        fr  = rt.fea_rts
        Qsc = np.asarray(fr.Q_sc, float)
        Qc  = np.asarray(fr.Q_c,  float)
        lc  = 0
        coord = np.asarray(rt.mesh.coord, float)
        conn  = np.asarray(rt.mesh.conn, int)[:, :2]
        mid_y = np.abs(0.5 * (coord[conn[:, 0], 1] + coord[conn[:, 1], 1]))
        order = np.argsort(mid_y)
        span  = mid_y[order]

        def comp(Q, row):
            return Q[row, order, lc] if Q.ndim == 3 else Q[row, order]

        series = {
            "N  [N]":       comp(Qc,  0),
            "Sy  [N]":      comp(Qsc, 1),
            "Sz  [N]":      comp(Qsc, 2),
            "T  [N.mm]":    comp(Qsc, 3),
            "My  [N.mm]":   comp(Qc,  4),
            "Mz  [N.mm]":   comp(Qc,  5),
        }
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, (name, vals) in zip(axes.ravel(), series.items()):
            ax.plot(span, vals, color="navy", lw=1.4)
            ax.axhline(0.0, color="black", lw=0.6)
            ax.set_title(name)
            ax.set_xlabel("|Y|  [mm]")
            ax.grid(True, alpha=0.3)
        fig.suptitle("Best-design internal force / moment diagrams (load case 0)")
        fig.tight_layout()
        self._save(fig, "design_forces")

    # ------------------------------------------------------------------- all
    def run_all(self) -> None:
        '''Render every available figure; skip sources with missing inputs.'''
        self.logger.info("Generating LHS sweep figures ...")
        self.plot_lhs()
        self.logger.info("Generating sensitivity figures ...")
        self.plot_sensitivity()
        self.logger.info("Generating best-design figures ...")
        self.plot_best_design()
        self.logger.info(f"Done. Figures written to {self.data.out_dir}")


# ================================================================================
# Entry point
# ================================================================================

if __name__ == "__main__":
    data = StatsData(
        aircraft = "da62",
        sweep    = "tune-de-3",
        run_name = "da62_tune-de-3_LHS-0",
    )
    RunStats(data).run_all()
