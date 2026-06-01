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
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import spearmanr

# ================ Module imports ================

# Utilities
from cl3o.paths import ROOT_DIR, OUTPUTS_DIR
from cl3o.utils import io_utils as io

SAMPLE_CHOSEN = 3
RUN_NAME = "da62_opt-1"

# ================================================================================
# Matplotlib global setup
# ================================================================================

def _setup_mpl() -> None:
    '''Apply project-wide rcParams: serif/CM font, sizes. Call after sns.set_theme.'''
    plt.rcParams.update({
        "font.family"       : "serif",
        "mathtext.fontset"  : "cm",
        "axes.titlesize"    : 11,
        "axes.labelsize"    : 10,
        "xtick.labelsize"   : 9,
        "ytick.labelsize"   : 9,
        "legend.fontsize"   : 8,
    })


# ================================================================================
# Label maps  (pt-BR, LaTeX mathtext, ASCII-only source)
# ================================================================================

_PARAM_LABELS: dict[str, str] = {
    "NP"    : r"$N_P$",
    "CR"    : r"$CR$",
    "F"     : r"$F$",
    "lambda": r"$\lambda$",
}


# ================================================================================
# Formatting utilities
# ================================================================================

def _comma_fmt(precision: int = 2) -> mticker.FuncFormatter:
    '''Tick formatter replacing the decimal point with a comma.'''
    def _fmt(x: float, pos: int) -> str:
        return f"{x:.{precision}f}".replace(".", ",")
    return mticker.FuncFormatter(_fmt)


def _apply_comma(ax: plt.Axes, xp: int = 2, yp: int = 2) -> None:
    '''Apply comma-decimal formatters on both axes (skips log-scaled axes).'''
    if ax.get_xscale() != "log":
        ax.xaxis.set_major_formatter(_comma_fmt(xp))
    if ax.get_yscale() != "log":
        ax.yaxis.set_major_formatter(_comma_fmt(yp))


def _cfmt(v: float, prec: int = 3) -> str:
    '''Format a scalar float with comma decimal (for inline annotations).'''
    return f"{v:.{prec}f}".replace(".", ",")


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
_LHS_OUTCOMES = ["best_f_final", "feasible_f", "n_genss", "gen_to_half", "elapsed_s"]


# ================================================================================
# Helper - loaders and pure stats
# ================================================================================

class StatsHelper:
    '''Static loaders for CL3O result artifacts and small stats utilities.'''

    # Sub-directory names a run may use for its per-generation pickles.
    _PKL_SUBDIRS = ("generations", "opt_files", "")

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
    def _find_pkl_files(run_dir: Path) -> list[Path]:
        '''Return sorted gen_*.pkl paths from the first non-empty sub-directory.'''
        for sub in StatsHelper._PKL_SUBDIRS:
            candidate = run_dir / sub if sub else run_dir
            if candidate.is_dir():
                files = sorted(candidate.glob("gen_*.pkl"))
                if files:
                    return files
        return []

    @staticmethod
    def load_last_pkl(run_dir: str | Path) -> object:
        '''
        Load the highest-numbered gen_*.pkl from a run directory.

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
        pkl_files = StatsHelper._find_pkl_files(run_dir)
        if not pkl_files:
            raise FileNotFoundError(
                f"[CL3O] No gen_*.pkl files found in run directory.\n"
                f"| path : {run_dir}\n"
                f"Run a DE sweep first to produce the gen_*.pkl snapshots."
            )
        with open(pkl_files[-1], "rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def load_all_pkls(run_dir: str | Path) -> list[tuple[int, object]]:
        '''
        Load every gen_*.pkl from a run, returning sorted (k, rt) pairs.

        Args:
            run_dir: Path to the run folder under outputs/.

        Returns:
            List of (generation_index, RuntimeData) sorted by generation.

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
        pkl_files = StatsHelper._find_pkl_files(run_dir)
        if not pkl_files:
            raise FileNotFoundError(
                f"[CL3O] No gen_*.pkl files found in run directory.\n"
                f"| path : {run_dir}\n"
                f"Run a DE sweep first to produce the gen_*.pkl snapshots."
            )
        result: list[tuple[int, object]] = []
        for pf in pkl_files:
            try:
                k = int(pf.stem.split("_")[-1])
            except ValueError:
                k = len(result)
            with open(pf, "rb") as fh:
                result.append((k, pickle.load(fh)))
        return result

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
        sns.set_theme(context="paper", style="whitegrid", font="serif")
        _setup_mpl()

    # ---------------------------------------------------------------- helpers
    def _save(self, fig: plt.Figure, name: str) -> None:
        '''Save and close a figure as <out_dir>/<run_name>/<name>.<fmt>.'''
        path = self.data.out_dir / f"{RUN_NAME}" / f"{name}.{self.data.fmt}"
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
        df = pd.read_csv(p)
        
        self._lhs_param_scatter(df)

        curves = StatsHelper.load_rate_curves(
            self.data.outputs_root, self.data.rate_pattern
        )
        if curves:
            self._lhs_convergence(curves, df)
            self._lhs_convergence_expand(curves, df)
            self._lhs_convergence_expand_simple(curves, df)
        else:
            self.logger.warning(
                "[CL3O] No per-sample rate.csv found -- skipping convergence figure."
            )
        self._lhs_speed_ecdf(df)

    def _lhs_param_scatter(self, df: pd.DataFrame) -> None:
        params = [c for c in _LHS_PARAMS if c in df.columns]
        hue    = "converged" if "converged" in df.columns else None
        df = df.copy()
        # Composite score penalises slow convergence: f* * k_conv / k_max
        if {"best_f_final", "gen_to_half", "n_genss"}.issubset(df.columns):
            df["_score"] = df["best_f_final"] * df["gen_to_half"] / \
                           ( 400 * df["best_f_final"].min())
            y_col   = "_score"
            y_label = r"$\frac{f^* \cdot k_{\mathrm{conv}}}{f^*_{min} \cdot k_{\max}}$"
        else:
            y_col   = "best_f_final"
            y_label = r"$f^{\,*}$"
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        flat = axes.ravel()
        for ax, par in zip(flat, params):
            show_legend = hue is not None and ax is flat[0]
            sns.scatterplot(data=df, x=par, y=y_col, hue=hue,
                            palette="Set1", ax=ax, legend=show_legend)
            x = df[par].to_numpy(float)
            y = df[y_col].to_numpy(float)
            if np.ptp(x) > 0:
                b, a = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, a + b * xs, color="black", lw=1.0, ls="--")
            ax.set_xlabel(_PARAM_LABELS.get(par, par))
            ax.set_ylabel(y_label)
            ax.set_ylim([0.0, 1.0])
            _apply_comma(ax)
        for ax in flat[len(params):]:
            ax.set_visible(False)
        fig.suptitle("Hyper-parameter vs final fitness (colour = converged)")
        fig.tight_layout()
        self._save(fig, "lhs_param_scatter")

    def _lhs_convergence(
        self, curves: dict[int, pd.DataFrame], df: pd.DataFrame
    ) -> None:
        best_idx = SAMPLE_CHOSEN

        fig, ax = plt.subplots(figsize=(9, 6))
        for idx, rc in curves.items():
            is_best = idx == best_idx
            ax.plot(
                rc["k"], rc["best_f"],
                color  = "royalblue" if is_best else "0.7",
                lw     = 1.8 if is_best else 0.8,
                zorder = 3 if is_best else 1,
                label  = f"amostra {idx} (melhor)" if is_best else None,
            )
            conv_rows = rc[rc["conv"] == "Y"]
            if not conv_rows.empty:
                kk = conv_rows.iloc[0]
                ax.scatter([kk["k"]], [kk["best_f"]], s=18,
                           color="black", zorder=4)
        
        ax.set_xlabel(r"Geração $k$")
        ax.set_ylabel(r"$f^{\,*}$ [kg]")
        ax.set_title(f"Curvas de Convergência DE -- {len(curves)} amostras")
        if best_idx is not None:
            ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "lhs_convergence")

    def _lhs_convergence_expand(
        self, curves: dict[int, pd.DataFrame], df: pd.DataFrame
    ) -> None:
        '''
        Convergence curves with x-axis rescaled by total run time.

        Each sample's generation axis k is multiplied by that sample's total
        wall-clock time (elapsed_s), so the x-axis reads k * t_total.  Three
        samples are highlighted: the one with the minimum k_conv * t_total
        (crimson), the one with the lowest final best_f (royalblue), and the
        one with the lowest NP (darkorange).

        Args:
            curves: Dict of sample_idx -> rate DataFrame (columns k, best_f, conv).
            df:     LHS results table (must contain elapsed_s, gen_to_half,
                    sample_idx, best_f_final, NP).
        '''
        required = {"elapsed_s", "n_gens", "sample_idx"}
        if not required.issubset(df.columns):
            self.logger.warning(
                "[CL3O] results.csv missing columns for expand plot -- skipping."
            )
            return

        meta = df.set_index("sample_idx")[["elapsed_s", "n_gens"]].copy()
        meta["k_conv_time"] = meta["n_gens"] * meta["elapsed_s"] / 60

        valid = {idx: meta.loc[idx] for idx in curves if idx in meta.index}
        if not valid:
            return

        best_idx = min(valid, key=lambda i: valid[i]["k_conv_time"])

        df_valid = df[df["sample_idx"].isin(valid)].copy()

        # Sample with the lowest final best_f across the entire sweep.
        best_f_idx: int | None = None
        if "best_f_final" in df.columns and not df_valid.empty:
            best_f_idx = int(
                df_valid.loc[df_valid["best_f_final"].idxmin(), "sample_idx"]
            )

        # Sample with the lowest NP across the entire sweep.
        best_np_idx: int | None = None
        if "NP" in df.columns and not df_valid.empty:
            best_np_idx = int(
                df_valid.loc[df_valid["NP"].idxmin(), "sample_idx"]
            )

        fig, ax = plt.subplots(figsize=(9, 6))
        for idx, rc in curves.items():
            if idx not in valid:
                continue
            row        = valid[idx]
            t_total    = float(row["elapsed_s"] / 60)
            t_vals     = rc["k"].to_numpy(float) * t_total
            is_best    = idx == best_idx
            is_best_f  = idx == best_f_idx
            is_best_np = idx == best_np_idx
            if is_best:
                color, lw, zorder = "crimson", 1.8, 3
            elif is_best_f:
                color, lw, zorder = "royalblue", 1.8, 3
            elif is_best_np:
                color, lw, zorder = "darkorange", 1.8, 3
            elif idx == 8:
                color, lw, zorder = "darkgreen", 1.8, 3
            elif idx == 14:
                color, lw, zorder = "purple", 1.8, 3
            else:
                color, lw, zorder = "0.7", 0.8, 1
            label = None
            if is_best:
                label = (
                    rf"amostra {idx}: menor $k_{{\mathrm{{conv}}}}\!\cdot\!t$  "
                    rf"(${row['k_conv_time']:.0f}$)"
                )
            elif is_best_f:
                f_row = df[df["sample_idx"] == idx]
                f_val = float(f_row["best_f_final"].iloc[0]) if not f_row.empty else float("nan")
                label = rf"amostra {idx}: menor $f^*$  ($f^* = {_cfmt(f_val, 4)}$ kg)"
            elif is_best_np:
                np_row = df[df["sample_idx"] == idx]
                np_val = int(np_row["NP"].iloc[0]) if not np_row.empty else 0
                label = rf"amostra {idx}: menor $N_P$  ($N_P = {np_val}$)"
            elif idx == 8 or idx == 14:
                label = (
                    rf"amostra {idx}: baixo custo ($k_{{\mathrm{{conv}}}}\!\cdot\!t$ = "
                    rf"${row['k_conv_time']:.0f}$)"
                )
            highlighted = is_best or is_best_f or is_best_np or idx == 8 or idx == 14
            ax.plot(
                t_vals, rc["best_f"],
                color  = color,
                lw     = lw,
                zorder = zorder,
                label  = label,
            )
            # Mark k_conv * t_total point for every sample.
            t_conv    = float(row["k_conv_time"])
            conv_rows = rc[rc["conv"] == "Y"]
            if not conv_rows.empty:
                f_conv = float(conv_rows.iloc[0]["best_f"])
                ax.scatter(
                    [t_conv], [f_conv],
                    s      = 10,
                    color  = color if highlighted else "0.5",
                )

        ax.set_xlabel(r"$k \cdot t_{\mathrm{total}}$ [gen. $\times$ min]")
        ax.set_ylabel(r"$f^{\,*}$ [kg]")
        ax.set_title(
            r"Curvas de Convergência -- métrica $k \cdot t_{\mathrm{total}}$"
            f"\n{len(curves)} amostras"
        )
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "lhs_convergence_expand")

    def _lhs_convergence_expand_simple(
        self, curves: dict[int, pd.DataFrame], df: pd.DataFrame
    ) -> None:
        '''
        Generate all curves with sample labels
        '''
        required = {"elapsed_s", "n_gens", "sample_idx"}
        if not required.issubset(df.columns):
            self.logger.warning(
                "[CL3O] results.csv missing columns for expand plot -- skipping."
            )
            return

        meta = df.set_index("sample_idx")[["elapsed_s", "n_gens"]].copy()
        meta["k_conv_time"] = meta["n_gens"] * meta["elapsed_s"] / 60

        valid = {idx: meta.loc[idx] for idx in curves if idx in meta.index}
        if not valid:
            return

        fig, ax = plt.subplots(figsize=(9, 6))
        for idx, rc in curves.items():
            if idx not in valid:
                continue
            row        = valid[idx]
            t_total    = float(row["elapsed_s"] / 60)
            t_vals     = rc["k"].to_numpy(float) * t_total
        
            label = rf"amostra {idx}"
            ax.plot(t_vals, rc["best_f"], lw = 1.8, label = label)
            # Mark k_conv * t_total point for every sample.
            t_conv    = float(row["k_conv_time"])
            conv_rows = rc[rc["conv"] == "Y"]
            if not conv_rows.empty:
                f_conv = float(conv_rows.iloc[0]["best_f"])
                ax.scatter(
                    [t_conv], [f_conv],
                    s      = 10,
                    color  = "black",
                    zorder = 4,
                )

        ax.set_xlabel(r"$k \cdot t_{\mathrm{total}}$ [gen. $\times$ min]")
        ax.set_ylabel(r"$f^{\,*}$ [kg]")
        ax.set_title(
            r"Curvas de Convergência -- métrica $k \cdot t_{\mathrm{total}}$"
            f"\n{len(curves)} amostras"
        )
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "lhs_convergence_expand_simple")

    def _lhs_speed_ecdf(self, df: pd.DataFrame) -> None:
        if not {"n_gens", "elapsed_s"}.issubset(df.columns):
            return
        d = df.copy()
        # NP-proportional marker size; fall back to uniform if NP absent.
        if "NP" in d.columns:
            np_vals  = d["NP"].to_numpy(float)
            sizes    = 30 + 180 * (np_vals - np_vals.min()) / (np.ptp(np_vals) or 1.0)
            size_col = "NP"
        else:
            sizes    = 60
            size_col = None
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            d["n_gens"], d["elapsed_s"],
            s      = sizes,
            c      = np_vals if size_col else "steelblue",
            cmap   = "viridis" if size_col else None,
            vmin   = 10 if size_col else None,
            vmax   = 80 if size_col else None,
            alpha  = 0.8,
            zorder = 3,
        )
        # Annotate each point with sample index.
        for _, row in d.iterrows():
            ax.annotate(
                str(int(row["sample_idx"])),
                xy         = (row["gen_to_half"], row["elapsed_s"]),
                xytext     = (4, 4),
                textcoords = "offset points",
                fontsize   = 7,
                color      = "0.3",
            )
        if size_col:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(r"$N_P$", fontsize=9)
            cbar.set_ticks([10, 20, 30, 40, 50, 60, 70, 80])
        ax.set_xlabel(r"$k_{\mathrm{conv}}$")
        ax.set_ylabel(r"$t_{\mathrm{total}}$ [s]")
        ax.set_title(r"Custo de Convergência por Amostra LHS -- $k_{\mathrm{conv}}$ vs $t$")
        _apply_comma(ax, xp=0, yp=0)
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
        ax.set_xlabel(r"$\eta^2$")
        ax.set_ylabel("")
        title = r"Sensibilidade por Grupo Estrutural -- ANOVA Unidirecional"
        if summary is not None and "F_stat" in summary and "p_value" in summary:
            F = _cfmt(summary["F_stat"], 3)
            p = f"{summary['p_value']:.3g}".replace(".", ",")
            title += f"\n$F = {F}$,   $p = {p}$"
        ax.set_title(title)
        ax.xaxis.set_major_formatter(_comma_fmt(3))  # y-axis is categorical; only format x
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "anova_eta_sq")

    def _anova_group_means(self, df: pd.DataFrame) -> None:
        # Approximate box statistics from aggregate CSV columns.
        # Q1/Q3 use the normal-distribution factor (0.6745*sigma), giving the
        # exact quartiles for Gaussian data and a reasonable proxy otherwise.
        # Whiskers are clamped to the observed min/max reported by the ANOVA.
        d = df.dropna(subset=["mean_f", "std_f"]).reset_index(drop=True)
        colors = sns.color_palette("muted", len(d))
        stats = []
        for _, row in d.iterrows():
            mu, sigma = float(row["mean_f"]), float(row["std_f"])
            q1 = mu - 0.6745 * sigma
            q3 = mu + 0.6745 * sigma
            whislo = float(row["min_f"]) if "min_f" in row.index else q1 - 1.5 * (q3 - q1)
            whishi = float(row["max_f"]) if "max_f" in row.index else q3 + 1.5 * (q3 - q1)
            stats.append({
                "label"  : str(row["group"]),
                "med"    : mu,
                "q1"     : q1,
                "q3"     : q3,
                "whislo" : max(whislo, q1 - 1.5 * (q3 - q1)),
                "whishi" : min(whishi, q3 + 1.5 * (q3 - q1)),
                "fliers" : [],
            })
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.bxp(
            stats,
            showfliers=False,
            patch_artist=True,
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for element in ("medians", "whiskers", "caps"):
            for line in bp[element]:
                line.set_color("0.3")
                line.set_linewidth(1.2)
        ax.set_ylabel(r"Fitness $f$ (distribuicao por grupo)")
        ax.set_xlabel("")
        ax.set_title("Distribuicao de Fitness por Grupo Estrutural")
        ax.yaxis.set_major_formatter(_comma_fmt(2))
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "anova_group_means")

    # ----------------------------------------------------------- best design
    def plot_best_design(self) -> None:
        '''Render margins figures from all pickled generations of a run.'''
        run_dir = self.data.outputs_root / self.data.run_name
        try:
            all_pkls = StatsHelper.load_all_pkls(run_dir)
        except FileNotFoundError as exc:
            self.logger.warning(f"{exc}\n-- skipping best-design figures.")
            return
        self._design_margins_evolution(all_pkls)
        self._design_margins_dist(all_pkls[-1][1])

    def _design_margins_evolution(
        self, all_pkls: list[tuple[int, object]]
    ) -> None:
        '''MS_min of the best individual tracked over every generation.'''
        ks       = [k   for k, _  in all_pkls]
        tsw_mins = [float(getattr(rt.tsw,   "MS_min", np.nan)) for _, rt in all_pkls]
        dsp_mins = [float(getattr(rt.displ, "MS_min", np.nan)) for _, rt in all_pkls]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, mins, kind in (
            (axes[0], tsw_mins, "Tsai-Wu"),
            (axes[1], dsp_mins, "Deslocamento"),
        ):
            ax.plot(ks, mins, color="steelblue", lw=1.4)
            ax.axhline(0.0, color="crimson", lw=0.9, ls="--",
                       label=r"$MS = 0$ (limite)")
            ax.set_xlabel(r"Geração $k$")
            ax.set_ylabel(r"$MS_{\min}$")
            ax.set_title(f"Evolucao de $MS_{{\\min}}$ -- {kind}")
            ax.legend(fontsize=8)
            _apply_comma(ax)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            r"Margem de Seguranca Mínima do Melhor Indivíduo por Geração"
        )
        fig.tight_layout()
        self._save(fig, "design_margins_evolution")

    def _design_margins_dist(self, rt: object) -> None:
        '''Histograms of all MS values from the last generation, split by range.'''
        tsw_ms = np.concatenate([
            np.asarray(getattr(rt.tsw,   "MS_panels", []), float).ravel(),
            np.asarray(getattr(rt.tsw,   "MS_booms",  []), float).ravel(),
        ])
        dsp_ms = np.concatenate([
            np.asarray(getattr(rt.displ, "MS_u",  []), float).ravel(),
            np.asarray(getattr(rt.displ, "MS_th", []), float).ravel(),
        ])
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        specs = [
            (axes[0, 0], tsw_ms,  0.0, 100.0, "Tsai-Wu",     r"$MS$"),
            (axes[1, 0], dsp_ms,  0.0, 100.0, "Deslocamento", r"$MS$"),
        ]
        for ax, ms, lo, hi, kind, rng in specs:
            ms = ms[np.isfinite(ms)]
            subset = ms[(ms >= lo) & (ms < hi)] if hi is not None else ms[ms >= lo]
            if subset.size:
                sns.histplot(subset, ax=ax, bins=20, color="steelblue")
            ax.set_xlabel(r"Margem de Seguranca $MS$")
            ax.set_ylabel("Contagem")
            ax.set_title(f"{kind} -- {rng}")
            _apply_comma(ax)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(
            r"Distribuição de $MS$ -- Melhor Individuo"
        )
        fig.tight_layout()
        self._save(fig, "design_margins_dist")

    # ------------------------------------------------------------------- all
    def run_all(self) -> None:
        '''Render every available figure; skip sources with missing inputs.'''
        self.logger.info("Generating LHS sweep figures ...")
        self.plot_lhs()
        self.logger.info("Generating sensitivity figures ...")
        self.plot_sensitivity()
        self.logger.info("Generating best-design figures ...")
        # self.plot_best_design()
        # self.logger.info(f"Done. Figures written to {self.data.out_dir}")


# ================================================================================
# Entry point
# ================================================================================

if __name__ == "__main__":
    data = StatsData(
        aircraft = "da62",
        sweep    = "tune-de-3",
        # run_name = "da62_tune-de-3_LHS-0",
        run_name = RUN_NAME,
    )
    RunStats(data).run_all()
