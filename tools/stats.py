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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

# ======================== Module imports ========================

# Utilities
from cl3o.paths import ROOT_DIR, OUTPUTS_DIR
from cl3o.utils import io_utils as io

# ======================== Global variables ========================

_RUN_NAME   = "da62_opt-tune-5-lhs-17-seed-67 (83)"
_ANOVA_PATH = Path("sensitivity-100-0.15") / _RUN_NAME.removeprefix("da62_")

# ================================================================================
# Matplotlib global setup
# ================================================================================

def _setup_mpl() -> None:
    '''Apply project-wide rcParams: serif/CM font, sizes. Call after sns.set_theme.'''
    plt.rcParams.update({
        "font.family"       : "serif",
        "mathtext.fontset"  : "cm",
        "axes.titlesize"    : 18,
        "axes.labelsize"    : 14,
        "xtick.labelsize"   : 10,
        "ytick.labelsize"   : 10,
        "legend.fontsize"   : 12,
    })


# ================================================================================
# Label maps  (pt-BR, LaTeX mathtext, ASCII-only source)
# ================================================================================

_PARAM_LABELS: dict[str, str] = {
    "NP"    : r"$\mathrm{{NP}}$",
    "CR"    : r"$\mathrm{{CR}}$",
    "F"     : r"$\mathrm{{F}}$",
    "lambda": r"$\lambda$",
}

_PARAM_RANGES: dict[str, tuple] = {
    "NP"    : (16,  80),
    "CR"    : (0.5, 1.0),
    "F"     : (0.3, 1.5),
    "lambda": (0.0, 1.0),
}

# Grandeza -> LaTeX symbol maps for the FEA / cross-section validation figures.
# Keys are upper-cased before lookup so "uZ", "thX", "I1" all resolve.
_FEA_SYM: dict[str, str] = {
    "RX" : r"$R_x$", "RY" : r"$R_y$", "RZ" : r"$R_z$",
    "MX" : r"$M_x$", "MY" : r"$M_y$", "MZ" : r"$M_z$",
    "UX" : r"$u_x$", "UY" : r"$u_y$", "UZ" : r"$u_z$",
    "THX": r"$\theta_x$", "THY": r"$\theta_y$", "THZ": r"$\theta_z$",
}

# Reference DOFs for the FEA displacement-validation relative errors. The
# cantilever benchmark is bending-dominated, so linear displacements (uX, uY,
# uZ) are taken relative to the dominant transverse displacement uZ, and the
# rotations (thX, thY, thZ) relative to the dominant twist thX -- evaluated per
# caso (2D / 3D). Reaction rows keep their own-value relative error.
_FEA_DISP_GRANDEZAS = ("UX", "UY", "UZ")
_FEA_ROT_GRANDEZAS  = ("THX", "THY", "THZ")
_FEA_DISP_REF       = "UZ"
_FEA_ROT_REF        = "THX"

_GEOM_SYM: dict[str, str] = {
    "XC" : r"$X_C$", "ZC" : r"$Z_C$", "XS" : r"$X_S$", "ZS" : r"$Z_S$",
    "A"  : r"$A$",
    "IXX": r"$I_{XX}$", "IZZ": r"$I_{ZZ}$", "IXZ": r"$I_{XZ}$",
    "I1" : r"$I_1$", "I2" : r"$I_2$",
    "J"  : r"$J$", "THP": r"$\theta_P$",
}

_GEOM_CASE_ABBR: dict[str, str] = {
    "simetrico"   : "Sim.",
    "assimetrico" : "Assim.",
}

# Validation traffic-light palette (error / ratio severity).
_VAL_GREEN  = "#27ae60"
_VAL_ORANGE = "#f39c12"
_VAL_RED    = "#e74c3c"


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


def _erro_color(erro_pct: float) -> str:
    '''Traffic-light colour by absolute relative error: green<=2, orange<=10.'''
    a = abs(float(erro_pct))
    if a <= 5.0:
        return _VAL_GREEN
    if a <= 10.0:
        return _VAL_ORANGE
    return _VAL_RED


def _razao_color(ratio: float) -> str:
    '''Traffic-light colour by CL3O/FEMAP deviation: green<10%, orange<30%.'''
    d = abs(float(ratio) - 1.0) * 100.0
    if d < 10.0:
        return _VAL_GREEN
    if d < 30.0:
        return _VAL_ORANGE
    return _VAL_RED


def _erro_legend_handles() -> list[Patch]:
    '''Three Patch handles for the error/severity legend (pt-BR mathtext).'''
    return [
        Patch(facecolor=_VAL_GREEN,  label=r"$|e| \leq 5\%$"),
        Patch(facecolor=_VAL_ORANGE, label=r"$5\% < |e| \leq 10\%$"),
        Patch(facecolor=_VAL_RED,    label=r"$|e| > 10\%$"),
    ]


# ================================================================================
# Validation scatter encoders
# ================================================================================

def _build_point_encoders(
    df             : pd.DataFrame,
    marker_by_caso : dict[str, str],
    sym_map        : dict[str, str],
    case_abbr      : dict[str, str] | None = None,
) -> tuple:
    '''
    Build the (point_style, legend_handles) pair shared by the validation
    scatters (FEA beam and cross-section).

    Every (unidade, grandeza, caso) row gets its own colour, assigned fresh
    within each unidade panel from the 20-colour tab20 palette so even dense
    panels stay distinguishable; the marker shape echoes the caso. Legend
    entries mirror each point exactly (real colour + marker) and carry the
    caso (optionally abbreviated) inline.

    Args:
        df:             Validation table with columns unidade, grandeza, caso.
        marker_by_caso: Maps a caso to its marker glyph (fallback 's').
        sym_map:        Maps an upper-cased grandeza to its LaTeX symbol.
        case_abbr:      Optional caso -> short label map for legend text.

    Returns:
        Tuple (point_style, legend_handles): a row -> (color, marker) callable
        and a rows -> list[Line2D] legend builder.
    '''
    tab = list(plt.get_cmap("tab20").colors)
    color_by_key: dict[tuple[str, str, str], tuple] = {}
    for u in dict.fromkeys(df["unidade"].astype(str)):
        sub = df[df["unidade"].astype(str) == u]
        for i, (_, r) in enumerate(sub.iterrows()):
            color_by_key[(u, str(r["grandeza"]), str(r["caso"]))] = tab[i % len(tab)]

    def _key(r: pd.Series) -> tuple[str, str, str]:
        return (str(r["unidade"]), str(r["grandeza"]), str(r["caso"]))

    def point_style(r: pd.Series) -> tuple:
        return (color_by_key.get(_key(r), "0.5"),
                marker_by_caso.get(str(r["caso"]), "s"))

    def legend_handles(rows: pd.DataFrame) -> list[Line2D]:
        handles: list[Line2D] = []
        for _, r in rows.iterrows():
            sym  = sym_map.get(str(r["grandeza"]).upper(), str(r["grandeza"]))
            case = case_abbr.get(str(r["caso"]), str(r["caso"])) if case_abbr \
                else str(r["caso"])
            handles.append(Line2D(
                [], [], linestyle="None",
                marker=marker_by_caso.get(str(r["caso"]), "s"),
                color=color_by_key.get(_key(r), "0.5"),
                label=f"{sym} ({case})",
            ))
        return handles

    return point_style, legend_handles


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
    fea_csv         derived: out_dir/fea_validation/fea_validation.csv
    geom_csv        derived: out_dir/geom_validation/geom_validation.csv
    seed_csv        path to the seed-per-sample .xlsx (columns: sample, seed,
                    f*, k_conv, time [min]); None disables seed analysis
    '''
    aircraft     : str = "da62"
    sweep        : str = "tune-de-3"
    run_name     : str | None = None
    tools_out    : Path = field(default_factory=lambda: ROOT_DIR / "tools" / "output")
    outputs_root : Path = field(default_factory=lambda: OUTPUTS_DIR)
    out_dir      : Path | None = None
    dpi          : int = 150
    fmt          : str = "pdf"
    seed_csv     : Path | None = None

    lhs_results   : Path = field(init=False)
    anova_results : Path = field(init=False)
    anova_summary : Path = field(init=False)
    rate_pattern  : str  = field(init=False)
    fea_csv       : Path = field(init=False)
    geom_csv      : Path = field(init=False)

    def __post_init__(self) -> None:
        self.aircraft     = self.aircraft.lower()
        self.tools_out    = Path(self.tools_out)
        self.outputs_root = Path(self.outputs_root)
        if self.out_dir is None:
            self.out_dir = self.tools_out / "stats"
        self.out_dir = Path(self.out_dir)
        self.lhs_results   = self.tools_out / self.sweep / "results.csv"
        self.anova_results = self.tools_out / f"{_ANOVA_PATH}" / "anova_results.csv"
        self.anova_summary = self.tools_out / f"{_ANOVA_PATH}" / "anova_summary.csv"
        self.rate_pattern  = f"{self.aircraft}_{self.sweep}_LHS-*"
        self.fea_csv       = self.out_dir / "fea_validation"  / "fea_validation.csv"
        self.geom_csv      = self.out_dir / "geom_validation" / "geom_validation.csv"
        if self.run_name is None:
            self.run_name = f"{self.aircraft}_{self.sweep}_LHS-0"
        if self.seed_csv is not None:
            self.seed_csv = Path(self.seed_csv)
        self.out_dir.mkdir(parents=True, exist_ok=True)


# Hyper-parameter columns of the LHS results.csv.
_LHS_PARAMS = ["NP", "CR", "F", "lambda"]


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
    def fea_relative_errors(df: pd.DataFrame) -> pd.DataFrame:
        '''
        Recompute erro_pct for the FEA displacement rows against a per-case
        reference DOF.

        Linear displacements (uX, uY, uZ) are referenced to uZ and rotations
        (thX, thY, thZ) to thX, both taken from the FEMAP column of the same
        caso. Reaction rows are left untouched (they keep their own-value
        relative error). The signed error is 100 * (cl3o - femap) / |ref|.

        Args:
            df: FEA validation table with columns tipo, caso, grandeza,
                femap, cl3o, erro_pct.

        Returns:
            A copy of df with erro_pct overwritten for the displacement rows.
        '''
        d = df.copy()
        disp = d["tipo"].astype(str) == "Deslocamento"
        for _, sub in d[disp].groupby(d["caso"].astype(str)):
            grd = sub["grandeza"].astype(str).str.upper()
            ref_disp = sub.loc[grd == _FEA_DISP_REF, "femap"]
            ref_rot  = sub.loc[grd == _FEA_ROT_REF,  "femap"]
            ref_disp = float(ref_disp.iloc[0]) if not ref_disp.empty else None
            ref_rot  = float(ref_rot.iloc[0])  if not ref_rot.empty  else None
            for idx in sub.index:
                g   = str(d.at[idx, "grandeza"]).upper()
                ref = ref_rot if g in _FEA_ROT_GRANDEZAS else ref_disp
                if ref is None or abs(ref) < 1e-12:
                    continue
                d.at[idx, "erro_pct"] = (
                    100.0 * (float(d.at[idx, "cl3o"]) - float(d.at[idx, "femap"]))
                    / abs(ref)
                )
        return d

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
    def _resolve_pkl_files(run_dir: str | Path) -> list[Path]:
        '''
        Locate the gen_*.pkl snapshots of a run, raising if none are found.

        Args:
            run_dir: Path to the run folder under outputs/.

        Returns:
            Sorted list of gen_*.pkl paths.

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
        return pkl_files

    @staticmethod
    def load_last_pkl(run_dir: str | Path) -> object:
        '''Load the highest-numbered gen_*.pkl from a run directory.'''
        pkl_files = StatsHelper._resolve_pkl_files(run_dir)
        with open(pkl_files[-1], "rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def load_seed_sensitivity(path: str | Path) -> pd.DataFrame:
        '''
        Read and normalise the seed-per-sample Excel table.

        Expected columns: sample, seed, f*, k_conv, time [min].
        The ``sample`` column may have NaN in continuation rows (merged cells);
        those are forward-filled so every row carries its sample index.
        The result is always sorted by (sample, seed).

        Args:
            path: Path to the .xlsx file.

        Returns:
            Cleaned DataFrame with integer ``sample`` and ``seed`` columns.
        '''
        df = pd.read_excel(path)
        df["sample"] = df["sample"].ffill().astype(int)
        df["seed"]   = df["seed"].astype(int)
        return df.sort_values(["sample", "seed"]).reset_index(drop=True)

    @staticmethod
    def load_all_pkls(run_dir: str | Path) -> list[tuple[int, object]]:
        '''Load every gen_*.pkl from a run, returning sorted (k, rt) pairs.'''
        result: list[tuple[int, object]] = []
        for pf in StatsHelper._resolve_pkl_files(run_dir):
            try:
                k = int(pf.stem.split("_")[-1])
            except ValueError:
                k = len(result)
            with open(pf, "rb") as fh:
                result.append((k, pickle.load(fh)))
        return result


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
    def _save(self, fig: plt.Figure, name: str, subdir: str = _RUN_NAME) -> None:
        '''Save and close a figure as <out_dir>/<subdir>/<name>.<fmt>.'''
        path = self.data.out_dir / subdir / f"{name}.{self.data.fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.data.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"  saved -> {path}")

    def _read_optional_csv(
        self, path: Path, label: str, hint: str = ""
    ) -> pd.DataFrame | None:
        '''Read a CSV, or log a warning and return None when it is absent.'''
        if not path.is_file():
            msg = f"[CL3O] {label} not found -- skipping.\n| path : {path}"
            if hint:
                msg += f"\n{hint}"
            self.logger.warning(msg)
            return None
        return pd.read_csv(path)

    # ------------------------------------------------------------------- LHS
    def plot_lhs(self) -> None:
        '''Render the LHS-sweep figures (skips gracefully if missing).'''
        df = self._read_optional_csv(self.data.lhs_results, "LHS results.csv")
        if df is None:
            return

        self._lhs_param_scatter(df)

        curves = StatsHelper.load_rate_curves(
            self.data.outputs_root, self.data.rate_pattern
        )
        if curves:
            self._plot_convergence(
                curves, df, time_scaled=False, fname="lhs_convergence",
                xlabel=r"Geracao $k$", marker_size=18, conv_zorder=4,
            )
            self._plot_convergence(
                curves, df, time_scaled=True, fname="lhs_convergence_expand",
                xlabel=r"$k \cdot \Delta t$ [gen. $\times$ min]",
                marker_size=10, conv_zorder=1,
            )
            self._lhs_convergence_expand_simple(curves, df)
        else:
            self.logger.warning(
                "[CL3O] No per-sample rate.csv found -- skipping convergence figure."
            )
        self._lhs_speed_ecdf(df)
        self._lhs_speed_ecdf_3d(df)

    def _lhs_param_scatter(self, df: pd.DataFrame) -> None:
        params = [c for c in _LHS_PARAMS if c in df.columns]
        df = df.copy()
        # Composite score penalises slow convergence: ( f / f_min ) * ( k_conv / k_max )
        k_max = 400
        if {"best_f_final", "gen_to_half", "n_gens"}.issubset(df.columns):
            df["_score"] = df["best_f_final"] * df["n_gens"] / \
                           ( k_max * df["best_f_final"].min())
            y_col   = "_score"
            y_label = r"$k_\text{adm}$"
        else:
            y_col   = "best_f_final"
            y_label = r"$f^\star_{\mathrm{min}}$"
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        flat = axes.ravel()
        for ax, par in zip(flat, params):
            sns.scatterplot(data=df, x=par, y=y_col, color="#CA09A7", ax=ax)
            x = df[par].to_numpy(float)
            y = df[y_col].to_numpy(float)
            if np.ptp(x) > 0:
                b, a = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 2)
                ax.plot(xs, a + b * xs, color="#435421", lw=1.0, ls="--")
            ax.set_xlabel(_PARAM_LABELS.get(par, par))
            ax.set_ylabel(y_label)
            ax.set_ylim([0.0, 1.0])
            if par in _PARAM_RANGES:
                lo, hi = _PARAM_RANGES[par]
                margin = (hi - lo) * 0.05
                ax.set_xlim(lo - margin, hi + margin)
                ax.axvline(lo, color="k", lw=1.0, ls="--", zorder=3)
                ax.axvline(hi, color="k", lw=1.0, ls="--", zorder=3)
                n_ticks = 5
                ticks = np.linspace(lo, hi, n_ticks)
                if par == "NP":
                    ticks = ticks.astype(int)
                ax.set_xticks(ticks)
            _apply_comma(ax)
        for ax in flat[len(params):]:
            ax.set_visible(False)
        fig.tight_layout()
        self._save(fig, "lhs_param_scatter")

    def _resolve_highlights(
        self,
        curves: dict[int, pd.DataFrame],
        df: pd.DataFrame,
        num_low_costs: int,
    ) -> tuple[dict, int, int, int, int, tuple[int, int]]:
        '''
        Resolve the highlighted-sample indices and per-sample timing metadata.

        Returns (valid, best_idx, best_f_idx, best_np_idx), where valid maps
        each sample_idx present in both curves and the results table to its
        meta row (elapsed_s, n_gens, k_conv_time = n_gens * elapsed_s / 60).
        best_idx is the minimum-cost sample (k_conv_time), best_f_idx the lowest
        final fitness, and best_np_idx the smallest population NP; each is None
        when its column is absent or no valid sample exists.

        Args:
            curves: Dict of sample_idx -> rate DataFrame.
            df:     LHS results table.

        Returns:
            Tuple (valid, best_idx, best_f_idx, best_np_idx).
        '''
        valid: dict = {}
        best_idx = best_f_idx = min_np_idx = max_np_idx = None
        low_cost_idx = []
        if not {"elapsed_s", "n_gens", "sample_idx"}.issubset(df.columns):
            return valid, best_idx, best_f_idx, min_np_idx, max_np_idx, low_cost_idx

        meta = df.set_index("sample_idx")[["elapsed_s", "n_gens"]].copy()
        meta["k_conv_time"] = meta["n_gens"] * meta["elapsed_s"] / 60
        valid = {idx: meta.loc[idx] for idx in curves if idx in meta.index}
        if valid:
            best_idx = min(valid, key=lambda i: valid[i]["k_conv_time"])
            sorted_by_cost = sorted(valid, key=lambda i: valid[i]["k_conv_time"])
            low_cost_idx = [idx for idx in sorted_by_cost if idx != best_idx][:num_low_costs]
            df_valid = df[df["sample_idx"].isin(valid)]
            if "best_f_final" in df.columns and not df_valid.empty:
                best_f_idx = int(
                    df_valid.loc[df_valid["best_f_final"].idxmin(), "sample_idx"]
                )
            if "NP" in df.columns and not df_valid.empty:
                min_np_idx = int(
                    df_valid.loc[df_valid["NP"].idxmin(), "sample_idx"]
                )
                max_np_idx = int(
                    df_valid.loc[df_valid["NP"].idxmax(), "sample_idx"]
                )
        return valid, best_idx, best_f_idx, min_np_idx, max_np_idx, tuple(low_cost_idx)

    def _plot_convergence(
        self,
        curves      : dict[int, pd.DataFrame],
        df          : pd.DataFrame,
        time_scaled : bool,
        fname       : str,
        xlabel      : str,
        marker_size : float,
        conv_zorder : int,
    ) -> None:
        '''
        Render DE convergence curves (best_f vs generation), one per sample.

        With time_scaled=False the x-axis is the raw generation index k; with
        time_scaled=True it is k rescaled by the sample's wall-clock minutes
        (k * dt), and samples without a results row are dropped. Up to five
        samples are highlighted -- the lowest k_conv*dt (crimson), lowest final
        f* (royalblue), lowest NP (darkorange), plus fixed samples 8 (darkgreen)
        and 14 (purple) -- the rest are grey. Every curve marks its convergence
        generation.

        Args:
            curves:      Dict of sample_idx -> rate DataFrame (k, best_f, conv).
            df:          LHS results table (sample_idx, elapsed_s, n_gens,
                         best_f_final, NP).
            time_scaled: Scale the x-axis by per-sample wall-clock minutes.
            fname:       Output file stem.
            xlabel:      X-axis label.
            marker_size: Size of the convergence-point marker.
            conv_zorder: Draw order of the convergence-point marker.
        '''
        n_low_costs = 2
        valid, best_idx, best_f_idx, min_np_idx, max_np_idx, low_cost = \
            self._resolve_highlights(curves, df, n_low_costs)
        if time_scaled and not valid:
            self.logger.warning(
                "[CL3O] results.csv missing columns for expand plot -- skipping."
            )
            return

        # low_cost = (0, 4)

        def _style(idx: int) -> tuple:
            for key, style in (
                (best_idx,    ("crimson",    1.8, 3)),
                (best_f_idx,  ("royalblue",  1.8, 3)),
                (min_np_idx,  ("darkorange", 1.8, 3)),
                (max_np_idx,  ("brown",      1.8, 3)),
                (low_cost[0], ("darkgreen",  1.8, 3)),
                (low_cost[1], ("purple",     1.8, 3)),
            ):
                if key is not None and idx == key:
                    return (*style, True)
            return "0.7", 0.8, 1, False

        def _label(idx: int) -> str | None:
            def cost() -> str:
                return (rf"$k_{{\mathrm{{conv}}}}\cdot\Delta t$ = "
                        rf"${valid[idx]['k_conv_time']:.0f}$")
            if idx == best_idx and idx in valid:
                return rf"amostra {idx+1}: menor custo ({cost()})"
            if idx == best_f_idx:
                row   = df[df["sample_idx"] == idx]
                f_val = float(row["best_f_final"].iloc[0]) if not row.empty else float("nan")
                sub   = r"f^\star_{\mathrm{min}}"
                return rf"amostra {idx+1}: menor ${sub}$  (${sub} = {_cfmt(f_val, 2)}$)"
            if idx == min_np_idx:
                row    = df[df["sample_idx"] == idx]
                np_val = int(row["NP"].iloc[0]) if not row.empty else 0
                return rf"amostra {idx+1}: menor $\mathrm{{NP}}$  ($\mathrm{{NP}} = {np_val}$)"
            if idx == max_np_idx:
                row    = df[df["sample_idx"] == idx]
                np_val = int(row["NP"].iloc[0]) if not row.empty else 0
                return rf"amostra {idx+1}: maior $\mathrm{{NP}}$  ($\mathrm{{NP}} = {np_val}$)"
            if idx in low_cost and idx in valid:
                return rf"amostra {idx+1}: baixo custo ({cost()})"
            return None

        fig, ax = plt.subplots(figsize=(9, 6))
        for idx, rc in curves.items():
            if time_scaled and idx not in valid:
                continue
            color, lw, zorder, highlighted = _style(idx)
            k = rc["k"].to_numpy(float)
            x = k * float(valid[idx]["elapsed_s"] / 60) if time_scaled else k
            ax.plot(x, rc["best_f"], color=color, lw=lw, zorder=zorder, label=_label(idx))
            conv_rows = rc[rc["conv"] == "Y"]
            if not conv_rows.empty:
                cr     = conv_rows.iloc[0]
                x_conv = float(valid[idx]["k_conv_time"]) if time_scaled else float(cr["k"])
                ax.scatter([x_conv], [float(cr["best_f"])], s=marker_size,
                           color=color if highlighted else "0.5", zorder=conv_zorder)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$f^\star$ [kg]")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        self._save(fig, fname)

    def _lhs_convergence_expand_simple(
        self, curves: dict[int, pd.DataFrame], df: pd.DataFrame
    ) -> None:
        '''Time-scaled convergence curves, every sample labelled, no highlighting.'''
        valid, *_ = self._resolve_highlights(curves, df, 2)
        if not valid:
            self.logger.warning(
                "[CL3O] results.csv missing columns for expand plot -- skipping."
            )
            return

        fig, ax = plt.subplots(figsize=(9, 6))
        for idx, rc in curves.items():
            if idx not in valid:
                continue
            x = rc["k"].to_numpy(float) * float(valid[idx]["elapsed_s"] / 60)
            ax.plot(x, rc["best_f"], lw=1.8, label=rf"amostra {idx}")
            conv_rows = rc[rc["conv"] == "Y"]
            if not conv_rows.empty:
                ax.scatter(
                    [float(valid[idx]["k_conv_time"])],
                    [float(conv_rows.iloc[0]["best_f"])],
                    s=10, color="black", zorder=4,
                )

        ax.set_xlabel(r"$k \cdot \Delta t$ [gen. $\times$ min]")
        ax.set_ylabel(r"$f^\star$ [kg]")
        ax.legend()
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
            d["n_gens"], d["elapsed_s"] / 60,
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
                str(int(row["sample_idx"])+1),
                xy         = (row["n_gens"], row["elapsed_s"] / 60),
                xytext     = (5, 5),
                textcoords = "offset points",
                fontsize   = 7,
                color      = "0.3",
            )
        if size_col:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(r"$\mathrm{{NP}}$")
            cbar.set_ticks([10, 20, 30, 40, 50, 60, 70, 80])
        ax.set_xlabel(r"$k_{\mathrm{conv}}$")
        ax.set_ylabel(r"$\Delta t$ [min]")
        # ax.set_title(r"Custo de Convergência por Amostra LHS -- $k_{\mathrm{conv}}$ vs $t$")
        _apply_comma(ax, xp=0, yp=0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, "lhs_speed_ecdf")

    def _lhs_speed_ecdf_3d(self, df: pd.DataFrame) -> None:
        '''
        3-D scatter of convergence cost per LHS sample.

        Axes
        ----
        x  k_conv         (number of generations to convergence)
        y  Delta_t [min]  (elapsed wall-clock time)
        z  f*  [kg]       (best final fitness of the sample)

        Marker colour and size follow NP exactly as in ``_lhs_speed_ecdf``.
        Each point is annotated with its 1-based sample index.  The view
        elevation and azimuth are chosen so all three axes are readable without
        overlapping labels.

        Args:
            df: LHS results table (must contain n_gens, elapsed_s, best_f_final).
        '''
        required = {"n_gens", "elapsed_s", "best_f_final"}
        if not required.issubset(df.columns):
            self.logger.warning(
                "[CL3O] _lhs_speed_ecdf_3d: missing columns "
                f"{required - set(df.columns)} -- skipping."
            )
            return

        d = df.dropna(subset=list(required)).copy()
        if d.empty:
            return

        # NP-proportional marker size; fall back to uniform if NP absent.
        if "NP" in d.columns:
            np_vals  = d["NP"].to_numpy(float)
            sizes    = 30 + 180 * (np_vals - np_vals.min()) / (np.ptp(np_vals) or 1.0)
            size_col = "NP"
        else:
            np_vals  = None
            sizes    = 60
            size_col = None

        fig = plt.figure(figsize=(9, 6))
        ax  = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            d["n_gens"],
            d["elapsed_s"] / 60,
            d["best_f_final"],
            s      = sizes,
            c      = np_vals if size_col else "steelblue",
            cmap   = "viridis" if size_col else None,
            vmin   = 10 if size_col else None,
            vmax   = 80 if size_col else None,
            alpha  = 0.8,
            zorder = 3,
            depthshade = True,
        )

        # Annotate each point with 1-based sample index.
        for _, row in d.iterrows():
            idx = int(row["sample_idx"])
            
            if idx in (8, 9, 12, 13, 16, 18):
                if idx == 12:
                    ax.text(
                        float(row["n_gens"]),
                        float(row["elapsed_s"])*0.995 / 60,
                        float(row["best_f_final"])*1.005,
                        s          = str(idx + 1),
                        fontsize   = 7,
                        color      = "0.3",
                        ha         = "right",
                        va         = "bottom",
                    )
                elif idx == 13:
                    ax.text(
                        float(row["n_gens"])*1.01,
                        float(row["elapsed_s"])*1.1 / 60,
                        float(row["best_f_final"])*0.995,
                        s          = str(idx + 1),
                        fontsize   = 7,
                        color      = "0.3",
                        ha         = "left",
                        va         = "top",
                    )
                else:
                    ax.text(
                        float(row["n_gens"])*1.01,
                        float(row["elapsed_s"]) / 60,
                        float(row["best_f_final"])*0.995,
                        s          = str(idx + 1),
                        fontsize   = 7,
                        color      = "0.3",
                        ha         = "left",
                        va         = "top",
                    )
            else:
                ax.text(
                    float(row["n_gens"])*1.01,
                    float(row["elapsed_s"]) / 60,
                    float(row["best_f_final"])*1.005,
                    s          = str(idx + 1),
                    fontsize   = 7,
                    color      = "0.3",
                    ha         = "left",
                    va         = "bottom",
                )

        if size_col:
            cbar = fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.6)
            cbar.set_label(r"$\mathrm{NP}$")
            cbar.set_ticks([10, 20, 30, 40, 50, 60, 70, 80])

        ax.set_xlabel(r"$k_{\mathrm{conv}}$",labelpad=8)
        ax.set_ylabel(r"$\Delta t$ [min]",labelpad=8)
        ax.set_zlabel(r"$f^\star_{\mathrm{min}}$ [kg]",labelpad=8)
        
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: _cfmt(v, 0))
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: _cfmt(v, 1))
        )
        ax.zaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: _cfmt(v, 2))
        )
        pane_color = (0.985, 0.985, 0.985, 1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"].update({
                "color": (0.55, 0.55, 0.55, 0.18),
                "linewidth": 0.5,
            })
            axis.set_pane_color(pane_color)
        
        ax.tick_params(colors="0.25")
        ax.view_init(elev=15, azim=-30)
        fig.tight_layout()
        self._save(fig, "lhs_speed_ecdf_3d")

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
        # ax.set_title(title)
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
        # ax.set_title("Distribuicao de Fitness por Grupo Estrutural")
        ax.yaxis.set_major_formatter(_comma_fmt(2))
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=16)
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
            # ax.set_title(f"Evolucao de $MS_{{\\min}}$ -- {kind}")
            ax.legend()
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
            # ax.set_title(f"{kind} -- {rng}")
            _apply_comma(ax)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(
            r"Distribuição de $MS$ -- Melhor Individuo"
        )
        fig.tight_layout()
        self._save(fig, "design_margins_dist")

    # ------------------------------------------------------ validation (FEA / geom)
    def _val_scatter(
        self,
        groups          : list[dict],
        label_fn,
        suptitle        : str,
        subdir          : str,
        fname           : str,
        point_style_fn  = None,
        legend_handles_fn = None,
        layout          : tuple[int, int] = (2, 2),
        figsize         : tuple[float, float] = (11, 9),
    ) -> None:
        '''
        Render a grid of FEMAP-vs-CL3O scatters, one panel per magnitude group.

        Args:
            groups:            Up to layout[0]*layout[1] dicts with keys 'rows'
                               (DataFrame subset), 'scale' (divide raw values by
                               this) and 'unit' (axis tag).
            label_fn:          Maps a CSV row to its legend label (used only
                               when legend_handles_fn is None).
            suptitle:          Figure super-title.
            subdir:            Output sub-directory under out_dir.
            fname:             Output file stem.
            point_style_fn:    Optional row -> (color, marker) callable. When
                               None, each point cycles through tab10 with a
                               round 'o' marker (legacy behaviour).
            legend_handles_fn: Optional rows -> list[Line2D] callable building a
                               panel's legend handles. When None, every point is
                               labelled individually via label_fn.
            layout:            (nrows, ncols) subplot grid. Default 2x2.
            figsize:           Figure size in inches. Default (11, 9).
        '''
        tab = list(plt.get_cmap("tab10").colors)
        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
        flat = np.atleast_1d(axes).ravel()
        for ax, g in zip(flat, groups):
            rows  = g["rows"]
            scale = float(g["scale"])
            vals  : list[float] = []
            for i, (_, r) in enumerate(rows.iterrows()):
                xv = float(r["femap"]) / scale
                yv = float(r["cl3o"])  / scale
                if point_style_fn is not None:
                    color, marker = point_style_fn(r)
                else:
                    color, marker = tab[i % len(tab)], "o"
                ax.scatter([xv], [yv], color=color, marker=marker,
                           label=None if legend_handles_fn else label_fn(r),
                           zorder=3)
                vals.extend((xv, yv))
            if vals:
                lo, hi = min(vals), max(vals)
                pad    = (hi - lo) * 0.08 or 1.0
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                        color="0.5", ls="--", lw=1.0, zorder=1)
            ax.set_xlabel(rf"FEMAP {g['unit']}")
            ax.set_ylabel(rf"CL3O {g['unit']}")
            if legend_handles_fn is not None:
                handles = legend_handles_fn(rows)
                if handles:
                    ax.legend(handles=handles, loc="lower right",
                              ncol=2, fontsize=10, handletextpad=0.3,
                              columnspacing=0.8)
            else:
                ax.legend(loc="lower right")
            _apply_comma(ax, xp=0, yp=0)
            ax.grid(True, alpha=0.3)
        for ax in flat[len(groups):]:
            ax.set_visible(False)
        # fig.suptitle(suptitle)
        fig.tight_layout()
        self._save(fig, fname, subdir)

    def _val_erro_pct(
        self,
        panels       : list[pd.DataFrame],
        label_fn,
        suptitle     : str,
        subdir       : str,
        fname        : str,
        panel_titles : list[str] | None = None,
        xlim         : tuple[float, float] | None = None,
    ) -> None:
        '''
        Render a 1x2 horizontal error-bar chart (one panel per case group).

        Bars are sorted by signed erro_pct (largest on top), coloured by the
        traffic-light severity, with a solid 0-line and dashed +/-5% guides.

        Args:
            panels:       One DataFrame subset per panel (left, right).
            label_fn:     Maps a CSV row to its y-tick label.
            suptitle:     Figure super-title.
            subdir:       Output sub-directory under out_dir.
            fname:        Output file stem.
            panel_titles: Optional per-panel titles (e.g. left/right case).
            xlim:         Optional symmetric x-limits (e.g. (-25, 25)) applied
                          to every panel, centring the 0-error axis.
        '''
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for i, (ax, sub) in enumerate(zip(axes, panels)):
            d = sub.dropna(subset=["erro_pct"]).copy()
            d = d.sort_values("erro_pct", ascending=True)
            vals   = d["erro_pct"].to_numpy(float)
            labels = [label_fn(r) for _, r in d.iterrows()]
            colors = [_erro_color(v) for v in vals]
            y      = np.arange(len(d))
            ax.barh(y, vals, color=colors, zorder=3)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=14)
            if panel_titles is not None and i < len(panel_titles):
                ax.set_title(panel_titles[i])
            ax.axvline(0.0,  color="k", lw=1.0, zorder=2)
            ax.axvline(5.0,  color="0.3", lw=0.9, ls="--", zorder=2)
            ax.axvline(-5.0, color="0.3", lw=0.9, ls="--", zorder=2)
            ax.set_xlabel(r"Erro (%)")
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.xaxis.set_major_formatter(_comma_fmt(2))
            ax.grid(axis="x", alpha=0.3)
        axes[-1].legend(handles=_erro_legend_handles(),
                        loc="lower right")
        # fig.suptitle(suptitle)
        fig.tight_layout()
        self._save(fig, fname, subdir)

    def _val_razao(
        self,
        panels       : list[pd.DataFrame],
        label_fn,
        suptitle     : str,
        subdir       : str,
        fname        : str,
        panel_titles : list[str] | None = None,
    ) -> None:
        '''
        Render a 1x2 vertical CL3O/FEMAP ratio chart (one panel per case group).

        Rows with a non-positive or non-finite ratio (sign flip / zero FEMAP)
        are dropped. Bars are coloured by ratio deviation, with a dashed unity
        reference line.

        Args:
            panels:       One DataFrame subset per panel (left, right).
            label_fn:     Maps a CSV row to its x-tick label.
            suptitle:     Figure super-title.
            subdir:       Output sub-directory under out_dir.
            fname:        Output file stem.
            panel_titles: Optional per-panel titles (e.g. left/right case).
        '''
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for i, (ax, sub) in enumerate(zip(axes, panels)):
            d   = sub.copy()
            fem = d["femap"].to_numpy(float)
            cle = d["cl3o"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = cle / fem
            mask  = np.isfinite(ratio) & (ratio > 0.0)
            d     = d[mask]
            ratio = ratio[mask]
            labels = [label_fn(r) for _, r in d.iterrows()]
            colors = [_razao_color(r) for r in ratio]
            x      = np.arange(len(d))
            ax.bar(x, ratio, color=colors, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=16)
            if panel_titles is not None and i < len(panel_titles):
                ax.set_title(panel_titles[i])
            ax.axhline(1.0, color="k", lw=1.0, ls="--", zorder=2)
            ax.set_ylabel(r"CL3O / FEMAP")
            ax.yaxis.set_major_formatter(_comma_fmt(2))
            ax.grid(axis="y", alpha=0.3)
        # fig.suptitle(suptitle)
        fig.tight_layout()
        self._save(fig, fname, subdir)

    def plot_fea_validation(self) -> None:
        '''Render the three FEMAP-vs-CL3O beam-FEA validation figures.'''
        df = self._read_optional_csv(self.data.fea_csv, "fea_validation.csv")
        if df is None:
            return
        # Displacement/rotation errors are relative to the bending-dominant DOFs
        # (uZ / thX), recomputed from the raw FEMAP/CL3O columns per caso.
        df = StatsHelper.fea_relative_errors(df)

        def _label(r: pd.Series) -> str:
            sym = _FEA_SYM.get(str(r["grandeza"]).upper(), str(r["grandeza"]))
            return f"{r['caso']} - {sym}"

        # Per-point encoding: marker shape echoes the benchmark caso (2D = circle,
        # 3D = triangle) and each point gets its own colour within its unit panel.
        point_style, legend_handles = _build_point_encoders(
            df, marker_by_caso={"2D": "o", "3D": "^"}, sym_map=_FEA_SYM,
        )

        # Scatter grouped by unit; moments shown in kN.mm, rotations in degrees.
        specs = [
            ("N",     1.0,           r"[N]"),
            ("N*mm",  1.0e3,         r"[kN$\cdot$mm]"),
            ("mm",    1.0,           r"[mm]"),
            ("rad",   np.pi / 180.0, r"[$^\circ$]"),
        ]
        groups = [
            {"rows": df[df["unidade"] == u], "scale": s, "unit": lab}
            for u, s, lab in specs
        ]
        self._val_scatter(
            groups, _label,
            "Validação CL3O -- FEMAP vs CL3O por grupo de grandeza",
            "fea_validation", "fea_scatter",
            point_style_fn=point_style,
            legend_handles_fn=legend_handles,
        )

        panels = [df[df["tipo"] == "Reacao"], df[df["tipo"] == "Deslocamento"]]
        self._val_erro_pct(
            panels, _label,
            "Validação CL3O -- Erro relativo ao FEMAP",
            "fea_validation", "fea_erro_pct",
            panel_titles=["Reações", "Deslocamentos"],
            xlim=(-25.0, 25.0),
        )
        self._val_razao(
            panels, _label,
            "Validação CL3O -- Razao CL3O / FEMAP por Grandeza",
            "fea_validation", "fea_razao",
            panel_titles=["Reações", "Deslocamentos"],
        )

    def plot_geom_validation(self) -> None:
        '''Render the three FEMAP-vs-CL3O cross-section validation figures.'''
        df = self._read_optional_csv(self.data.geom_csv, "geom_validation.csv")
        if df is None:
            return

        def _sym(r: pd.Series) -> str:
            return _GEOM_SYM.get(str(r["grandeza"]).upper(), str(r["grandeza"]))

        def _scatter_label(r: pd.Series) -> str:
            case = _GEOM_CASE_ABBR.get(str(r["caso"]), str(r["caso"]))
            return f"{case} - {_sym(r)}"

        # Per-point encoding: marker shape echoes the case (sim. = circle,
        # assim. = triangle) and each point gets its own colour within its unit
        # panel; legend entries (abbreviated case inline) mirror each point.
        point_style, legend_handles = _build_point_encoders(
            df, marker_by_caso={"simetrico": "o", "assimetrico": "^"},
            sym_map=_GEOM_SYM, case_abbr=_GEOM_CASE_ABBR,
        )

        # Scatter grouped by unit; inertias/areas rescaled to readable decades.
        # Split into two side-by-side figures matching the former 2x2 columns:
        #   pt1 = left column  (mm, mm4),   pt2 = right column (mm2, graus).
        def _group(u: str, s: float, lab: str) -> dict:
            return {"rows": df[df["unidade"] == u], "scale": s, "unit": lab}

        left_groups = [
            _group("mm",  1.0,   r"[mm]"),
            _group("mm4", 1.0e5, r"[$\times 10^5$ mm$^4$]"),
        ]
        right_groups = [
            _group("mm2",   1.0e2, r"[$\times 10^2$ mm$^2$]"),
            _group("graus", 1.0,   r"[$^\circ$]"),
        ]
        for groups, fname in (
            (left_groups,  "geom_scatter_pt1"),
            (right_groups, "geom_scatter_pt2"),
        ):
            self._val_scatter(
                groups, _scatter_label,
                "Validação Secao Transversal -- FEMAP vs CL3O por grupo de grandeza",
                "geom_validation", fname,
                point_style_fn=point_style,
                legend_handles_fn=legend_handles,
                layout=(1, 2),
                figsize=(12, 5.5),
            )

        panels = [df[df["caso"] == "simetrico"], df[df["caso"] == "assimetrico"]]
        self._val_erro_pct(
            panels, _sym,
            "Validação Secao Transversal -- Erro relativo ao FEMAP",
            "geom_validation", "geom_erro_pct",
            panel_titles=["Seção Simétrica", "Seção Assimétrica"],
        )
        self._val_razao(
            panels, _sym,
            "Validação Secao Transversal -- Razao CL3O / FEMAP por Grandeza",
            "geom_validation", "geom_razao",
            panel_titles=["Seção Simétrica", "Seção Assimétrica"],
        )

    # ------------------------------------------------------- seed sensitivity
    def plot_seed_sensitivity(self) -> None:
        '''
        Render the seed-sensitivity figure (skips gracefully if missing).

        Figure produced
        ---------------
        seed_f_boxplot  -- boxplot of f* per sample with per-seed strip
                           annotations and a secondary axis for computational
                           cost k_conv * Delta_t [gen x min].
        '''
        if self.data.seed_csv is None or not Path(self.data.seed_csv).is_file():
            self.logger.warning(
                "[CL3O] seed_csv not configured or file not found -- "
                "skipping seed-sensitivity figures.\n"
                f"| path : {self.data.seed_csv}"
            )
            return
        df = StatsHelper.load_seed_sensitivity(self.data.seed_csv)
        self._seed_f_boxplot(df)

    def _seed_f_boxplot(self, df: pd.DataFrame) -> None:
        '''
        Boxplot of f* per LHS sample with a colour-coded strip per seed.

        One box per sample (x-tick = bare sample index). Each seed gets a
        fixed colour from tab10; strip points are jittered horizontally.
        A single legend maps colour to seed value.

        Args:
            df: Cleaned seed table with columns sample, seed, f*
                (as returned by StatsHelper.load_seed_sensitivity).
        '''
        d       = df.copy()
        samples = sorted(d["sample"].unique())
        seeds   = sorted(d["seed"].unique())
        n       = len(samples)
        palette = {s: c for s, c in zip(seeds, plt.get_cmap("tab10").colors)}
        x_pos   = {s: i for i, s in enumerate(samples)}

        fig, ax = plt.subplots(figsize=(max(5, n * 2.5), 5))

        # ---- boxplot ----------------------------------------------------------
        box_data = [d.loc[d["sample"] == s, "f*"].to_numpy(float) for s in samples]
        bp = ax.boxplot(
            box_data,
            positions    = list(range(n)),
            widths       = 0.4,
            patch_artist = True,
            showfliers   = False,
            zorder       = 2,
        )
        for patch in bp["boxes"]:
            patch.set(facecolor="steelblue", alpha=0.30)
        for element in ("medians", "whiskers", "caps"):
            for line in bp[element]:
                line.set(color="0.3", linewidth=1.2)

        # ---- strip points coloured by seed ------------------------------------
        for s in samples:
            xi  = x_pos[s]
            sub = d[d["sample"] == s].sort_values("seed")
            for (_, row) in sub.iterrows():
                ax.scatter(
                    xi, float(row["f*"]),
                    color  = palette[row["seed"]],
                    s      = 20,
                    marker = 'd',
                    zorder = 4,
                )

        # ---- legend -----------------------------------------------------------
        handles = [
            Line2D([], [], linestyle="None", marker="d",
                   color=palette[s], label=f"seed {s}")
            for s in seeds
        ]
        ax.legend(handles=handles, fontsize=9, loc="upper right")

        # ---- formatting -------------------------------------------------------
        ax.set_xticks(range(n))
        ax.set_yticks([33.5, 34, 34.5, 35, 35.5, 36, 36.5])
        ax.set_xticklabels([str(s) for s in samples])
        ax.set_xlabel(r"Amostra")
        ax.set_ylabel(r"$f^\star_{\mathrm{min}}$ [kg]")
        ax.yaxis.set_major_formatter(_comma_fmt(2))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "seed_f_boxplot")

    # ------------------------------------------------------------------- all
    def run_all(self) -> None:
        '''Render every available figure; skip sources with missing inputs.'''
        self.logger.info("Generating LHS sweep figures ...")
        self.plot_lhs()
        self.logger.info("Generating sensitivity figures ...")
        self.plot_sensitivity()
        self.logger.info("Generating FEA validation figures ...")
        self.plot_fea_validation()
        self.logger.info("Generating cross-section validation figures ...")
        self.plot_geom_validation()
        self.logger.info("Generating seed-sensitivity figures ...")
        self.plot_seed_sensitivity()
        self.logger.info("Generating best-design figures ...")
        # self.plot_best_design()
        # self.logger.info(f"Done. Figures written to {self.data.out_dir}")


# ================================================================================
# Entry point
# ================================================================================

if __name__ == "__main__":
    data = StatsData(
        aircraft = "da62",
        sweep    = "tune-de-5",
        # run_name = "da62_tune-de-3_LHS-0",
        run_name = _RUN_NAME,
        seed_csv = ROOT_DIR / "tools" / "output" / "seed_por_sample.xlsx",
    )
    RunStats(data).run_all()
