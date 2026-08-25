import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import polars as pl

from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector, SpanSelector, Cursor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde


from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *


# ============================================================================
# Base selector
# ============================================================================
class _BaseSelector:
    """Common plumbing: id, styling, selection state, df access."""

    _ids = itertools.count(1)

    # shared style
    ACCENT = "#002fff"
    ACCENT_SOFT = "#002fff4b"
    SPAN_FACE = "#7E7EE9"
    BG = "#e3e4e7"
    GREY_OUT = "darkgrey"

    def __init__(self, df: pl.DataFrame, limits=None, **kwargs):
        self.id = next(self._ids)
        self.df = self._ensure_polars(df)
        self.kwargs = kwargs
        self.cmap = plt.get_cmap(kwargs.get("cmap", "magma"))
        self.selection = {"mask": None, "df": None}
        self.fig = None
        self.ax = None
        if limits is not None:
            self._check_limits(limits)

    @staticmethod
    def _ensure_polars(df) -> pl.DataFrame:
        """Accept polars DataFrames, loader Input wrappers, or pandas frames."""
        if isinstance(df, pl.DataFrame):
            return df
        if hasattr(df, "df") and isinstance(getattr(df, "df"), pl.DataFrame):
            return df.df
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                return pl.from_pandas(df)
        except ImportError:
            pass
        raise TypeError(f"Expected a polars DataFrame, got {type(df).__name__}.")

    def _check_limits(self, limits):
        if len(limits) == 2 and all(isinstance(v, (int, float)) for v in limits):
            self.selection.update({"min": limits[0], "max": limits[1]})
            if self.selection["min"] > self.selection["max"]:
                self.selection["min"], self.selection["max"] = self.selection["max"], self.selection["min"]
        else:
            for i in limits:
                if not len(i) == 2:
                    raise InvalidParameterValueError(f"limits must be a list of numeric pairs; got {i!r}")
                if not isinstance(i[0], (int, float)) or not isinstance(i[1], (int, float)):
                    raise InvalidParameterValueError(f"limits must be numeric values; got {i!r}")
            self.selection.update({"verts": limits})

    # ---- helpers -----------------------------------------------------------
    def _style_ax(self, ax, grid_axis="both"):
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_facecolor(self.BG)
        ax.grid(True, axis=grid_axis, which="major", color="white", linewidth=0.5, zorder=0)
        ax.grid(True, axis=grid_axis, which="minor", color="white", linewidth=0.25, zorder=0, alpha=0.75)

    @staticmethod
    def _validate_log(*flags):
        for f in flags:
            if not isinstance(f, bool):
                raise InvalidParameterValueError(f"{f!r} is not a valid log value; must be boolean")

    def _label_kwargs(self, **overrides):
        kw = dict(color=self.ACCENT, fontsize=10, fontstyle="italic",
                  ha="center", va="bottom", zorder=100000, clip_on=False)
        kw.update(overrides)
        return kw

    def _sample(self, df: pl.DataFrame, n: int = 10000, frac: float = None, **kw) -> pl.DataFrame:
        _seed = kw.get("random_state", 42)
        if frac is not None:
            if not isinstance(frac, float) or not (0 < frac <= 1):
                raise InvalidParameterValueError(f"sample_fraction must be a float between 0 and 1; got {frac!r}")
            return df.sample(fraction=frac, seed=_seed)
        elif df.height > n:
            return df.sample(n=n, seed=_seed)
        else:
            return df

    @staticmethod
    def _col_f64(df: pl.DataFrame, col: str) -> np.ndarray:
        """Column as float64 numpy array (nulls -> NaN)."""
        return df[col].cast(pl.Float64, strict=False).to_numpy()

    def _filter_verts(self, df: pl.DataFrame = None) -> pl.DataFrame:
        """Apply the selected polygon gate to the *full* dataframe."""
        verts = self.selection.get("verts")
        source = self.df if df is None else self._ensure_polars(df)
        if verts is None:
            return source

        xv = self._col_f64(source, self._x)
        yv = self._col_f64(source, self._y)

        finite = np.isfinite(xv) & np.isfinite(yv)
        if self._log_x:
            finite &= xv > 0
        if self._log_y:
            finite &= yv > 0

        inside = np.zeros(source.height, dtype=bool)
        inside[finite] = MplPath(verts).contains_points(
            np.column_stack([xv[finite], yv[finite]])
        )
        return source.filter(pl.Series(inside))

    def _filter_minmax(self, df: pl.DataFrame = None) -> pl.DataFrame:
        """Apply the selected min/max range to the *full* dataframe."""
        source = self.df if df is None else self._ensure_polars(df)
        min_val, max_val = self.selection.get("min"), self.selection.get("max")
        if min_val is None or max_val is None:
            return source
        return source.filter(pl.col(self._metric).is_between(min_val, max_val))

    # ---- public API --------------------------------------------------------
    @property
    def result(self) -> dict:
        """Filtered dataframe (raises if nothing selected yet)."""
        return self.selection

    @property
    def mask(self) -> np.ndarray:
        return self.selection["mask"]

    @property
    def limits(self) -> dict:
        if self.selection.get("verts", None) is not None:
            return {"verts": self.selection["verts"]}
        elif self.selection.get("min", None) is not None and self.selection.get("max", None) is not None:
            return {"min": self.selection.get("min", None), "max": self.selection.get("max", None)}
        else:
            return {}

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id})"


# ============================================================================
# 2D polygon gate
# ============================================================================
class GateSelector(_BaseSelector):
    """2D density scatter with polygon gating."""

    def show(self, x: str, y: str, *, log: bool = False, **kwargs):
        kw = {**self.kwargs, **kwargs}
        log_x, log_y = kw.get("log_x", log), kw.get("log_y", log)
        self._validate_log(log_x, log_y)

        # remember gate geometry so we can filter the FULL df on the backend
        self._x, self._y = x, y
        self._log_x, self._log_y = log_x, log_y

        df = self._sample(self.df, **kw)

        xdata, ydata = self._col_f64(df, x), self._col_f64(df, y)
        finite = np.isfinite(xdata) & np.isfinite(ydata)
        if log_x: finite &= xdata > 0
        if log_y: finite &= ydata > 0
        df = df.filter(pl.Series(finite))
        xs, ys = xdata[finite], ydata[finite]

        xk = np.log(xs) if log_x else xs
        yk = np.log(ys) if log_y else ys

        bw = kw.get("kde_bw", "scott")
        dens = gaussian_kde(np.vstack([xk, yk]), bw_method=bw)(np.vstack([xk, yk]))

        self.fig, self.ax = plt.subplots(figsize=kw.get("figsize", (7, 7)))
        fig, ax = self.fig, self.ax
        pts = ax.scatter(xs, ys, c=dens, s=0.85, cmap=self.cmap, zorder=3)
        ax.set_xlabel(x); ax.set_ylabel(y)
        if log_x: ax.set_xscale("log", nonpositive="clip")
        if log_y: ax.set_yscale("log", nonpositive="clip")
        self._style_ax(ax)

        # marginal KDEs
        divider = make_axes_locatable(ax)
        for side, vals, is_log in [("top", xk, log_x), ("right", yk, log_y)]:
            m_ax = divider.append_axes(side, size="7%", pad=0,
                                       sharex=ax if side == "top" else None,
                                       sharey=ax if side == "right" else None)
            grid = np.linspace(vals.min(), vals.max(), 400)
            d = gaussian_kde(vals, bw_method=bw)(grid)
            g = np.exp(grid) if is_log else grid
            m_ax.plot(*((g, d) if side == "top" else (d, g)), color="grey", lw=1)
            m_ax.set_axis_off()

        base_colors = pts.get_facecolors().copy()
        # ensure one color per point (scatter may collapse to a single row)
        if len(base_colors) == 1 and len(xs) > 1:
            base_colors = np.repeat(base_colors, len(xs), axis=0)

        # preserve any predefined verts passed via `limits`
        preset_verts = self.selection.get("verts")
        self.selection = {"mask": None, "verts": preset_verts, "df": None}

        fig._cursor = Cursor(ax, useblit=True, color=self.ACCENT_SOFT,
                             linewidth=1, horizOn=True, vertOn=True)

        # Overlay axes in linear [0, 1] display space so the polygon selector
        # moves/stretches uniformly in pixels regardless of log scaling on `ax`.
        ax_sel = fig.add_axes(ax.get_position(), frameon=False)
        ax_sel.set_axes_locator(ax.get_axes_locator())
        ax_sel.set_xlim(0, 1)
        ax_sel.set_ylim(0, 1)
        ax_sel.set_navigate(False)
        ax_sel.set_axis_off()

        def _sel_to_data(verts):
            """axes-fraction [0,1] verts on ax_sel -> data coords on ax."""
            disp = ax_sel.transAxes.transform(verts)
            return ax.transData.inverted().transform(disp)

        def on_select(verts):
            data_verts = _sel_to_data(verts)
            inside = MplPath(data_verts).contains_points(np.column_stack([xs, ys]))
            colors = base_colors.copy()
            colors[~inside] = mcolors.to_rgba(self.GREY_OUT)
            pts.set_facecolors(colors)
            fig.canvas.draw_idle()
            # store verts; visualized df reflects the sampled subset, but
            # `apply()` re-runs the gate on the full dataframe
            self.selection.update(mask=inside, verts=data_verts,
                                  df=df.filter(pl.Series(inside)))

        fig._poly_selector = PolygonSelector(
            ax_sel, on_select, useblit=True,
            props=dict(color=self.ACCENT, lw=1.25, zorder=99999),
            handle_props=dict(markeredgecolor=self.ACCENT, markerfacecolor="white",
                              marker="h", alpha=1, zorder=99999),
        )

        # if verts were provided up front, draw them and apply the gate visually
        if preset_verts is not None:
            data_verts = [tuple(v) for v in preset_verts]
            disp = ax.transData.transform(data_verts)
            sel_verts = ax_sel.transAxes.inverted().transform(disp)
            fig._poly_selector.verts = [tuple(v) for v in sel_verts]

            inside = MplPath(data_verts).contains_points(np.column_stack([xs, ys]))
            colors = base_colors.copy()
            colors[~inside] = mcolors.to_rgba(self.GREY_OUT)
            pts.set_facecolors(colors)
            self.selection.update(mask=inside, verts=data_verts,
                                  df=df.filter(pl.Series(inside)))
            fig.canvas.draw_idle()

        plt.show()
        return self

    def apply(self, df: pl.DataFrame = None) -> pl.DataFrame:
        """Apply the selected gate to the *full* dataframe (not just the sampled subset)."""
        return self._filter_verts(df=df)


# ============================================================================
# 1D histogram threshold
# ============================================================================
class ThresholdSelector(_BaseSelector):
    """Histogram with a horizontal SpanSelector for range thresholding (log-aware)."""

    def show(self, metric: str, *, log: bool = False, **kwargs):
        kw = {**self.kwargs, **kwargs}
        self._validate_log(log)
        df = self.df
        self._metric = metric

        raw = self._col_f64(df, metric)
        finite = np.isfinite(raw)
        if log:
            finite &= raw > 0
        data = raw[finite]

        self.fig, self.ax = plt.subplots(figsize=kw.get("figsize", (7, 4)))
        fig, ax = self.fig, self.ax

        # log-spaced bins when requested so bars are visually uniform
        bins_arg = kw.get("bins", "auto")
        if log and isinstance(bins_arg, int):
            bins_arg = np.logspace(np.log10(data.min()), np.log10(data.max()), bins_arg + 1)

        counts, bins, patches = ax.hist(data, bins=bins_arg,
                                        edgecolor="white", linewidth=0.5, zorder=3)
        norm = plt.Normalize(0, counts.max() * kw.get("density_factor", 1.05))
        base_fc = [self.cmap(norm(c)) for c in counts]
        for p, fc in zip(patches, base_fc):
            p.set_facecolor(fc)

        ax.set_xlabel(metric); ax.set_ylabel("Count")
        if log:
            ax.set_xscale("log", nonpositive="clip")
        self._style_ax(ax)

        # linear proxy axis so span width stays uniform in pixels (log-aware)
        ax_sel = ax.twiny()
        ax_sel.set_axes_locator(ax.get_axes_locator())
        ax_sel.set_navigate(False); ax_sel.set_axis_off()
        to_data = (lambda v: np.exp(v)) if log else (lambda v: v)
        to_sel = (lambda v: np.log(v)) if log else (lambda v: v)

        def _sync():
            lo, hi = ax.get_xlim()
            ax_sel.set_xlim(to_sel(lo), to_sel(hi))
        _sync()
        ax.callbacks.connect("xlim_changed", lambda *_: _sync())

        # preserve any predefined min/max passed via `limits`
        preset_min = self.selection.get("min")
        preset_max = self.selection.get("max")
        self.selection = {"min": None, "max": None, "mask": None, "df": None}
        min_lbl = ax.text(0, 0, "", **self._label_kwargs(visible=False))
        max_lbl = ax.text(0, 0, "", **self._label_kwargs(visible=False))

        def onselect(vmin, vmax):
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            xmin, xmax = to_data(vmin), to_data(vmax)

            if xmin == xmax:
                self.selection.update(min=None, max=None, mask=None, df=None)
                for p, fc in zip(patches, base_fc):
                    p.set_facecolor(fc)
                min_lbl.set_visible(False); max_lbl.set_visible(False)
                fig.canvas.draw_idle()
                return

            for p, fc, lo, hi in zip(patches, base_fc, bins[:-1], bins[1:]):
                p.set_facecolor(fc if xmin <= 0.5 * (lo + hi) <= xmax else self.GREY_OUT)

            mask = (raw >= xmin) & (raw <= xmax)
            self.selection.update(min=xmin, max=xmax, mask=mask,
                                  df=df.filter(pl.Series(mask)))

            y_top = ax.get_ylim()[1]
            for lbl, v in [(min_lbl, xmin), (max_lbl, xmax)]:
                lbl.set_position((v, y_top)); lbl.set_text(f"{v:.4g}"); lbl.set_visible(True)

            fig.canvas.draw_idle()

        fig._span_selector = SpanSelector(
            ax_sel, onselect, "horizontal", useblit=True, interactive=True,
            drag_from_anywhere=True,
            props=dict(alpha=0.2, facecolor=self.SPAN_FACE, zorder=2),
            handle_props=dict(lw=1.25, color=self.ACCENT, zorder=99999),
        )

        # if limits were provided up front, draw the span and trigger selection
        if preset_min is not None and preset_max is not None:
            fig._span_selector.extents = (to_sel(preset_min), to_sel(preset_max))
            onselect(to_sel(preset_min), to_sel(preset_max))

        plt.show()
        return self

    def apply(self, df: pl.DataFrame = None) -> pl.DataFrame:
        """Apply the selected min/max range to the *full* dataframe (not just the sampled subset)."""
        return self._filter_minmax(df=df)


# ============================================================================
# 1D jitter strip
# ============================================================================
class JitterSelector(_BaseSelector):
    """Jittered strip plot with vertical SpanSelector (log-aware)."""

    def show(self, metric: str, *, log: bool = False, **kwargs):
        kw = {**self.kwargs, **kwargs}
        self._validate_log(log)
        df = self.df
        self._metric = metric

        raw = self._col_f64(df, metric)
        finite = np.isfinite(raw)
        if log: finite &= raw > 0
        data = raw[finite]
        kde_vals = np.log(data) if log else data

        bw = kw.get("kde_bw", 0.2)
        x_center = 1.0

        density = gaussian_kde(kde_vals, bw_method=bw)(kde_vals)
        dnorm = plt.Normalize(density.min(), density.max() * kw.get("density_factor", 1.05))
        base_colors = self.cmap(dnorm(density))

        self.fig, self.ax = plt.subplots(figsize=kw.get("figsize", (4, 7)))
        fig, ax = self.fig, self.ax

        jitter = (np.random.rand(len(data)) - 0.5) * kw.get("jitter_width", 1)
        pts = ax.scatter(np.full(len(data), x_center) + jitter, data, s=4, marker="o",
                         c=base_colors, alpha=kw.get("scatter_alpha", 1.0),
                         zorder=10, edgecolors="none")

        ax.set_ylabel(metric)
        ax.set_xticks([x_center]); ax.set_xticklabels([metric])
        ax.set_xlim(x_center - 0.55, x_center + 0.55)
        if log: ax.set_yscale("log", nonpositive="clip")
        self._style_ax(ax, grid_axis="y")

        # right marginal KDE
        divider = make_axes_locatable(ax)
        ax_right = divider.append_axes("right", size="15%", pad=0, sharey=ax)
        kde_y = gaussian_kde(kde_vals, bw_method=bw)
        grid = np.linspace(kde_vals.min(), kde_vals.max(), 400)
        y_dens = kde_y(grid)
        y_grid = np.exp(grid) if log else grid
        ax_right.plot(y_dens, y_grid, color="grey", lw=1, zorder=0, alpha=0.25)
        kde_line, = ax_right.plot(y_dens, y_grid, color="grey", lw=1, zorder=2)
        ax_right.set_axis_off()

        # linear proxy axis for uniform dragging in log space
        ax_sel = ax.twinx()
        ax_sel.set_axes_locator(ax.get_axes_locator())
        ax_sel.set_navigate(False); ax_sel.set_axis_off()
        to_data = (lambda v: np.exp(v)) if log else (lambda v: v)
        to_sel = (lambda v: np.log(v)) if log else (lambda v: v)

        def _sync():
            lo, hi = ax.get_ylim()
            ax_sel.set_ylim(to_sel(lo), to_sel(hi))
        _sync()
        ax.callbacks.connect("ylim_changed", lambda *_: _sync())

        # preserve any predefined min/max passed via `limits`
        preset_min = self.selection.get("min")
        preset_max = self.selection.get("max")
        self.selection = {"min": None, "max": None, "mask": None, "df": None}
        min_lbl = ax.text(0, 0, "", **self._label_kwargs(ha="left", va="top", visible=False))
        max_lbl = ax.text(0, 0, "", **self._label_kwargs(ha="left", va="bottom", visible=False))

        def onselect(vmin, vmax):
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            ymin, ymax = to_data(vmin), to_data(vmax)

            if ymin == ymax:
                self.selection.update(min=None, max=None, mask=None, df=None)
                pts.set_facecolors(base_colors)
                kde_line.set_data(y_dens, y_grid)
                min_lbl.set_visible(False); max_lbl.set_visible(False)
                fig.canvas.draw_idle()
                return

            local_mask = (data >= ymin) & (data <= ymax)
            colors = base_colors.copy()
            colors[~local_mask] = mcolors.to_rgba(self.GREY_OUT, alpha=kw.get("scatter_alpha", 1.0))
            pts.set_facecolors(colors)

            # highlight selected KDE segment with interpolated edges
            gm = (y_grid >= ymin) & (y_grid <= ymax)
            edges = kde_y(np.log([ymin, ymax]) if log else np.array([ymin, ymax]))
            if gm.any():
                kde_line.set_data(np.concatenate(([edges[0]], y_dens[gm], [edges[1]])),
                                  np.concatenate(([ymin], y_grid[gm], [ymax])))
            else:
                kde_line.set_data(edges, [ymin, ymax])

            mask = (raw >= ymin) & (raw <= ymax)
            self.selection.update(min=ymin, max=ymax, mask=mask,
                                  df=df.filter(pl.Series(mask)))

            y0, y1 = ax.get_ylim()
            off_min = ymin * 0.02 if log else 0.01 * (y1 - y0)
            off_max = ymax * 0.02 if log else 0.01 * (y1 - y0)
            x_right = ax.get_xlim()[1]
            max_lbl.set_position((x_right, ymax + off_max)); max_lbl.set_text(f"{ymax:.4g}"); max_lbl.set_visible(True)
            min_lbl.set_position((x_right, ymin - off_min)); min_lbl.set_text(f"{ymin:.4g}"); min_lbl.set_visible(True)

            fig.canvas.draw_idle()

        fig._span_selector = SpanSelector(
            ax_sel, onselect, "vertical", useblit=True, interactive=True,
            drag_from_anywhere=True,
            props=dict(alpha=0.2, facecolor=self.SPAN_FACE, zorder=4),
            handle_props=dict(lw=1.25, color=self.ACCENT, zorder=99999),
        )

        # if limits were provided up front, draw the span and trigger selection
        if preset_min is not None and preset_max is not None:
            fig._span_selector.extents = (to_sel(preset_min), to_sel(preset_max))
            onselect(to_sel(preset_min), to_sel(preset_max))

        plt.show()
        return self

    def apply(self, df: pl.DataFrame = None) -> pl.DataFrame:
        """Apply the selected min/max range to the *full* dataframe (not just the sampled subset)."""
        return self._filter_minmax(df=df)


# ============================================================================
# Facade / registry
# ============================================================================
class DataFilter:
    """
    Facade holding a dataframe and a registry of interactive selectors.

    Usage:
        f = DataFilter(df)
        g = f.gate(df, x="Track length", y="Straightness index", log_x=True)
        t = f.threshold(df, "Track points", bins=50)
        j = f.jitter(df, "Speed mean", log=True)
    """

    def __init__(self, df: pl.DataFrame = None, seed: int = 42, **default_kwargs):
        np.random.seed(seed)
        self.df = df if df is not None else pl.DataFrame()
        self.default_kwargs = default_kwargs
        self.selectors: dict[int, _BaseSelector] = {}

    def _register(self, sel: _BaseSelector) -> _BaseSelector:
        self.selectors[sel.id] = sel
        return sel

    # ---- factory methods ---------------------------------------------------
    def gate(self, df: pl.DataFrame, x: str, y: str, *, limits: list = None, **kw) -> GateSelector:
        """
        Create a 2D polygon gate selector for the given x/y columns of `df`.

        limits : list, optional
            Predefined polygon vertices to initialize and draw the gate.
        ```
        [[x1, y1],
         [x2, y2],
         [x3, y3], ...]
        ```
        """
        sel = GateSelector(df, limits, **{**self.default_kwargs, **kw})
        return self._register(sel.show(x, y, **kw))

    def threshold(self, df: pl.DataFrame, metric: str, *, limits: list = None, **kw) -> ThresholdSelector:
        """
        Create a 1D histogram threshold selector for the given metric column of `df`.

        limits : list, optional
            Predefined min/max limits to initialize and draw the threshold: [min, max].
        """
        sel = ThresholdSelector(df, limits, **{**self.default_kwargs, **kw})
        return self._register(sel.show(metric, **kw))

    def jitter(self, df: pl.DataFrame, metric: str, *, limits: list = None, **kw) -> JitterSelector:
        """
        Create a 1D jitter strip selector for the given metric column of `df`.

        limits : list, optional
            Predefined min/max limits to initialize and draw the threshold: [min, max].
        """
        sel = JitterSelector(df, limits, **{**self.default_kwargs, **kw})
        return self._register(sel.show(metric, **kw))


filter = DataFilter()