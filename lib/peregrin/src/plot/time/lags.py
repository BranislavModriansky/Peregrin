from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Literal, Any

from ..._pckg_exceptions._pckg_errors import *
from ..._pckg_exceptions._pckg_warnings import *

from ...various import is_empty, get_aliases
from ...compute.stats import Stats, stats
from ..painter import paint
from ...categorizer import categorize


class MSD:
    """
    #### *Mean Squared Displacement analysis and visualization class.*

    Computes MSD (and optionally its dispersion: sd / sem / min-max / ci) on call
    from the input *spot* data via :class:`Stats`, aggregating per group as defined
    by ``grouping_level`` over the category hierarchy
    ``['track_uid', 'subsubgroup', 'subgroup', 'group', 'subset', 'set']``.
    """

    # Constants for color adjustments in linear fits
    SATURATION_SCALE = 0.7
    SATURATION_MIN = 0.02
    SATURATION_MAX = 1.0
    BRIGHTNESS_SCALE = 0.8
    BRIGHTNESS_MIN = 0.06

    ALIASES = {
        'grouping_level': ['grouping', 'groupby', 'group_by', 'grouping_level'],
        'fig_size': ['figsize', 'figure_size', 'fig_size'],
        'color_by': ['color_by', 'colour_by', 'colorby', 'colourby'],
        'color': ['color', 'colour'],
    }

    # Columns produced by Stats.time_intervals for MSD.
    MSD_COL = 'MSD'
    MSD_SD_COL = 'MSD_sd'
    MSD_SEM_COL = 'MSD_sem'

    def __init__(self):
        ...

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plot(
        self,
        data: pd.DataFrame,
        band: Optional[Literal['sd', 'sem', 'min-max', 'ci']] = None,
        categories: Optional[dict[str, list]] = None,
        *,
        grouping_level: Literal['highest', 'lowest'] | str | int = 'highest',
        log: bool = False,
        linear_fit: bool = False,
        **kw,
    ) -> plt.Figure:

        self.data = data.copy() if data is not None else pd.DataFrame()
        self.band = band
        self.categories = categories
        self.log = log
        self.linear_fit = linear_fit
        self.line = kw.get('line', True)
        self.scatter = kw.get('scatter', False)

        self.kwargs = get_aliases(kw, self.ALIASES)
        self.grouping_level = self.kwargs.get('grouping_level', grouping_level)

        self._arrange_data()

        # ---- compute MSD on call ------------------------------------- #
        self._compute_msd()

        # If nothing could be computed, return an empty figure.
        fig, ax = plt.subplots(figsize=self.kwargs.get('fig_size', (10, 7)))
        if is_empty(self.data):
            return fig

        if self.log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        self._resolve_group_key()
        self._resolve_band()
        color_map = self._build_color_map()

        self._set_axis_labels(ax)

        groups = list(self.data.groupby(self.group_key, sort=False, observed=True))
        n_groups = len(groups)

        for idx, (name, gdata) in enumerate(groups):
            gdata = gdata.sort_values('time_lag')
            label = self._group_label(name)

            x_data = gdata['time_lag'].to_numpy(dtype=float)
            y_data = gdata[self.MSD_COL].to_numpy(dtype=float)

            color = self._resolve_color(color_map.get(name), idx)

            # ---- error band ------------------------------------------ #
            band_bottom, band_top = self._band_bounds(gdata, y_data)
            if band_bottom is not None:
                mask = np.isfinite(band_bottom) & np.isfinite(band_top)
                if np.any(mask):
                    ax.fill_between(
                        x_data[mask], band_bottom[mask], band_top[mask],
                        color=color, alpha=0.10, linewidth=0, zorder=2,
                    )

            # ---- main line ------------------------------------------- #
            if self.line:
                ax.plot(
                    x_data, y_data, marker='none', label=label,
                    linestyle='-', color=color, alpha=1.0, zorder=6,
                )

            # ---- scatter markers ------------------------------------- #
            if self.scatter:
                ax.plot(
                    x_data, y_data, marker='o', markersize=6, label=None,
                    linestyle='none', color=color, zorder=5,
                )

            # ---- linear fit ------------------------------------------ #
            if self.linear_fit:
                self._add_linear_fit(ax, x_data, y_data, color, idx, n_groups)

        self._set_ylim(ax, self.data[self.MSD_COL].to_numpy(dtype=float))
        self._style_axes(ax, fig)

        return fig


    def _arrange_data(self) -> None:
        """Ensure the input data is in a suitable format for MSD computation."""
        if is_empty(self.data):
            self.data = pd.DataFrame()
            return

        # Categorize the data if categories are provided.
        if self.categories:
            self.data = categorize(self.data, self.categories)

    # ------------------------------------------------------------------ #
    # Computation
    # ------------------------------------------------------------------ #
    def _compute_msd(self) -> None:
        """Compute MSD (+ requested dispersion) from spot data via Stats."""
        if is_empty(self.data):
            self.data = pd.DataFrame()
            return

        # Decide which error statistics Stats must produce.
        need_descr_err = self.band in ('sd', 'sem', 'min-max')
        need_infer_err = self.band in ('sem', 'ci')
        bootstrap_ci = self.band == 'ci'

        engine = Stats(
            cat_descr=True,
            cat_descr_err=need_descr_err,
            cat_infer_err=need_infer_err,
            bootstrap_ci=bootstrap_ci,
        )

        # Request only the MSD columns we actually need.
        subset = self._msd_subset(engine)

        self.data = engine.time_intervals(
            self.data,
            subset=subset,
            grouping_level=self.grouping_level,
        )

    def _msd_subset(self, engine: Stats) -> list[str]:
        """Metric columns to request from Stats.time_intervals for MSD."""
        subset = [self.MSD_COL]
        match self.band:
            case 'sd':
                subset.append(self.MSD_SD_COL)
            case 'sem':
                subset += [self.MSD_SD_COL, self.MSD_SEM_COL]
            case 'ci':
                subset += [
                    f'MSD_{engine.CI_STATISTIC}_ci{engine.CONFIDENCE_LEVEL}_low',
                    f'MSD_{engine.CI_STATISTIC}_ci{engine.CONFIDENCE_LEVEL}_high',
                ]
            case 'min-max':
                # Stats does not emit MSD min/max; derived from the band bounds
                # of the mean ± sd as a fallback (see _band_bounds).
                subset.append(self.MSD_SD_COL)
            case _:
                pass
        return subset

    # ------------------------------------------------------------------ #
    # Grouping / colors
    # ------------------------------------------------------------------ #
    def _resolve_group_key(self) -> None:
        """Determine the column(s) that identify a plotted group."""
        hierarchy = Stats.DEFAULT_CATEGORIES  # track_uid ... set
        present = [c for c in hierarchy if c in self.data.columns]

        # Prefer the coarsest present category as the plotted group key.
        # `grouping_level` already constrained what Stats produced; here we
        # just pick the label column to iterate over.
        self.group_key = present[-1] if present else 'grouping_level'
        if self.group_key not in self.data.columns:
            # Fall back to a single implicit group.
            self.data['_group'] = 'all'
            self.group_key = '_group'

    def _group_label(self, name: Any) -> str:
        return str(name)

    def _build_color_map(self) -> dict[Any, Any]:
        """One color per group, via the painter (or a supplied color_by)."""
        keys = list(self.data[self.group_key].dropna().unique())

        color_by = self.kwargs.get('color_by')
        if color_by is not None and color_by in self.data.columns:
            colors = paint(self.data, color_by=color_by, **self._paint_kwargs())
            return dict(zip(self.data[self.group_key], np.asarray(colors)))

        # Ask the painter for one color per group.
        color = self.kwargs.get('color', 'random')
        try:
            per_group = paint(
                pd.DataFrame(index=np.arange(len(keys))),
                color=color if color in ('random', 'random greys') else 'random',
            )
            per_group = np.asarray(per_group)
            return {k: per_group[i] for i, k in enumerate(keys)}
        except Exception:
            return {k: f"C{i % 10}" for i, k in enumerate(keys)}

    def _paint_kwargs(self) -> dict:
        allowed = ('palette', 'cmap', 'lut_vmin', 'lut_vmax')
        return {k: v for k, v in self.kwargs.items() if k in allowed}

    # ------------------------------------------------------------------ #
    # Error band handling
    # ------------------------------------------------------------------ #
    def _resolve_band(self) -> None:
        """Validate that the requested band's columns exist; disable otherwise."""
        cols = self.data.columns
        engine = Stats()

        match self.band:
            case 'sd':
                self._band_ok = self.MSD_SD_COL in cols
            case 'sem':
                self._band_ok = self.MSD_SEM_COL in cols
            case 'min-max':
                # Derived from mean ± sd fallback.
                self._band_ok = self.MSD_SD_COL in cols
            case 'ci':
                low = f'MSD_{engine.CI_STATISTIC}_ci{engine.CONFIDENCE_LEVEL}_low'
                high = f'MSD_{engine.CI_STATISTIC}_ci{engine.CONFIDENCE_LEVEL}_high'
                self._ci_low, self._ci_high = low, high
                self._band_ok = low in cols and high in cols
            case _:
                self._band_ok = False

        if self.band and not self._band_ok:
            warnings.warn(
                f"Requested error band '{self.band}' is unavailable in the "
                "computed MSD data. Ignoring error band.",
                category=PlottingWarning, stacklevel=2,
            )

    def _band_bounds(self, gdata: pd.DataFrame, y_data: np.ndarray):
        """Return (bottom, top) arrays for the error band, or (None, None)."""
        if not getattr(self, '_band_ok', False):
            return None, None

        match self.band:
            case 'sd':
                err = gdata[self.MSD_SD_COL].to_numpy(dtype=float) / 2.0
                return np.maximum(y_data - err, 0.0), y_data + err
            case 'sem':
                err = gdata[self.MSD_SEM_COL].to_numpy(dtype=float)
                return np.maximum(y_data - err, 0.0), y_data + err
            case 'min-max':
                err = gdata[self.MSD_SD_COL].to_numpy(dtype=float)
                return np.maximum(y_data - err, 0.0), y_data + err
            case 'ci':
                low = gdata[self._ci_low].to_numpy(dtype=float)
                high = gdata[self._ci_high].to_numpy(dtype=float)
                return low, high
            case _:
                return None, None

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #
    def _set_axis_labels(self, ax: plt.Axes) -> None:
        ax.set_xlabel(f'Time lag [{Stats.t_unit}]', fontsize=12)
        ax.set_ylabel('MSD [µm²]', fontsize=12)

    def _set_ylim(self, ax: plt.Axes, y_vals: np.ndarray) -> None:
        if self.log:
            return
        finite = y_vals[np.isfinite(y_vals)]
        if finite.size == 0:
            return
        miny, maxy = float(np.min(finite)), float(np.max(finite))
        lower = miny - 0.5 * abs(miny)
        upper = maxy + 0.05 * abs(maxy) if maxy != 0 else 1.0
        ax.set_ylim(lower, upper)

    def _style_axes(self, ax: plt.Axes, fig: plt.Figure) -> None:
        if self.kwargs.get('title'):
            ax.set_title(
                self.kwargs['title'],
                color=self.kwargs.get('text_color', 'black'),
                fontsize=self.kwargs.get('title_fontsize', 14),
                fontweight=self.kwargs.get('title_fontweight', 'bold'),
            )

        if self.kwargs.get('grid', False):
            ax.grid(True, color='whitesmoke', zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False)

        fig.set_facecolor(self.kwargs.get('fig_background', 'white'))

    # ------------------------------------------------------------------ #
    # Colors helpers
    # ------------------------------------------------------------------ #
    def _resolve_color(self, color: Optional[Any], idx: int = 0) -> Any:
        if color is not None and mcolors.is_color_like(color):
            return color
        return f"C{idx % 10}"

    def _compute_fit_color(self, base_color: Any) -> str:
        safe_color = self._resolve_color(base_color, 0)
        base_rgb = mcolors.to_rgb(safe_color)
        hsv = mcolors.rgb_to_hsv(np.array(base_rgb))
        hsv[1] = np.clip(hsv[1] * self.SATURATION_SCALE, self.SATURATION_MIN, self.SATURATION_MAX)
        hsv[2] = np.clip(hsv[2] * self.BRIGHTNESS_SCALE, self.BRIGHTNESS_MIN, hsv[2])
        return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))

    # ------------------------------------------------------------------ #
    # Linear fit
    # ------------------------------------------------------------------ #
    def _add_linear_fit(self, ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                        color: Any, idx: int, n_tags: int) -> None:

        valid = (
            np.isfinite(x_data) & np.isfinite(y_data)
            & (x_data > 0) & (y_data > 0)
        )
        xv = x_data[valid]
        yv = y_data[valid]
        if xv.size < 2:
            return

        if self.log:
            lxv, lyv = np.log10(xv), np.log10(yv)
            a, b = np.polyfit(lxv, lyv, 1)
            x_fit = np.logspace(lxv.min(), lxv.max(), 200)
            y_fit = (10.0 ** b) * (x_fit ** a)
        else:
            a, b = np.polyfit(xv, yv, 1)
            x_fit = np.linspace(xv.min(), xv.max(), 200)
            y_fit = a * x_fit + b

        fit_color = self._compute_fit_color(color)
        ax.plot(
            x_fit, y_fit, linestyle='-.', color=fit_color,
            linewidth=2, zorder=7, alpha=0.8,
        )

        try:
            if self.log:
                lxv, lyv = np.log10(xv), np.log10(yv)
                lxrange = lxv.max() - lxv.min()
                lyrange = lyv.max() - lyv.min()
                lxrange = lxrange if np.isfinite(lxrange) and lxrange > 0 else 1.0
                lyrange = lyrange if np.isfinite(lyrange) and lyrange > 0 else 1.0
                x_text = 10.0 ** (lxv.max() - 0.03 * lxrange)
                y_base_log = b + a * lxv.max()
                v_offset = (idx - (n_tags - 1) / 2.0) * 0.03 * lyrange
                y_text = 10.0 ** (y_base_log + v_offset)
            else:
                x_text = xv.max() - 0.03 * (xv.max() - xv.min())
                y_base = a * xv.max() + b
                v_offset = (idx - (n_tags - 1) / 2.0) * 0.03 * (yv.max() - yv.min())
                y_text = y_base + v_offset

            slope_text = f"D = {a:.2f} [µm²·{Stats.t_unit}⁻¹]"
            ax.text(
                x_text, y_text, slope_text, color=color,
                fontsize=7, fontweight='bold',
                verticalalignment='center', horizontalalignment='left',
                bbox=dict(facecolor='none', alpha=1, edgecolor='none'),
                zorder=7,
            )
        except Exception:
            pass


def TurnAnglesHeatmap(
    data: pd.DataFrame,
    *,
    grouping_level: Literal['highest', 'lowest'] | str | int = 'highest',
    angle_range: int = 15,
    tlag_range: int = 1,
    cmap: str = "plasma",
    **kwargs,
) -> Optional[plt.Figure]:
    """Plot mean directional change (turning angle) over time lags as a colormesh.

    Directional-change statistics are computed on call via :class:`Stats`.
    """
    text_color = kwargs.get('text_color', 'black')
    title = kwargs.get('title', '')

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (6, 6)))

    if is_empty(data):
        warnings.warn("No data available for plotting.",
                      category=PlottingWarning, stacklevel=2)
        return None

    engine = Stats(cat_descr=True, cat_descr_err=True, cat_infer_err=False)
    data = engine.time_intervals(
        data,
        subset=['directional_change_mean'],
        grouping_level=grouping_level,
    )

    if is_empty(data) or 'directional_change_mean' not in data.columns:
        return None

    lags = np.sort(data['time_lag'].unique())
    if lags.size < 2:
        return None

    tlag_range = lags[1] - lags[0]

    # One "sample" per group per lag.
    hierarchy = Stats.DEFAULT_CATEGORIES
    group_key = next((c for c in reversed(hierarchy) if c in data.columns), None)
    n = data[group_key].nunique() if group_key else 1

    xvals = data['directional_change_mean'].to_numpy(dtype=float)
    yvals = data['time_lag'].to_numpy(dtype=float)

    x_bins = np.arange(0, 181, angle_range)
    y_bins = np.arange(0, lags.max() + tlag_range, tlag_range)

    H, xe, ye = np.histogram2d(xvals, yvals, bins=[x_bins, y_bins])

    pcm = ax.pcolormesh(
        xe, ye, H.T / max(n, 1),
        cmap=cmap, shading='auto',
        norm=mcolors.Normalize(vmin=0, vmax=np.nanmax(H / max(n, 1)) or 1.0),
    )

    ax.set_xlabel("Mean directional change [°]", color=text_color)
    ax.set_ylabel(f"Time lag [{Stats.t_unit}]", color=text_color)
    ax.tick_params(colors=text_color, width=0.5)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(text_color)
        spine.set_linewidth(0.5)

    ax.set_title(title, color=text_color)

    if kwargs.get('strip_background', True):
        fig.set_facecolor('none')

    cbar = plt.colorbar(pcm, ax=ax, aspect=25, pad=0.04)
    cbar.set_label('Fraction of groups', color=text_color)
    cbar.ax.tick_params(colors=text_color, width=0.5)
    for spine in cbar.ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(0.5)

    return fig


import warnings  # placed at bottom to avoid clutter above; standard-lib import

msd = MSD().plot