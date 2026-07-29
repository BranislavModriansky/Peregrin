from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Literal, Any

from ..._pckg_exceptions._pckg_errors import *
from ..._pckg_exceptions._pckg_warnings import *

from ...various import is_empty, get_aliases
from ...compute.stats import Stats
from ..painter import paint
from ...categorizer import categorize

import warnings


class TimeSeries:
    """
    #### *Time-series (frame-wise) analysis and visualization class.*

    Computes a chosen frame-level metric (and optionally its dispersion:
    ``sd`` / ``sem`` / ``min-max`` / ``ci``) **on call** from the input *spot*
    data via :class:`Stats`, aggregating per group as defined by
    ``grouping_level`` over the category hierarchy
    ``['track_uid', 'subsubgroup', 'subgroup', 'group', 'subset', 'set']``.

    A single metric is plotted against ``time_point`` (one line per group), with
    an optional shaded dispersion band, optional scatter markers, and optional
    log axes.
    """

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    ALIASES = {
        'grouping_level': ['grouping', 'groupby', 'group_by', 'grouping_level'],
        'fig_size': ['figsize', 'figure_size', 'fig_size'],
        'color_by': ['color_by', 'colour_by', 'colorby', 'colourby'],
        'color': ['color', 'colour'],
    }

    # Metrics available from Stats.frames whose base output column equals the
    # metric name + '_mean'. The user passes the base metric name; the class
    # resolves the concrete Stats columns.
    DEFAULT_METRIC = 'instantaneous_speed'

    def __init__(self):
        ...

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plot(
        self,
        data: pd.DataFrame,
        metric: str = DEFAULT_METRIC,
        band: Optional[Literal['sd', 'sem', 'minmax', 'ci']] = None,
        categories: Optional[dict[str, list]] = None,
        *,
        grouping_level: Literal['highest', 'lowest'] | str | int = 'highest',
        log: bool = False,
        **kw,
    ) -> plt.Figure:

        self.data = data.copy() if data is not None else pd.DataFrame()
        self.metric = metric
        self.band = band
        self.categories = categories
        self.log = log
        self.line = kw.get('line', True)
        self.scatter = kw.get('scatter', False)

        self.kwargs = get_aliases(kw, self.ALIASES)
        self.grouping_level = self.kwargs.get('grouping_level', grouping_level)

        # Resolve metric-derived column names.
        self.col = f'{self.metric}'
        self.sd_col = f'{self.metric}_sd'
        self.sem_col = f'{self.metric}_sem'

        self._arrange_data()

        # ---- compute frame stats on call ----------------------------- #
        self._compute_series()

        fig, ax = plt.subplots(figsize=self.kwargs.get('fig_size', (10, 7)))
        if is_empty(self.data) or self.mean_col not in self.data.columns:
            return fig

        if self.log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        self._resolve_group_key()
        self._resolve_band()
        color_map = self._build_color_map()

        self._set_axis_labels(ax)

        groups = list(self.data.groupby(self.group_key, sort=False, observed=True))

        for idx, (name, gdata) in enumerate(groups):
            gdata = gdata.sort_values('time_point')

            x_data = gdata['time_point'].to_numpy(dtype=float)
            y_data = gdata[self.mean_col].to_numpy(dtype=float)

            color = self._resolve_color(color_map.get(name), idx)
            label = self._group_label(name)

            # ---- dispersion band ------------------------------------- #
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
                    x_data, y_data, marker='o', markersize=5, label=None,
                    linestyle='none', color=color, zorder=5,
                )

        self._set_ylim(ax, self.data[self.mean_col].to_numpy(dtype=float))
        self._style_axes(ax, fig)

        return fig

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def _arrange_data(self) -> None:
        """Ensure the input data is in a suitable format for computation."""
        if is_empty(self.data):
            self.data = pd.DataFrame()
            return

        if self.categories:
            self.data = categorize(self.data, self.categories)

    # ------------------------------------------------------------------ #
    # Computation
    # ------------------------------------------------------------------ #
    def _compute_series(self) -> None:
        """Compute the frame metric (+ requested dispersion) from spot data."""
        if is_empty(self.data):
            self.data = pd.DataFrame()
            return

        need_descr_err = self.band in ('sd', 'sem', 'min-max')
        need_infer_err = self.band in ('sem', 'ci')
        bootstrap_ci = self.band == 'ci'

        engine = Stats(
            cat_descr=True,
            cat_descr_err=need_descr_err,
            cat_infer_err=need_infer_err,
            bootstrap_ci=bootstrap_ci,
        )

        subset = self._metric_subset(engine)

        # The frame stats operate on spot data first computed via Stats.spots
        # (so that instantaneous / cumulative metrics exist), then aggregated
        # per frame via Stats.frames.
        spots = engine.spots(self.data)

        self.data = engine.frames(
            spots,
            subset=subset,
            grouping_level=self.grouping_level,
        )

    def _metric_subset(self, engine: Stats) -> list[str]:
        """Metric columns to request from Stats.frames."""
        subset = [self.mean_col]
        match self.band:
            case 'sd' | 'min-max':
                subset.append(self.sd_col)
            case 'sem':
                subset += [self.sd_col, self.sem_col]
            case 'ci':
                subset += [
                    f'{self.metric}_{engine.CI_STATISTIC}_ci{engine.CONFIDENCE_LEVEL}_low',
                    f'{self.metric}_{engine.CI_STATISTIC}_ci{engine.CONFIDENCE_LEVEL}_high',
                ]
            case _:
                pass
        return subset

    # ------------------------------------------------------------------ #
    # Grouping / colors
    # ------------------------------------------------------------------ #
    def _resolve_group_key(self) -> None:
        """Determine the column that identifies a plotted group.

        The correct group key is the *finest* hierarchy column that Stats
        actually aggregated over. Stats tags each block with `grouping_level`
        set to the coarsest column of that block (e.g. 'set' for 'highest').
        We therefore key on that tagged column, NOT simply on `present[-1]`,
        which may be a re-attached parent that collapses multiple groups
        onto the same time_point (causing zig-zag artifacts).
        """
        hierarchy = Stats.DEFAULT_CATEGORIES  # track_uid ... set
        present = [c for c in hierarchy if c in self.data.columns]

        # Prefer the column named by the `grouping_level` tag (the actual key
        # Stats grouped by for 'highest'/index). Fall back to coarsest present.
        key = None
        if 'grouping_level' in self.data.columns:
            tagged = self.data['grouping_level'].dropna().unique()
            if len(tagged) == 1 and tagged[0] in self.data.columns:
                key = tagged[0]

        if key is None:
            key = present[-1] if present else 'grouping_level'

        self.group_key = key
        if self.group_key not in self.data.columns:
            self.data['_group'] = 'all'
            self.group_key = '_group'

        # Guard against duplicate (group, time_point) rows that would otherwise
        # produce zig-zag lines. Keep one row per group per time point.
        if 'time_point' in self.data.columns:
            self.data = self.data.drop_duplicates(
                subset=[self.group_key, 'time_point'], keep='first'
            )

    def _group_label(self, name: Any) -> str:
        return str(name)

    def _build_color_map(self) -> dict[Any, Any]:
        """One color per group, via the painter (or a supplied color_by)."""
        keys = list(self.data[self.group_key].dropna().unique())

        color_by = self.kwargs.get('color_by')
        if color_by is not None and color_by in self.data.columns:
            colors = paint(self.data, color_by=color_by, **self._paint_kwargs())
            return dict(zip(self.data[self.group_key], np.asarray(colors)))

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
    # Dispersion band handling
    # ------------------------------------------------------------------ #
    def _resolve_band(self) -> None:
        """Validate that the requested band's columns exist; disable otherwise."""
        cols = self.data.columns
        engine = Stats()

        match self.band:
            case 'sd' | 'min-max':
                self._band_ok = self.sd_col in cols
            case 'sem':
                self._band_ok = self.sem_col in cols
            case 'ci':
                low = f'{self.metric}_{engine.CI_STATISTIC}_ci{engine.CONFIDENCE_LEVEL}_low'
                high = f'{self.metric}_{engine.CI_STATISTIC}_ci{engine.CONFIDENCE_LEVEL}_high'
                self._ci_low, self._ci_high = low, high
                self._band_ok = low in cols and high in cols
            case _:
                self._band_ok = False

        if self.band and not self._band_ok:
            warnings.warn(
                f"Requested dispersion band '{self.band}' is unavailable in the "
                f"computed data for metric '{self.metric}'. Ignoring band.",
                category=PlottingWarning, stacklevel=2,
            )

    def _band_bounds(self, gdata: pd.DataFrame, y_data: np.ndarray):
        """Return (bottom, top) arrays for the dispersion band, or (None, None)."""
        if not getattr(self, '_band_ok', False):
            return None, None

        match self.band:
            case 'sd':
                err = gdata[self.sd_col].to_numpy(dtype=float) / 2.0
                return y_data - err, y_data + err
            case 'min-max':
                err = gdata[self.sd_col].to_numpy(dtype=float)
                return y_data - err, y_data + err
            case 'sem':
                err = gdata[self.sem_col].to_numpy(dtype=float)
                return y_data - err, y_data + err
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
        unit = Stats().stat_units(self.metric) or ''
        ylabel = self._pretty_metric(self.metric)
        if unit:
            ylabel = f'{ylabel} [{unit}]'
        ax.set_xlabel(f'Time [{Stats.t_unit}]', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

    @staticmethod
    def _pretty_metric(metric: str) -> str:
        return metric.replace('_', ' ').strip().capitalize()

    def _set_ylim(self, ax: plt.Axes, y_vals: np.ndarray) -> None:
        if self.log:
            return
        finite = y_vals[np.isfinite(y_vals)]
        if finite.size == 0:
            return
        miny, maxy = float(np.min(finite)), float(np.max(finite))
        lower = miny - 0.05 * abs(miny) if miny != 0 else 0.0
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
    # Color helpers
    # ------------------------------------------------------------------ #
    def _resolve_color(self, color: Optional[Any], idx: int = 0) -> Any:
        if color is not None and mcolors.is_color_like(color):
            return color
        return f"C{idx % 10}"


time_series = TimeSeries().plot