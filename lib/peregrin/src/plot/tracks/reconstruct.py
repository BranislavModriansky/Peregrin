from inspect import Arguments

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation
from PIL import Image

import warnings
from io import BytesIO

from ...settings import params
from ..._pckg_exceptions._pckg_errors import *
from ..._pckg_exceptions._pckg_warnings import *

from ..categorizer import categorize
from ..painter import (
    retrieve_palette, retrieve_cmap, random_color, random_grey,
    cmap_lut, dyes, is_color_code,
)
from ...various import Values, get_aliases, is_empty, clock
from ...compute.stats import Stats


class TracksResult:
    """
    Container returned by :func:`reconstruct`.

    Holds the static figure plus all the cached plotting arrays needed to
    (re)build the tracks, so that :meth:`animate` can grow the trajectories
    over time without recomputing anything.
    """

    def __init__(self, builder, figure):
        # Keep a reference to the builder so we reuse its cached geometry/colors.
        self._builder = builder
        self.figure = figure
        self.polar = builder.common_start

    # Convenience passthroughs -------------------------------------------------
    @property
    def fig(self) -> plt.Figure:
        return self.figure

    @property
    def segments(self) -> np.ndarray:
        return self._builder.segments

    def show(self):
        plt.show()

    def save(self, path, **kwargs):
        self.figure.savefig(path, **kwargs)

    # Animation ---------------------------------------------------------------
    def animate(
        self,
        *,
        loop: bool = True,
        speed: float = 1.0,
        interval: float = 40.0,
        frames: int = None,
        blit: bool = False,
        show_heads: bool = True,
        repeat_delay: float = 0.0,
        on_new_figure: bool = True,
    ) -> FuncAnimation:
        """
        Vectorized growing-trajectory animation.

        Segments are precomputed once. Each frame only reveals more of the
        already-computed segments by mutating a single LineCollection — no image
        encoding, so it stays a live vector animation.

        Parameters
        ----------
        on_new_figure : bool
            Render onto a fresh figure so the static reconstruction figure is
            left untouched (and re-running does not stack tracks).
        """
        b = self._builder

        # --- fresh, cleansed axes so nothing stacks on re-run --------------
        if on_new_figure:
            fig, ax = b._new_axes(polar=self.polar)
        else:
            fig = self.figure
            ax = fig.axes[0]
            # Cleanse any previously drawn animation artists.
            for coll in list(ax.collections):
                coll.remove()

        plot_x, plot_y = b._plot_coords(polar=self.polar)
        starts = b._track_starts
        run_lengths = b._run_lengths

        if run_lengths.size == 0:
            return FuncAnimation(fig, lambda _f: (), frames=1, blit=False)

        max_len = int(run_lengths.max())
        n_frames = frames if frames is not None else max(2, int(max_len / max(speed, 1e-6)))
        reveal_grid = np.linspace(1, max_len, n_frames).astype(np.intp)

        # --- fully vectorized reveal schedule ------------------------------
        # For each within-track segment, the reveal index at which it appears
        # = its within-track position + 1. Compare against the frame's reveal.
        seg_lens = b._within_track_seg_lengths                 # (n_tracks,)
        # local index (0..L-2) of every within-track segment, concatenated
        seg_local_idx = np.concatenate(
            [np.arange(c) for c in seg_lens]) if seg_lens.sum() else np.empty(0, np.intp)
        seg_appear = seg_local_idx + 1                         # reveal needed

        # Precompute colors ONCE for the full segment set (RGBA), so per-frame
        # we only pass a boolean-cropped view — no recompute, no color glitches.
        full_colors = b._segment_colors_for_mask(np.ones(seg_appear.size, dtype=bool))

        base_lc = LineCollection(
            np.empty((0, 2, 2)),
            linewidths=b.kwargs.get('lw', 1),
            zorder=11,
        )
        ax.add_collection(base_lc)

        head_scatter = None
        if show_heads:
            head_scatter = ax.scatter(
                [], [],
                marker=b.kwargs.get('head_shape', 'o'),
                s=b.kwargs.get('head_size', 10),
                zorder=13,
            )

        segments_all = b.segments

        def _frame(reveal):
            mask = seg_appear <= reveal                        # vectorized crop
            base_lc.set_segments(segments_all[mask])
            if not isinstance(full_colors, str):
                base_lc.set_color(np.asarray(full_colors)[mask])
            else:
                base_lc.set_color(full_colors)

            artists = [base_lc]
            if head_scatter is not None:
                visible = np.minimum(reveal, run_lengths)
                has_pt = visible > 0
                head_idx = starts + np.maximum(visible - 1, 0)
                if has_pt.any():
                    head_scatter.set_offsets(
                        np.column_stack((plot_x[head_idx][has_pt],
                                         plot_y[head_idx][has_pt])))
                else:
                    head_scatter.set_offsets(np.empty((0, 2)))
                artists.append(head_scatter)
            return artists

        def _init():
            base_lc.set_segments(np.empty((0, 2, 2)))
            arts = [base_lc]
            if head_scatter is not None:
                head_scatter.set_offsets(np.empty((0, 2)))
                arts.append(head_scatter)
            return arts

        anim = FuncAnimation(
            fig,
            lambda i: _frame(int(reveal_grid[i])),
            init_func=_init,
            frames=n_frames,
            interval=interval,
            blit=blit,
            repeat=loop,
            repeat_delay=repeat_delay,
        )
        self._anim = anim
        return anim


class ReconstructTracks:

    ALIASES = {
        "smoothing_index": ["smoothing", "smoothing_index", "smooth_window_size"],
        "seed": ["seed", "random_seed", "rng_seed"],
        "c_mode": ["color_mode", "colour_mode", "c_mode"],
        "cmap": ["cmap", "colormap", "colourmap"],
        "palette": ["palette", "color_palette", "colour_palette"],
        "color_by": ["color_by", "colour_by"],
        "lw": ["lw", "linewidth", "line_width"],
        "marker_fill": ["marker_fill", "head_fill", "fill"],
        "color": ["color", "colour", "c"],
    }

    _C_MODES = ('None', 'random', 'random_greys', 'categorical', 'numeric')
    _DYE_COLOR_SET = frozenset(dyes.colors.values())

    KEY_COLS = ['condition', 'replicate', 'track_id']
    REQUIRED_COLS = ['condition', 'replicate', 'track_id', 'track_uid',
                     'time_point', 'x_coordinate', 'y_coordinate']

    # Background -> face color, cartesian grid, polar grid lookups (built once).
    _BG_FACE = {
        'white': 'white', 'light': 'lightgrey', 'mid': 'darkgrey',
        'dark': 'dimgrey', 'black': 'black',
    }
    _BG_GRID_CART = {
        'white': ('gainsboro', 0.5), 'light': ('silver', 0.5),
        'mid': ('silver', 0.5), 'dark': ('grey', 0.5), 'black': ('dimgrey', 0.5),
    }
    _BG_GRID_POLAR = {
        'white': ('lightgrey', 0.7, 0.8), 'light': ('darkgrey', 0.7, 0.6),
        'mid': ('dimgrey', 0.7, 0.5), 'dark': ('grey', 0.7, 0.6),
        'black': ('dimgrey', 0.5, 0.4),
    }

    def __init__(self):
        # Cache the last input so we can skip re-sorting / re-arranging identical data.
        self._cache_key = None

    @clock
    def reconstruct(
        self,
        spot_data: pd.DataFrame,
        track_data: pd.DataFrame = None,
        *,
        conditions: list = None,
        replicates: list = None,
        common_start: bool = False,
        ignore_categories: bool = False,
        **kwargs
    ) -> TracksResult:

        self.spot_data = spot_data
        self.track_data = track_data
        self.conditions = conditions or []
        self.replicates = replicates or []
        self.common_start = common_start
        self.kwargs = get_aliases(kwargs, self.ALIASES)

        ignore = params.ignore_categories or ignore_categories
        if ignore:
            self.REQUIRED_COLS = ['track_uid', 'time_point',
                                  'x_coordinate', 'y_coordinate']
        if ignore_categories and not params.ignore_categories:
            warnings.warn(
                "ignore_categories is set to True for this function call, but the global "
                "setting is False. This function call will not ignore categories. To change "
                "the global setting, use `peregrin.settings(ignore_categories=True)`.",
                category=UserWarning, stacklevel=2)

        # Skip the whole sort/index/cache pipeline if the identical DataFrame +
        # relevant filters were already processed. id() + shape is a cheap,
        # safe fingerprint for the reactive-app "same data, new color" case.
        smoothing = self.kwargs.get('smoothing_index')
        cache_key = (id(spot_data), spot_data.shape, tuple(self.conditions),
                     tuple(self.replicates), ignore, smoothing)
        if cache_key != self._cache_key:
            self._arrange_data(ignore)
            if smoothing is not None and smoothing > 0:
                self._smooth(smoothing)
                self._cache_arrays(ignore)  # smoothing mutated coords
            self._cache_key = cache_key

        # Colors are cheap and often the only thing that changes; recompute every call.
        self._assign_color()

        figure = self.polar() if common_start else self.cartesian()
        return TracksResult(self, figure)

    # ------------------------------------------------------------------ #
    # Figure builders
    # ------------------------------------------------------------------ #

    @clock
    def cartesian(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(13, 10))
        self._build_tracks(ax)

        x = self._x
        if x.size:
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(self._y.min(), self._y.max())

        ax.set_aspect('equal', adjustable='box')
        text_color = self.kwargs.get('text_color', 'black')
        ax.set_xlabel('x_coordinate [µm]', color=text_color)
        ax.set_ylabel('y_coordinate [µm]', color=text_color)
        ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=text_color)

        self._background_color()
        ax.set_facecolor(self.face_color)
        self._apply_cartesian_ticks(ax, text_color)
        ax.grid(False)

        if self.kwargs.get('show_heads', True):
            self._head_markers(ax, polar=False)
        if self.kwargs.get('hide_backdrop', False):
            fig.set_facecolor('none')

        return fig

    def polar(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12.5, 9.5),
                               subplot_kw={'projection': 'polar'})
        self._get_radius()

        text_color = self.kwargs.get('text_color', 'black')
        ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=text_color)
        ax.set_ylim(0, self.y_max_global)
        ax.spines['polar'].set_visible(False)

        self._build_tracks(ax, polar=True)

        self._background_color()
        ax.set_facecolor(self.face_color)
        ax.grid(False)

        self._annotate_r_axis(ax)
        self._annotate_theta_axis(ax)

        if self.kwargs.get('show_heads', True):
            self._head_markers(ax, polar=True)
        if self.kwargs.get('hide_backdrop', False):
            fig.set_facecolor('none')

        return fig

    @staticmethod
    def _apply_cartesian_ticks(ax, text_color):
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.tick_params(axis='both', which='major', labelsize=8, colors=text_color)

    # ------------------------------------------------------------------ #
    # Data arrangement / caching
    # ------------------------------------------------------------------ #

    @clock
    def _arrange_data(self, ignore: bool = False):
        # When categories are ignored, categorize() is pure overhead: skip it.
        if not ignore:
            self.spot_data = categorize(
                self.spot_data, self.conditions, self.replicates, **self.kwargs)

        sort_cols = (['track_uid', 'time_point'] if ignore
                     else ['condition', 'replicate', 'track_id', 'time_point'])

        self.spot_data = (
            self.spot_data
            .reset_index(drop=True)
            .sort_values(by=['track_uid', 'time_point'], kind='mergesort')
        )

        if not is_empty(self.track_data):
            if not ignore:
                self.track_data = categorize(
                    self.track_data, self.conditions, self.replicates, **self.kwargs)
            self.track_data = self.track_data.sort_values(sort_cols[:-1], kind='stable')

        # Only set the MultiIndex when we actually need category-based grouping
        # (smoothing path). For ignore mode we work purely on track_uid via
        # cached numpy arrays, so skip the expensive set_index.
        if not ignore:
            if list(self.spot_data.index.names) != self.KEY_COLS:
                self.spot_data = self.spot_data.set_index(self.KEY_COLS)
            if (not is_empty(self.track_data)
                    and list(self.track_data.index.names) != self.KEY_COLS):
                self.track_data = self.track_data.set_index(self.KEY_COLS)

        self._cache_arrays(ignore)

    def _cache_arrays(self, ignore: bool = False):
        """Materialize plotting arrays and track run-boundaries once, after sorting."""
        self._x = self.spot_data['x_coordinate'].to_numpy(dtype=float, copy=False)
        self._y = self.spot_data['y_coordinate'].to_numpy(dtype=float, copy=False)

        n = self._x.size
        if n == 0:
            self._same_track = np.empty(0, dtype=bool)
            self._track_starts = np.empty(0, dtype=np.intp)
            self._track_ends = np.empty(0, dtype=np.intp)
            self._run_lengths = np.empty(0, dtype=np.intp)
            self._track_uids = np.empty(0, dtype=object)
            self._within_track_seg_lengths = np.empty(0, dtype=np.intp)
            return

        uids = self.spot_data['track_uid'].to_numpy()
        if n > 1:
            self._same_track = uids[1:] == uids[:-1]
            self._track_starts = np.concatenate(
                ([0], np.flatnonzero(~self._same_track) + 1))
        else:
            self._same_track = np.empty(0, dtype=bool)
            self._track_starts = np.array([0], dtype=np.intp)

        self._track_ends = np.append(self._track_starts[1:] - 1, n - 1)
        # Run length of every track, cached once (reused by polar / colors / radius).
        self._run_lengths = np.diff(np.append(self._track_starts, n))
        # One uid per track (at each start) for fast color mapping.
        self._track_uids = uids[self._track_starts]
        # Number of within-track segments per track (L - 1), for animation reveal.
        self._within_track_seg_lengths = np.maximum(self._run_lengths - 1, 0)

    @clock
    def _get_radius(self):
        """Global maximum radius, computed from cached numpy arrays (no groupby)."""
        if self._x.size == 0:
            self.y_max_global = 100.0
            self.y_max_label_global = "100 μm"
            return

        starts = self._track_starts
        x0 = np.repeat(self._x[starts], self._run_lengths)
        y0 = np.repeat(self._y[starts], self._run_lengths)
        r_max = np.hypot(self._x - x0, self._y - y0).max()

        self.y_max_global = r_max + 10.0
        self.y_max_label_global = f"{round(r_max) + 10} μm"

    @clock
    def _smooth(self, window: int):
        if not (isinstance(window, int) and window >= 1):
            warnings.warn(
                f"Invalid 'smoothing_index': {window} -> Must be a positive integer "
                "-> No smoothing applied.",
                category=UserWarning, stacklevel=2)
            return

        # Vectorized per-track smoothing directly on cached arrays — no groupby,
        # no per-track Python lambda. Endpoints are preserved.
        self._x = self._smooth_runs(self._x, window)
        self._y = self._smooth_runs(self._y, window)
        # Write back so downstream (DataFrame-based) consumers stay consistent.
        self.spot_data['x_coordinate'] = self._x
        self.spot_data['y_coordinate'] = self._y

    def _smooth_runs(self, values: np.ndarray, window: int) -> np.ndarray:
        """Rolling-mean smooth each track run, then linearly pin its endpoints."""
        out = values.copy()
        starts, ends = self._track_starts, self._track_ends
        for s, e in zip(starts, ends):
            seg = values[s:e + 1]
            m = seg.size
            if m < 2:
                continue
            # Rolling mean with min_periods=1 via cumulative sum.
            csum = np.concatenate(([0.0], np.cumsum(seg)))
            idx = np.arange(m)
            lo = np.maximum(0, idx - window + 1)
            smoothed = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
            t = np.linspace(0.0, 1.0, m)
            correction = ((seg[0] - smoothed[0]) * (1 - t)
                          + (seg[-1] - smoothed[-1]) * t)
            out[s:e + 1] = smoothed + correction
        return out

    # ------------------------------------------------------------------ #
    # Color assignment
    # ------------------------------------------------------------------ #

    @clock
    def _assign_color(self):
        color_by = self.kwargs.get('color_by')

        if color_by is not None:
            if self.kwargs.get('color') is not None:
                warnings.warn(
                    "Both 'color' and 'color_by' parameters are provided -> Parameter "
                    "'color' will be ignored -> Using 'color_by' for color assignment.",
                    category=ConflictingParametersWarning, stacklevel=2)

            datatype = None
            if isinstance(color_by, tuple) and len(color_by) == 2:
                color_by, datatype = color_by
                if datatype not in ('categorical', 'numeric'):
                    raise ValueError(
                        f"Invalid datatype parameter '{datatype}' for color_by. "
                        "Must be one of ['categorical', 'numeric'].")

            col = self.spot_data[color_by]

            if datatype == 'categorical' or is_categorical_dtype(col):
                self._colors = self._categorical_colors(col)
            elif datatype == 'numeric' or is_numeric_dtype(col):
                self._colors = self._numeric_colors(col)
            else:
                raise InvalidParameterValueError(
                    f"Invalid color_by value: '{color_by}'. Must be a column name in "
                    "spot_data with categorical or numeric data or a tuple where the "
                    "data type is specified (column_name, 'categorical'|'numeric').")
            self._single_color = None
            return

        c = self.kwargs.get('color', 'black')

        if c in self._DYE_COLOR_SET or is_color_code(c):
            # Single color: store the scalar; _build_tracks handles broadcast
            # without allocating an N-length array.
            self._single_color = c
            self._colors = None
            return

        n_tracks = self._track_uids.size
        if c == 'random_greys':
            colors = random_grey(n=n_tracks, code='hex', a=1.0)
        elif c in ('random', 'random_colors', 'random_colours'):
            colors = random_color(n=n_tracks, code='hex', a=1.0)
        else:
            raise InvalidParameterValueError(
                f"Invalid color parameter: {c}. Must be a valid color name, hex "
                "code, or one of ['random', 'random_greys'].")

        # Map one color per track onto every spot via run-length repeat.
        self._colors = np.repeat(np.asarray(colors, dtype=object), self._run_lengths)
        self._single_color = None

    def _categorical_colors(self, col: pd.Series) -> np.ndarray:
        palette = self.kwargs.get('palette', 'tab10')
        if isinstance(palette, (str, list)):
            categories = col.dropna().unique().tolist()
            palette = retrieve_palette(categories, palette)
        elif not isinstance(palette, dict):
            raise PlottingError(
                f"Invalid palette type: {type(palette)}. Must be str, list, or dict.")
        return col.map(palette).fillna("#000000FF").to_numpy()

    def _numeric_colors(self, col: pd.Series) -> np.ndarray:
        cmap = retrieve_cmap(self.kwargs.get('cmap', 'viridis'))
        norm, vals = cmap_lut(
            col,
            min=self.kwargs.get('lut_vmin'),
            max=self.kwargs.get('lut_vmax'),
        )
        try:
            # RGBA (N,4) float array — far cheaper downstream than an object column.
            return cmap(norm(np.asarray(vals, dtype=float)))
        except Exception as e:
            raise PlottingError(
                f"Error applying quantitative colormap: '{cmap}' to data: {e}")

    # ------------------------------------------------------------------ #
    # Segment / track drawing
    # ------------------------------------------------------------------ #

    def _plot_coords(self, *, polar: bool = False):
        """Return per-spot plotting coordinates (polar-transformed if requested)."""
        x, y = self._x, self._y
        if not polar:
            return x, y
        starts = self._track_starts
        x0 = np.repeat(x[starts], self._run_lengths)
        y0 = np.repeat(y[starts], self._run_lengths)
        dx, dy = x - x0, y - y0
        return np.arctan2(dy, dx), np.hypot(dx, dy)

    @clock
    def _build_tracks(self, ax: plt.Axes, *, polar: bool = False):
        """Build and plot track segments as one LineCollection from cached arrays."""
        x, y = self._x, self._y
        same = self._same_track

        if x.size < 2:
            self.segments = np.empty((0, 2, 2), dtype=float)
            self._seg_mask = np.empty(0, dtype=bool)
            return

        plot_x, plot_y = self._plot_coords(polar=polar)

        # Stack consecutive points into segments in one shot, then drop the
        # cross-track segments via the boolean mask.
        segs = np.stack(
            (np.column_stack((plot_x[:-1], plot_y[:-1])),
             np.column_stack((plot_x[1:], plot_y[1:]))),
            axis=1,
        )
        # Store both the masked (drawn) segments and the mask itself so the
        # animator can index colors consistently.
        self._seg_mask = same
        self.segments = segs[same]

        color_arg = self._segment_color_arg(same)

        ax.add_collection(
            LineCollection(
                self.segments,
                colors=color_arg,
                linewidths=self.kwargs.get('lw', 1),
            )
        )

    def _live_line_collection(self, ax: plt.Axes) -> LineCollection:
        """Create an empty LineCollection used as the growing-track artist."""
        lc = LineCollection(
            np.empty((0, 2, 2)),
            colors=self._segment_color_arg(self._seg_mask),
            linewidths=self.kwargs.get('lw', 1),
            zorder=11,
        )
        ax.add_collection(lc)
        return lc

    def _segment_colors_for_mask(self, mask: np.ndarray):
        """Colors for the within-track segments selected by `mask` (animation)."""
        if self._single_color is not None:
            return self._single_color
        if self._colors is None:
            return 'black'
        # self.segments already correspond to within-track segments; index the
        # per-segment colors (color of the segment's starting spot) with `mask`.
        seg_colors_all = self._colors[:-1][self._seg_mask]
        return seg_colors_all[mask]

    def _segment_color_arg(self, same: np.ndarray):
        """Resolve the `colors=` argument for the LineCollection."""
        if self._single_color is not None:
            return self._single_color
        if self._colors is None:
            return 'black'

        seg_colors = self._colors[:-1][same]
        # Collapse a uniform color to a single scalar (faster matplotlib path).
        if seg_colors.size and self._all_same_color(seg_colors):
            return seg_colors[0]
        return seg_colors

    @staticmethod
    def _all_same_color(colors: np.ndarray) -> bool:
        """True if every entry in `colors` is the same color."""
        first = colors[0]
        if colors.dtype != object:
            return bool((colors == first).all())
        # Object array (e.g. RGBA tuples/arrays): compare per element safely.
        return all(np.array_equal(c, first) for c in colors)

    @clock
    def _head_markers(self, ax: plt.Axes, *, polar: bool = False):
        """Draw markers at the last spot of each track."""
        ends = getattr(self, '_track_ends', None)
        if ends is None or ends.size == 0:
            return

        if self._single_color is not None:
            colors = self._single_color
        elif self._colors is not None:
            colors = self._colors[ends]
        else:
            colors = 'black'

        if polar:
            starts = self._track_starts
            dx = self._x[ends] - self._x[starts]
            dy = self._y[ends] - self._y[starts]
            px, py = np.arctan2(dy, dx), np.hypot(dx, dy)
        else:
            px, py = self._x[ends], self._y[ends]

        outline = self.kwargs.get('outline_head', True)
        fill = self.kwargs.get('fill_head', False)
        ax.scatter(
            px, py,
            marker=self.kwargs.get('head_shape', 'o'),
            s=self.kwargs.get('head_size', 10),
            edgecolor=colors if outline else 'none',
            facecolor=colors if fill else 'none',
            linewidths=self.kwargs.get('head_outline_width', 1.0),
            zorder=12,
        )

    # ------------------------------------------------------------------ #
    # Styling helpers (dict lookups, no per-call dict construction)
    # ------------------------------------------------------------------ #

    def _background_color(self):
        self.face_color = self._BG_FACE.get(self.kwargs.get('background', 'white'), 'white')

    def _grid_color(self, coord_system: str = 'cartesian'):
        bg = self.kwargs.get('background', 'white')
        if coord_system == 'cartesian':
            self.grid_color, self.grid_alpha = \
                self._BG_GRID_CART.get(bg, ('gainsboro', 0.5))
        else:
            self.grid_color, self.grid_a_alpha, self.grid_alpha = \
                self._BG_GRID_POLAR.get(bg, ('lightgrey', 0.7, 0.8))

    def _grid_style(self, ax: plt.Axes, coord_system: str = 'cartesian'):
        if coord_system == 'cartesian':
            ax.grid(True, which='both', axis='both', color=self.grid_color,
                    linestyle='-.', linewidth=1, alpha=self.grid_alpha)
            return

        style = self.gridstyle
        gc, ga = self.grid_color, self.grid_a_alpha
        match style:
            case 'simple-1' | 'simple-2':
                ax.xaxis.grid(True, color=gc, linestyle='-', linewidth=1, alpha=ga)
                ax.yaxis.grid(False)
                parity = 1 if style == 'simple-1' else 0
                for i, line in enumerate(ax.get_xgridlines()):
                    if i % 2 == parity:
                        line.set_color('none')
            case 'dartboard-1' | 'dartboard-2':
                ax.grid(True, lw=0.75, color=gc, alpha=ga)
                parity = 0 if style == 'dartboard-1' else 1
                for i, line in enumerate(ax.get_xgridlines()):
                    if i % 2 == parity:
                        line.set_linestyle('-.')
                        line.set_color(gc)
                        line.set_linewidth(0.75)
                        line.set_alpha(ga)
                for line in ax.get_ygridlines():
                    line.set_linestyle('--')
                    line.set_color(gc)
                    line.set_linewidth(0.75)
                    line.set_alpha(ga)
            case 'spindle':
                ax.xaxis.grid(True, color=gc, linestyle='-', linewidth=1, alpha=ga)
                ax.yaxis.grid(False)
            case 'radial':
                ax.xaxis.grid(False)
                ax.yaxis.grid(True, color=gc, linestyle='-', linewidth=1, alpha=ga)

    def _annotate_r_axis(self, ax: plt.Axes):
        match self.kwargs.get('r_axis', 'none'):
            case 'minimal':
                ax.set_yticklabels([])
                gc = getattr(self, 'grid_color', 'grey')
                ax.scatter(0, self.y_max_global + 35, color=gc,
                           marker='.', s=5, clip_on=False)
                ax.text(0, self.y_max_global + 50, self.y_max_label_global,
                        va='center', fontsize=10, color=gc, clip_on=False)
            case 'detailed':
                gc = getattr(self, 'grid_color', 'grey')
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, color=gc)
            case _:
                ax.set_yticklabels([])

    def _annotate_theta_axis(self, ax: plt.Axes):
        text_color = self.kwargs.get('text_color', 'black')
        if self.kwargs.get('theta_axis', 'none') == 'detailed':
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, color=text_color)
        else:
            ax.set_xticklabels([])

    @staticmethod
    def frame_interval_ms(fps: float) -> float:
        """Return the interval between frames in milliseconds."""
        if fps <= 0:
            raise ValueError("Frame rate must be positive.")
        return 1000.0 / fps

    def _new_axes(self, *, polar: bool = False):
        """Create a fresh figure/axes matching the static plot's framing."""
        self._background_color()
        if polar:
            fig, ax = plt.subplots(figsize=(12.5, 9.5),
                                   subplot_kw={'projection': 'polar'})
            self._get_radius()
            ax.set_ylim(0, self.y_max_global)
            ax.spines['polar'].set_visible(False)
            self._annotate_r_axis(ax)
            self._annotate_theta_axis(ax)
        else:
            fig, ax = plt.subplots(figsize=(13, 10))
            if self._x.size:
                ax.set_xlim(self._x.min(), self._x.max())
                ax.set_ylim(self._y.min(), self._y.max())
            ax.set_aspect('equal', adjustable='box')
            tc = self.kwargs.get('text_color', 'black')
            ax.set_xlabel('x_coordinate [µm]', color=tc)
            ax.set_ylabel('y_coordinate [µm]', color=tc)
            self._apply_cartesian_ticks(ax, tc)
        ax.set_facecolor(self.face_color)
        ax.grid(False)
        return fig, ax


reconstruct = ReconstructTracks().reconstruct