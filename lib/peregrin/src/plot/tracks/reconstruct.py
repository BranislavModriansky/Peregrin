from inspect import Arguments

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

import warnings

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
        # In ipympl the figure that is the cell's display object is what renders;
        # prefer the animation figure if one exists.
        fig = getattr(self, '_anim_fig', None) or self.figure
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        plt.show()
        return fig

    def save(self, path, **kwargs):
        self.figure.savefig(path, **kwargs)

    # Animation ---------------------------------------------------------------
    # ------------------------------------------------------------------ #
    # Animation
    # ------------------------------------------------------------------ #
    def animate(
        self,
        *,
        loop: bool = True,
        speed: float = 1.0,
        interval: float = 40.0,
        frames: int = None,
        show_heads: bool = True,
        repeat_delay: float = 0.0,
        controls: bool = True,
        blit: bool = True,
        mode: str = "auto",       # 'auto' | 'live' | 'jshtml' | 'video'
        bake_every: int = None,   # frames between background re-bakes (live mode)
    ):
        """
        Growing-trajectory animation.

        Two rendering strategies, because they solve two different problems:

        * mode='live'   -> real matplotlib blitting with a cached background.
                            Instead of re-rendering ALL revealed segments every
                            frame (the classic slowdown once you have many
                            tracks), previously revealed segments are baked
                            into a snapshot bitmap every `bake_every` frames.
                            Each frame then only draws the *new* segments +
                            the head markers on top of that snapshot. Needs a
                            persistent-canvas backend (%matplotlib widget,
                            Qt, Tk) to actually pay off.

        * mode='jshtml' -> frames are computed once, encoded, and handed to
                            the browser as a JS player (or html5 video for
                            mode='video'). This is the right choice for the
                            default inline Jupyter backend, where live
                            blitting can't help because there's no
                            persistent canvas: rendering is decoupled from
                            playback entirely, so it's smooth regardless of
                            per-frame Python cost.

        * mode='auto'   -> picks 'jshtml' unless an interactive backend is
                            already active (detected via the canvas class).
        """
        b = self._builder

        fig, ax = b._new_axes(polar=self.polar)
        if controls:
            fig.subplots_adjust(right=0.80)

        plot_x, plot_y = b._plot_coords(polar=self.polar)
        starts = b._track_starts
        run_lengths = b._run_lengths

        if run_lengths.size == 0:
            self._anim_fig = fig
            plt.close(fig)
            return FuncAnimation(fig, lambda _f: (), frames=1, blit=False)

        # --- precompute reveal schedule -----------------------------------
        max_len = int(run_lengths.max())
        n_frames = frames if frames is not None else max(2, int(max_len / max(speed, 1e-6)))
        reveal_grid = np.linspace(1, max_len, n_frames).astype(np.intp)

        seg_lens = b._within_track_seg_lengths
        if seg_lens.sum():
            seg_local_idx = np.concatenate([np.arange(c) for c in seg_lens])
        else:
            seg_local_idx = np.empty(0, np.intp)
        seg_appear = seg_local_idx + 1

        segments_all = b.segments
        full_colors = b._segment_colors_for_mask(np.ones(seg_appear.size, dtype=bool))
        uniform_color = isinstance(full_colors, str)

        order = np.argsort(seg_appear, kind='stable')
        seg_sorted = segments_all[order]
        seg_appear_sorted = seg_appear[order]
        colors_sorted = None if uniform_color else np.asarray(full_colors)[order]
        seg_count = np.searchsorted(seg_appear_sorted, reveal_grid, side='right')

        head_xy = None
        if show_heads:
            has_pt = run_lengths > 0
            last_idx = np.minimum(reveal_grid[:, None], (run_lengths - 1)[None, :])
            head_idx = starts[None, :] + np.maximum(last_idx, 0)
            hx = plot_x[head_idx][:, has_pt]
            hy = plot_y[head_idx][:, has_pt]
            head_xy = np.stack((hx, hy), axis=-1)

        lw = b.kwargs.get('lw', 1)

        # Decide rendering strategy -----------------------------------------
        if mode == "auto":
            backend_cls = type(fig.canvas).__name__
            mode = "live" if backend_cls not in ("FigureCanvasAgg",) else "jshtml"

        if mode == "live":
            anim = self._animate_live(
                fig, ax, seg_sorted, seg_count, colors_sorted, uniform_color,
                full_colors, head_xy, head_scatter_kwargs=dict(
                    marker=b.kwargs.get('head_shape', 'o'),
                    s=b.kwargs.get('head_size', 10)) if show_heads else None,
                lw=lw, n_frames=n_frames, interval=interval, blit=blit,
                loop=loop, repeat_delay=repeat_delay, bake_every=bake_every,
            )
        else:
            # Simple create-once/mutate-in-place loop; blit is irrelevant
            # here since matplotlib's Agg canvas has no persistent state
            # between frames anyway -- correctness matters, not blitting.
            base_lc = LineCollection(np.empty((0, 2, 2)), linewidths=lw, zorder=11)
            if uniform_color:
                base_lc.set_color(full_colors)
            ax.add_collection(base_lc)

            head_scatter = None
            if show_heads:
                head_scatter = ax.scatter(
                    [], [], marker=b.kwargs.get('head_shape', 'o'),
                    s=b.kwargs.get('head_size', 10), zorder=13)

            def _draw_frame(i):
                k = int(seg_count[i])
                base_lc.set_segments(seg_sorted[:k])
                if not uniform_color:
                    base_lc.set_color(colors_sorted[:k])
                arts = [base_lc]
                if head_scatter is not None:
                    head_scatter.set_offsets(head_xy[i])
                    arts.append(head_scatter)
                return arts

            anim = FuncAnimation(
                fig, _draw_frame, frames=n_frames, interval=interval,
                blit=False, repeat=loop, repeat_delay=repeat_delay,
            )

        self._anim = anim
        self._anim_fig = fig

        # Prevent matplotlib from also emitting a static duplicate of the
        # figure as a separate cell output in Jupyter.
        plt.close(fig)

        if controls and mode == "live":
            self._attach_controls(fig, anim, n_frames,
                                   lambda idx: anim._draw_frame_public(int(idx)))

        if mode in ("jshtml", "video"):
            from matplotlib.animation import HTMLWriter  # noqa: F401
            from IPython.display import HTML
            html = anim.to_jshtml() if mode == "jshtml" else anim.to_html5_video()
            self._html = HTML(html)
            return self._html

        return anim

    def _animate_live(self, fig, ax, seg_sorted, seg_count, colors_sorted,
                       uniform_color, full_colors, head_xy, head_scatter_kwargs,
                       lw, n_frames, interval, blit, loop, repeat_delay, bake_every):
        """
        Manual background-caching blit: previously revealed segments are
        baked into a snapshot bitmap periodically; each frame only draws the
        delta segments + head markers, so per-frame cost stays proportional
        to what's *new*, not to how much has accumulated so far.
        """
        frozen_lc = LineCollection(np.empty((0, 2, 2)), linewidths=lw, zorder=11)
        if uniform_color:
            frozen_lc.set_color(full_colors)
        ax.add_collection(frozen_lc)

        delta_lc = LineCollection(np.empty((0, 2, 2)), linewidths=lw, zorder=11)
        if uniform_color:
            delta_lc.set_color(full_colors)
        ax.add_collection(delta_lc)

        head_scatter = None
        if head_scatter_kwargs is not None:
            head_scatter = ax.scatter([], [], zorder=13, **head_scatter_kwargs)

        bake_every = bake_every or max(1, n_frames // 20)
        state = {"baked_k": 0, "background": None}

        def _bake(k):
            frozen_lc.set_segments(seg_sorted[:k])
            if not uniform_color:
                frozen_lc.set_color(colors_sorted[:k])
            delta_lc.set_segments(np.empty((0, 2, 2)))
            if head_scatter is not None:
                head_scatter.set_offsets(np.empty((0, 2)))
            fig.canvas.draw()
            state["background"] = fig.canvas.copy_from_bbox(ax.bbox)
            state["baked_k"] = k

        def _init():
            _bake(0)
            return [frozen_lc, delta_lc] + ([head_scatter] if head_scatter else [])

        def _draw_frame(i):
            k = int(seg_count[i])
            if state["background"] is None:
                _bake(0)
            fig.canvas.restore_region(state["background"])

            delta_lc.set_segments(seg_sorted[state["baked_k"]:k])
            if not uniform_color:
                delta_lc.set_color(colors_sorted[state["baked_k"]:k])
            ax.draw_artist(delta_lc)

            arts = [delta_lc]
            if head_scatter is not None:
                head_scatter.set_offsets(head_xy[i])
                ax.draw_artist(head_scatter)
                arts.append(head_scatter)

            fig.canvas.blit(ax.bbox)

            if (i + 1) % bake_every == 0 or i == n_frames - 1:
                _bake(k)
            return arts

        anim = FuncAnimation(
            fig, _draw_frame, init_func=_init, frames=n_frames,
            interval=interval, blit=blit, repeat=loop, repeat_delay=repeat_delay,
        )
        anim._draw_frame_public = _draw_frame  # expose for the manual slider/controls
        return anim

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
            self.spot_data = categorize(self.spot_data, 
                                        {'condition': self.conditions, 
                                         'replicate': self.replicates},
                                        **self.kwargs)

        sort_cols = (['track_uid', 'time_point'] if ignore
                     else ['condition', 'replicate', 'track_id', 'time_point'])

        self.spot_data = (
            self.spot_data
            .reset_index(drop=True)
            .sort_values(by=['track_uid', 'time_point'], kind='mergesort')
        )

        if not is_empty(self.track_data):
            if not ignore:
                self.track_data = categorize(self.track_data, 
                                             {'condition': self.conditions, 
                                              'replicate': self.replicates},
                                             **self.kwargs)
                
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

    # NOTE: _live_line_collection removed — the animator builds its own
    # LineCollection directly (create-once, mutate-in-place recipe).

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