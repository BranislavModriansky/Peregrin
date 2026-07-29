from inspect import Arguments
from re import match
from typing import Any, Optional

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

from ...categorizer import categorize
from ..painter import paint
from ...various import Values, get_aliases, is_empty, clock
from ...compute.stats import Stats


class AnimateTracks:
    """
    Handles growing-trajectory animation for a reconstructed set of tracks.

    Kept separate from :class:`ReconstructTracks` so that the (heavy) animation
    machinery is decoupled from the static plotting/data pipeline. Instances
    are bound to a *builder* (a :class:`ReconstructTracks`) whose cached arrays
    and styling kwargs drive every frame.
    """

    def __init__(self, builder):
        self._builder = builder
        self._anim = None
        self._anim_fig = None
        self._html = None

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

        fig, ax = b._new_axes(polar=b.align_at_start)
        if controls:
            fig.subplots_adjust(right=0.80)

        plot_x, plot_y = b._plot_coords(polar=b.align_at_start)
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
        head_colors = None
        if show_heads:
            has_pt = run_lengths > 0
            last_idx = np.minimum(reveal_grid[:, None], (run_lengths - 1)[None, :])
            head_idx = starts[None, :] + np.maximum(last_idx, 0)
            hx = plot_x[head_idx][:, has_pt]
            hy = plot_y[head_idx][:, has_pt]
            head_xy = np.stack((hx, hy), axis=-1)
            head_colors = b._head_colors(has_pt)

        lw = b.kwargs.get('lw', 1)

        # Decide rendering strategy -----------------------------------------
        if mode == "auto":
            backend_cls = type(fig.canvas).__name__
            mode = "live" if backend_cls not in ("FigureCanvasAgg",) else "jshtml"

        if mode == "live":
            outline = b.kwargs.get('outline_head', True)
            fill = b.kwargs.get('fill_head', False)
            anim = self._animate_live(
                fig, ax, seg_sorted, seg_count, colors_sorted, uniform_color,
                full_colors, head_xy, head_scatter_kwargs=dict(
                    marker=b.kwargs.get('head_shape', 'o'),
                    s=b.kwargs.get('head_size', 10),
                    linewidths=b.kwargs.get('head_outline_width', 1.0),
                ) if show_heads else None,
                head_colors=head_colors, head_outline=outline, head_fill=fill,
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
                outline = b.kwargs.get('outline_head', True)
                fill = b.kwargs.get('fill_head', False)
                head_scatter = ax.scatter(
                    [], [], marker=b.kwargs.get('head_shape', 'o'),
                    s=b.kwargs.get('head_size', 10),
                    linewidths=b.kwargs.get('head_outline_width', 1.0),
                    zorder=13)
                head_fc = head_colors if fill else 'none'
                head_ec = head_colors if outline else 'none'

            def _draw_frame(i):
                k = int(seg_count[i])
                base_lc.set_segments(seg_sorted[:k])
                if not uniform_color:
                    base_lc.set_color(colors_sorted[:k])
                arts = [base_lc]
                if head_scatter is not None:
                    head_scatter.set_offsets(head_xy[i])
                    if head_colors is not None:
                        head_scatter.set_facecolor(head_fc)
                        head_scatter.set_edgecolor(head_ec)
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
            b._attach_controls(fig, anim, n_frames,
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
                      lw, n_frames, interval, blit, loop, repeat_delay, bake_every,
                      head_colors=None, head_outline=True, head_fill=False):
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
        head_fc = head_ec = 'none'
        if head_scatter_kwargs is not None:
            head_scatter = ax.scatter([], [], zorder=13, **head_scatter_kwargs)
            if head_colors is not None:
                head_fc = head_colors if head_fill else 'none'
                head_ec = head_colors if head_outline else 'none'

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
                if head_colors is not None:
                    head_scatter.set_facecolor(head_fc)
                    head_scatter.set_edgecolor(head_ec)
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


class ReconstructTracks:
    """
    Container returned by :func:`reconstruct`.

    Holds the static figure plus all the cached plotting arrays needed to
    (re)build the tracks. Animation is delegated to :class:`AnimateTracks`
    via :meth:`animate`.
    """

    ALIASES = {
        'color':   ['color', 'colour'],
        'color_by': ['color_by', 'colour_by', 'colorby', 'colourby'],
        'grid_lw': ['grid_linewidth', 'grid_line_width', 'grid_lw'],
    }

    def __init__(self, builder=None, figure=None):
        self._builder = builder
        self.figure = figure
        self._use_polar = False
        self._segments = None
        self._animator = None

    # Convenience passthroughs -------------------------------------------------
    @property
    def fig(self) -> plt.Figure:
        return self.figure

    def _repr_html_(self):
        # Let Jupyter render the static figure when the container is displayed.
        try:
            from io import BytesIO
            import base64
            buf = BytesIO()
            self.figure.savefig(buf, format='png', bbox_inches='tight')
            data = base64.b64encode(buf.getvalue()).decode('ascii')
            return f'<img src="data:image/png;base64,{data}"/>'
        except Exception:
            return None

    @property
    def segments(self) -> np.ndarray:
        # Prefer segments built on this instance; fall back to a wrapped builder.
        if getattr(self, '_segments', None) is not None:
            return self._segments
        if getattr(self, '_builder', None) is not None:
            return self._builder.segments
        return np.empty((0, 2, 2), dtype=float)

    def show(self):
        # In ipympl the figure that is the cell's display object is what renders;
        # prefer the animation figure if one exists.
        anim_fig = getattr(self._animator, '_anim_fig', None) if self._animator else None
        fig = anim_fig or self.figure
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        plt.show()
        return fig

    def save(self, path, **kwargs):
        self.figure.savefig(path, **kwargs)

    def animate(self, **kwargs):
        """
        Create and return a growing-trajectory animation.

        Delegates to :class:`AnimateTracks`, which is bound to the builder
        holding this reconstruction's cached arrays and styling kwargs.
        """
        builder = self._builder if self._builder is not None else self
        self._animator = AnimateTracks(builder)
        return self._animator.animate(**kwargs)

    def reconstruct(
        self,
        spot_data: pd.DataFrame,
        track_data: pd.DataFrame = None,
        *,
        align_at_start: bool = False,
        categories: Optional[dict[str, list[Any]]] = None,
        **kwargs
    ) -> "ReconstructTracks":

        self.spot_data = spot_data.copy() if spot_data is not None else pd.DataFrame()
        self.align_at_start = align_at_start
        self.kwargs = get_aliases(kwargs, self.ALIASES)
        self.track_data = track_data.copy() if track_data is not None else None
        self.categories = categories

        smoothing = self.kwargs.get('smoothing_index')

        self._arrange_data()

        # Colors are cheap and often the only thing that changes; recompute every call.
        self._assign_color()

        # Build the static figure and keep it, but return the container so
        # callers can chain .animate() / .show() / .save().
        self.figure = self.polar() if self.align_at_start else self.cartesian()
        return self




    def cartesian(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(13, 10))
        self._build_tracks(ax)

        x = self._x
        self.x_gap = (x.max() - x.min()) * 0.025
        if x.size:
            ax.set_xlim(x.min() - self.x_gap, x.max() + self.x_gap)
            ax.set_ylim(self._y.min() - self.x_gap, self._y.max() + self.x_gap)

        ax.set_aspect('equal', adjustable='box')
        text_color = self.kwargs.get('text_color', 'black')
        ax.set_xlabel('x_coordinate [µm]', color=text_color)
        ax.set_ylabel('y_coordinate [µm]', color=text_color)
        ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=text_color)

        # self._background_color()
        # ax.set_facecolor(self.face_color)
        self._cartesian_ticks(ax)
        self._cartesian_spines(ax)

        if self.kwargs.get('grid', True):
            self._style_grid(ax, **self.kwargs)
        else:
            ax.grid(False)

        if self.kwargs.get('show_heads', True):
            self._head_markers(ax, polar=False)
        if self.kwargs.get('hide_backdrop', False):
            fig.set_facecolor('none')

        return plt.gcf()


    def polar(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12.5, 9.5),
                               subplot_kw={'projection': 'polar'})
        self._get_radius()

        text_color = self.kwargs.get('text_color', 'black')
        ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=text_color)
        ax.set_ylim(0, self.y_max)
        ax.spines['polar'].set_visible(False)

        self._build_tracks(ax, polar=True)

        # self._background_color()
        # ax.set_facecolor(self.face_color)
        if self.kwargs.get('grid', True):
            self._style_grid(ax, **self.kwargs)
        else:
            ax.grid(False)

        if isinstance(self.kwargs.get('annotate_r_axis', 'detailed'), str):
            self._annotate_r_axis(ax)
        else:
            ax.set_yticklabels([])
        if isinstance(self.kwargs.get('annotate_theta_axis', 'angular'), str):
            self._annotate_theta_axis(ax)
        else:
            ax.set_xticklabels([])

        if self.kwargs.get('show_heads', True):
            self._head_markers(ax, polar=True)
        if self.kwargs.get('hide_backdrop', False):
            fig.set_facecolor('none')

        return plt.gcf()
    
    def _cartesian_spines(self, ax):
        for spine in ax.spines.values():
            spine.set_color(self.kwargs.get('frame_color', 'black'))

    def _cartesian_ticks(self, ax):
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.tick_params(axis='both', which='major', labelsize=8, colors=self.kwargs.get('annotation_color', 'black'))


    def _arrange_data(self):
        if self.categories is not None:
            self.spot_data = categorize(self.spot_data, self.categories, **self.kwargs)

        if 'track_uid' not in self.spot_data.columns:
            self.spot_data['track_uid'] = self.spot_data.index

        self.spot_data = (
            self.spot_data
            .reset_index(drop=True)
            .sort_values(by=['track_uid', 'time_point'], kind='mergesort')
        )

        if not is_empty(self.track_data):
            if self.categories is not None:
                try:
                    self.track_data = categorize(self.track_data, self.categories, **self.kwargs)
                except Exception as e:
                    print(f"Error categorizing track_data: {e}")
            self.track_data = self.track_data.sort_values(['track_uid'], kind='stable')

        # Only set the MultiIndex when we actually need category-based grouping
        # (smoothing path). For ignore mode we work purely on track_uid via
        # cached numpy arrays, so skip the expensive set_index.
        if self.categories is not None:
            if list(self.spot_data.index.names) != self.KEY_COLS:
                self.spot_data = self.spot_data.set_index(self.KEY_COLS)
            if (not is_empty(self.track_data)
                    and list(self.track_data.index.names) != self.KEY_COLS):
                self.track_data = self.track_data.set_index(self.KEY_COLS)

        self._cache_arrays(self.categories is None)


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


    def _get_radius(self):
        """Global maximum radius, computed from cached numpy arrays (no groupby)."""
        if self._x.size == 0:
            self.y_max = 100.0
            return

        starts = self._track_starts
        x0 = np.repeat(self._x[starts], self._run_lengths)
        y0 = np.repeat(self._y[starts], self._run_lengths)
        r_max = np.hypot(self._x - x0, self._y - y0).max()

        self.y_max = r_max + 100.0

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


    def _assign_color(self):

        color = self.kwargs.get('color', 'black')

        # 'random' / 'random greys' produce ONE color per trajectory. We ask
        # the painter for exactly n_tracks colors and expand them to per-spot
        # via run-length repeat so whole tracks share a color.
        if color in ('random', 'random greys'):
            n_tracks = self._track_starts.size
            per_track = paint(
                # Give the painter a frame whose unique-index count == n_tracks
                # so it generates one color per track.
                pd.DataFrame(index=np.arange(n_tracks)),
                color=color,
            )
            per_track = np.asarray(per_track)
            self._colors = np.repeat(per_track, self._run_lengths, axis=0)
            self._single_color = None
            return

        c = paint(self.spot_data, **self.kwargs)

        if isinstance(c, str):
            self._single_color = c
            self._colors = None
            return
        else:
            # Map one color per track onto every spot via run-length repeat.
            self._colors = c
            self._single_color = None


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

    def _build_tracks(self, ax: plt.Axes, *, polar: bool = False):
        """Build and plot track segments as one LineCollection from cached arrays."""
        x, y = self._x, self._y
        same = self._same_track

        if x.size < 2:
            self._segments = np.empty((0, 2, 2), dtype=float)
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
        self._segments = segs[same]

        color_arg = self._segment_color_arg(same)

        ax.add_collection(
            LineCollection(
                self._segments,
                colors=color_arg,
                linewidths=self.kwargs.get('lw', 1),
            )
        )


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

        if self.kwargs.get('display_uid', False):
            self._annotate_uids(ax, px, py, colors)

    def _annotate_uids(self, ax: plt.Axes, px, py, colors):
        """Draw the track_uid label next to each track head."""
        uids = getattr(self, '_track_uids', None)
        if uids is None or len(uids) == 0:
            return

        # Resolve a per-track color list so labels can match their heads.
        if isinstance(colors, str):
            label_colors = [colors] * len(uids)
        else:
            label_colors = list(colors)

        fontsize = self.kwargs.get('uid_fontsize', 9)
        offset = self.kwargs.get('uid_offset', 2)

        for x, y, uid, col in zip(px, py, uids, label_colors):
            ax.annotate(
                str(uid),
                xy=(x, y),
                xytext=(offset, offset),
                textcoords='offset points',
                fontsize=fontsize,
                color=col,
                clip_on=True,
                zorder=14,
            )

    def _head_colors(self, has_pt: np.ndarray):
        """Per-track head colors, aligned to the tracks kept by `has_pt`.

        Returns either a scalar color (uniform) or an array in the same track
        order used to build `head_xy`.
        """
        if self._single_color is not None:
            return self._single_color
        ends = getattr(self, '_track_ends', None)
        if self._colors is None or ends is None or ends.size == 0:
            return 'black'
        colors = np.asarray(self._colors)[ends]
        return colors[has_pt]


    def _background(self, ax: plt.Axes, fig: plt.Figure):
        """Set the background color of the figure and axes."""
        ax.set_facecolor(self.kwargs.get('face_color', 'white'))
        fig.set_facecolor(self.kwargs.get('face_color', 'white'))


    def _style_grid(self, ax: plt.Axes, **kwargs):

        grid_color = self.kwargs.get('grid_color', 'gainsboro')
        grid_lw    = self.kwargs.get('grid_lw', 0.75)
        grid_ls    = self.kwargs.get('grid_ls', '-')

        # Cartesian grid
        if not self.align_at_start:
            ax.grid(
                True, 
                which='both', 
                axis='both', 
                color=grid_color,
                linestyle=grid_ls, 
                linewidth=grid_lw,
            )
        
        # Polar grid
        else:
            ax.grid(
                True, 
                lw=grid_lw,
                color=grid_color
            )

            grid_attributes = {
                'grid_ls_cardinal': grid_ls,
                'grid_color_cardinal': grid_color,
                'grid_ls_diagonal': grid_ls,
                'grid_color_diagonal': grid_color,
                'grid_ls_radial': grid_ls,
                'grid_color_radial': grid_color,
            }

            for attr in list(grid_attributes.keys()):
                if self.kwargs.get(attr) is not None:
                    grid_attributes[attr] = self.kwargs.get(attr)
                else: 
                    grid_attributes[attr] = self.kwargs.get(attr, grid_attributes[attr])

            for i, line in enumerate(ax.get_xgridlines()):
                if i % 2 == 0:
                    line.set_linestyle(grid_attributes['grid_ls_cardinal'])
                    line.set_color(grid_attributes['grid_color_cardinal'])
                else:
                    line.set_linestyle(grid_attributes['grid_ls_diagonal'])
                    line.set_color(grid_attributes['grid_color_diagonal'])

            for line in ax.get_ygridlines():
                line.set_linestyle(grid_attributes['grid_ls_radial'])
                line.set_color(grid_attributes['grid_color_radial'])


    def _annotate_r_axis(self, ax: plt.Axes):
        text_color = self.kwargs.get('text_color', 'dimgrey')

        # Use the outermost radial tick within the current limit as the label.
        r_lim = ax.get_ylim()[1]
        ticks = np.asarray(ax.get_yticks(), dtype=float)
        ticks = ticks[(ticks > 0) & (ticks <= r_lim + 1e-9)]
        if ticks.size:
            self.y_max = float(ticks.max())
            self.y_max_lbl = f"{round(self.y_max)} μm"

        match self.kwargs.get('annotate_r_axis', 'minimal'):
            case 'minimal':
                ax.set_yticklabels([])
                # ax.scatter(np.deg2rad(20), self.y_max, color=text_color,
                #            marker='.', s=5, clip_on=False)
                ax.text(np.deg2rad(25), self.y_max + 75, self.y_max_lbl, horizontalalignment='center', verticalalignment='center',
                        va='center', fontsize=10, color=text_color, clip_on=False)
            case 'detailed':
                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, color=text_color)
            case _:
                ax.set_yticklabels([])

    def _annotate_theta_axis(self, ax: plt.Axes):
        text_color = self.kwargs.get('text_color', 'dimgrey')
        match self.kwargs.get('annotate_theta_axis', 'compass'):
            case 'angular':
                # Pin ticks so relabeling doesn't trigger the fixed-ticks warning.
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, color=text_color)
            case 'compass':
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'],
                                   fontsize=10, color=text_color)
            case _:
                ax.set_xticklabels([])

    # @staticmethod
    # def frame_interval_ms(fps: float = 10) -> float:
    #     """Return the interval between frames in milliseconds."""
    #     if fps <= 0:
    #         raise ValueError("Frame rate must be positive.")
    #     return 1000.0 / fps

    def _new_axes(self, *, polar: bool = False):
        """Create a fresh figure/axes matching the static plot's framing.

        Reuses the same grid/axis styling as the already-built static
        reconstruction so the animation inherits its look without needing
        grid/axis arguments passed to :meth:`animate`.
        """
        # self._background_color()
        if polar:
            fig, ax = plt.subplots(figsize=(12.5, 9.5),
                                   subplot_kw={'projection': 'polar'})
            self._get_radius()
            tc = self.kwargs.get('text_color', 'black')
            ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=tc)
            ax.set_ylim(0, self.y_max)
            ax.spines['polar'].set_visible(False)

            if self.kwargs.get('grid', True):
                self._style_grid(ax, **self.kwargs)
            else:
                ax.grid(False)

            self._annotate_r_axis(ax)
            self._annotate_theta_axis(ax)

            if self.kwargs.get('hide_backdrop', False):
                fig.set_facecolor('none')
        else:
            fig, ax = plt.subplots(figsize=(13, 10))
            if self._x.size:
                ax.set_xlim(self._x.min() - self.x_gap, self._x.max() + self.x_gap)
                ax.set_ylim(self._y.min() - self.x_gap, self._y.max() + self.x_gap)
            ax.set_aspect('equal', adjustable='box')
            tc = self.kwargs.get('text_color', 'black')
            ax.set_xlabel('x_coordinate [µm]', color=tc)
            ax.set_ylabel('y_coordinate [µm]', color=tc)
            ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=tc)
            self._cartesian_ticks(ax)
            self._cartesian_spines(ax)

            if self.kwargs.get('grid', True):
                self._style_grid(ax, **self.kwargs)
            else:
                ax.grid(False)

            if self.kwargs.get('hide_backdrop', False):
                fig.set_facecolor('none')

        # ax.set_facecolor(self.face_color)
        return fig, ax


reconstruct = ReconstructTracks().reconstruct