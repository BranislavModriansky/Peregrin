from dataclasses import dataclass, field
from typing import Any, Optional

import io
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_hex

from ..._pckg_exceptions._pckg_errors import *
from ..._pckg_exceptions._pckg_warnings import *

from ..painter import paint
from ...various import get_aliases
from .._tooltip_toolkit import tooltip_assets


class AnimateTracks:
    """
    Growing-trajectory animation for a reconstructed set of tracks.

    Kept separate from :class:`ReconstructTracks` so the animation machinery
    stays decoupled from the static plotting pipeline. Instances are bound to
    a builder whose cached arrays and styling kwargs drive every frame.
    """

    def __init__(self, builder: "ReconstructTracks"):
        self._builder = builder
        self._anim = None
        self._anim_fig = None

    def animate(
        self,
        *,
        loop: bool = True,
        speed: float = 1.0,
        interval: float = 40.0,
        frames: Optional[int] = None,
        show_heads: bool = True,
        repeat_delay: float = 0.0,
        mode: str = 'auto',  # 'auto' | 'jshtml' | 'video'
    ):
        b = self._builder
        fig, ax = plt.subplots(
            figsize=(13, 10),
            subplot_kw={'projection': 'polar'} if b.align_at_start else {},
        )
        b._frame_axes(ax)

        plot_x, plot_y = b._plot_coords(polar=b.align_at_start)
        run_lengths = b._run_lengths

        if run_lengths.size == 0:
            plt.close(fig)
            return FuncAnimation(fig, lambda _f: (), frames=1, blit=False)

        max_len = int(run_lengths.max())
        n_frames = frames if frames is not None else max(2, int(max_len / max(speed, 1e-6)))
        reveal_grid = np.linspace(1, max_len, n_frames).astype(np.intp)

        seg_lens = b._within_track_seg_lengths
        seg_local_idx = (
            np.concatenate([np.arange(c) for c in seg_lens])
            if seg_lens.sum() else np.empty(0, np.intp)
        )
        seg_appear = seg_local_idx + 1

        full_colors = b._segment_colors_for_mask(np.ones(seg_appear.size, dtype=bool))
        uniform_color = isinstance(full_colors, str)

        order = np.argsort(seg_appear, kind='stable')
        seg_sorted = b.segments[order]
        colors_sorted = None if uniform_color else np.asarray(full_colors)[order]
        seg_count = np.searchsorted(seg_appear[order], reveal_grid, side='right')

        head_xy = head_colors = None
        if show_heads:
            has_pt = run_lengths > 0
            last_idx = np.minimum(reveal_grid[:, None], (run_lengths - 1)[None, :])
            head_idx = b._track_starts[None, :] + np.maximum(last_idx, 0)
            head_xy = np.stack(
                (plot_x[head_idx][:, has_pt], plot_y[head_idx][:, has_pt]), axis=-1)
            head_colors = b._head_colors(has_pt)

        lc = LineCollection(np.empty((0, 2, 2)),
                            linewidths=b.kwargs.get('lw', 1), zorder=11)
        if uniform_color:
            lc.set_color(full_colors)
        ax.add_collection(lc)

        head_scatter = None
        if show_heads:
            outline = b.kwargs.get('outline_head', True)
            fill = b.kwargs.get('fill_head', False)
            head_scatter = ax.scatter(
                [], [], marker=b.kwargs.get('head_shape', 'o'),
                s=b.kwargs.get('head_size', 10),
                linewidths=b.kwargs.get('head_outline_width', 1.0), zorder=13)
            head_fc = head_colors if fill else 'none'
            head_ec = head_colors if outline else 'none'

        def _draw_frame(i):
            k = int(seg_count[i])
            lc.set_segments(seg_sorted[:k])
            if not uniform_color:
                lc.set_color(colors_sorted[:k])
            arts = [lc]
            if head_scatter is not None:
                head_scatter.set_offsets(head_xy[i])
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
        plt.close(fig)  # avoid a duplicate static figure in Jupyter

        if mode in ('jshtml', 'video'):
            from IPython.display import HTML
            return HTML(anim.to_jshtml() if mode == 'jshtml' else anim.to_html5_video())
        return anim


class ReconstructTracks:
    """
    Track reconstruction plot built with Matplotlib.

    Every artist is tagged with a ``gid`` (``pg:tracks``, ``pg:heads``,
    ``pg:grid``, ``pg:background``, ``pg:figure``) so the exported SVG can be
    edited in place by the JS tooltip (see :class:`InteractiveTracks`).
    """

    ALIASES = {
        'color':    ['color', 'colour'],
        'color_by': ['color_by', 'colour_by', 'colorby', 'colourby'],
        'grid_lw':  ['grid_linewidth', 'grid_line_width', 'grid_lw'],
    }

    CATEGORY_COLS = ('set', 'subset', 'group', 'subgroup', 'subsubgroup')

    def __init__(self):
        self.figure: Optional[plt.Figure] = None
        self.kwargs: dict[str, Any] = {}
        self._segments = np.empty((0, 2, 2), dtype=float)
        self._animator: Optional[AnimateTracks] = None

    def reconstruct(
        self,
        spots: pl.DataFrame,
        *,
        align_at_start: bool = False,
        categories: Optional[dict[str, list[Any]]] = None,
        format: str = 'png',
        **kwargs,
    ):
        self.spot_data = self._ensure_polars(spots) if spots is not None else pl.DataFrame()
        self.align_at_start = align_at_start
        self.categories = categories
        self.kwargs = get_aliases(kwargs, self.ALIASES)
        self._format = str(format or 'png').lower()

        self._arrange_data()

        smoothing = self.kwargs.get('smoothing_index')
        if smoothing is not None:
            self._smooth(smoothing)

        self._assign_color()

        self.figure = self.polar() if self.align_at_start else self.cartesian()

        if self._format == 'html':
            plt.close(self.figure)
            return InteractiveTracks(self)
        return self

    @property
    def fig(self) -> plt.Figure:
        return self.figure

    @property
    def segments(self) -> np.ndarray:
        return self._segments

    def show(self):
        plt.show()
        return self.figure

    def save(self, path, **kwargs):
        path = Path(path)
        if path.suffix.lower() in {'.html', '.htm'}:
            path.write_text(InteractiveTracks(self).to_html(), encoding='utf-8')
        else:
            self.figure.savefig(path, **kwargs)
        return path

    def animate(self, **kwargs):
        self._animator = AnimateTracks(self)
        return self._animator.animate(**kwargs)

    def _repr_html_(self):
        try:
            import base64
            buf = io.BytesIO()
            self.figure.savefig(buf, format='png', bbox_inches='tight')
            data = base64.b64encode(buf.getvalue()).decode('ascii')
            return f'<img src="data:image/png;base64,{data}"/>'
        except Exception:
            return None

    # ---- data arrangement ------------------------------------------------------
    @staticmethod
    def _ensure_polars(df) -> pl.DataFrame:
        """Accept polars DataFrames, loader Input wrappers, or pandas frames."""
        if isinstance(df, pl.DataFrame):
            return df
        if isinstance(getattr(df, 'df', None), pl.DataFrame):
            return df.df
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                return pl.from_pandas(df)
        except ImportError:
            pass
        raise TypeError(f"Expected a polars DataFrame, got {type(df).__name__}.")

    def _categorize(self, df: pl.DataFrame) -> pl.DataFrame:
        for cat, values in self.categories.items():
            if cat not in df.columns:
                raise ColumnsNotFoundError(f"Column '{cat}' not found in DataFrame.")
            df = df.filter(pl.col(cat).is_in(values))
        return df

    def _arrange_data(self):
        if self.categories is not None:
            self.spot_data = self._categorize(self.spot_data)

        if 'track_uid' not in self.spot_data.columns:
            if 'track_id' not in self.spot_data.columns:
                raise ColumnsNotFoundError(
                    "Cannot determine track identity: no 'track_uid' or 'track_id' found.")
            cat_cols = [c for c in self.CATEGORY_COLS if c in self.spot_data.columns]
            key_cols = cat_cols + ['track_id']
            keys = (
                self.spot_data.select(key_cols)
                .unique(maintain_order=True)
                .with_row_index('track_uid')
                .with_columns(pl.col('track_uid').cast(pl.Int64))
            )
            self.spot_data = self.spot_data.join(keys, on=key_cols, how='left')

        self.spot_data = self.spot_data.sort(['track_uid', 'time_point'])
        self._cache_arrays()

    def _cache_arrays(self):
        """Materialize plotting arrays and track run-boundaries once, after sorting."""
        self._x = self.spot_data['x_coordinate'].cast(pl.Float64).to_numpy()
        self._y = self.spot_data['y_coordinate'].cast(pl.Float64).to_numpy()

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
        self._run_lengths = np.diff(np.append(self._track_starts, n))
        self._track_uids = uids[self._track_starts]
        self._within_track_seg_lengths = np.maximum(self._run_lengths - 1, 0)

    def _smooth(self, window: int):
        if not (isinstance(window, int) and window >= 1):
            warnings.warn(
                f"Invalid 'smoothing_index': {window} -> Must be a positive integer "
                "-> No smoothing applied.", category=UserWarning, stacklevel=2)
            return

        self._x = self._smooth_runs(self._x, window)
        self._y = self._smooth_runs(self._y, window)
        self.spot_data = self.spot_data.with_columns(
            pl.Series('x_coordinate', self._x),
            pl.Series('y_coordinate', self._y),
        )

    def _smooth_runs(self, values: np.ndarray, window: int) -> np.ndarray:
        """Rolling-mean smooth each track run, then linearly pin its endpoints."""
        out = values.copy()
        for s, e in zip(self._track_starts, self._track_ends):
            seg = values[s:e + 1]
            m = seg.size
            if m < 2:
                continue
            csum = np.concatenate(([0.0], np.cumsum(seg)))
            idx = np.arange(m)
            lo = np.maximum(0, idx - window + 1)
            smoothed = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
            t = np.linspace(0.0, 1.0, m)
            out[s:e + 1] = smoothed + (
                (seg[0] - smoothed[0]) * (1 - t) + (seg[-1] - smoothed[-1]) * t)
        return out

    # ---- colors ----------------------------------------------------------------
    def _assign_color(self):
        color = self.kwargs.get('color', 'black')
        color_by = self.kwargs.get('color_by')

        # Record how colors were generated so the tooltip offers matching options.
        if color in ('random', 'random greys') and color_by is None:
            self._color_mode, self._color_source = 'per_track', color
        elif color_by is not None:
            self._color_mode = 'lut'
            self._color_source = color_by[0] if isinstance(color_by, tuple) else color_by
        else:
            self._color_mode, self._color_source = 'uniform', color

        if self._color_mode == 'per_track':
            n_tracks = self._track_starts.size
            per_track = np.asarray(paint(pl.DataFrame(), color=color, n=max(n_tracks, 1)))
            self._colors = np.repeat(per_track, self._run_lengths, axis=0)
            self._single_color = None
            return

        paint_input = self.spot_data
        if color_by is not None:
            col = color_by[0] if isinstance(color_by, tuple) else color_by
            paint_input = self.spot_data.select([col])
        c = paint(paint_input, **self.kwargs)

        if isinstance(c, str):
            self._single_color, self._colors = c, None
        else:
            self._single_color, self._colors = None, np.asarray(c)

    # ---- static figures ----------------------------------------------------------
    def cartesian(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(13, 10))
        ax.patch.set_gid('pg:background')
        fig.patch.set_gid('pg:figure')

        self._build_tracks(ax)

        if self._x.size:
            gap = (self._x.max() - self._x.min()) * 0.025
            ax.set_xlim(self._x.min() - gap, self._x.max() + gap)
            ax.set_ylim(self._y.min() - gap, self._y.max() + gap)

        ax.set_aspect('equal', adjustable='box')
        text_color = self.kwargs.get('text_color', 'black')
        ax.set_xlabel('x_coordinate [µm]', color=text_color)
        ax.set_ylabel('y_coordinate [µm]', color=text_color)
        ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=text_color)

        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.tick_params(axis='both', which='major', labelsize=8,
                       colors=self.kwargs.get('annotation_color', 'black'))
        for spine in ax.spines.values():
            spine.set_color(self.kwargs.get('frame_color', 'black'))

        self._finish_axes(fig, ax, polar=False)
        return fig

    def polar(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})
        ax.patch.set_gid('pg:background')
        fig.patch.set_gid('pg:figure')

        text_color = self.kwargs.get('text_color', 'black')
        ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=text_color)
        ax.set_ylim(0, self._max_radius() + 100.0)
        ax.spines['polar'].set_visible(False)

        self._build_tracks(ax, polar=True)
        self._finish_axes(fig, ax, polar=True)
        return fig

    def _finish_axes(self, fig, ax, *, polar: bool):
        if self.kwargs.get('grid', True):
            self._style_grid(ax)
        else:
            ax.grid(False)
        if self.kwargs.get('show_heads', True):
            self._head_markers(ax, polar=polar)
        if self.kwargs.get('hide_backdrop', False):
            fig.set_facecolor('none')

    def _frame_axes(self, ax):
        """Minimal axes framing reused by the animator."""
        if self.kwargs.get('grid', True):
            self._style_grid(ax)

    def _max_radius(self) -> float:
        if self._x.size == 0:
            return 0.0
        x0 = np.repeat(self._x[self._track_starts], self._run_lengths)
        y0 = np.repeat(self._y[self._track_starts], self._run_lengths)
        return float(np.hypot(self._x - x0, self._y - y0).max())

    # ---- geometry / drawing --------------------------------------------------------
    def _plot_coords(self, *, polar: bool = False):
        """Return per-spot plotting coordinates (polar-transformed if requested)."""
        x, y = self._x, self._y
        if not polar:
            return x, y
        x0 = np.repeat(x[self._track_starts], self._run_lengths)
        y0 = np.repeat(y[self._track_starts], self._run_lengths)
        dx, dy = x - x0, y - y0
        return np.arctan2(dy, dx), np.hypot(dx, dy)

    def _build_tracks(self, ax: plt.Axes, *, polar: bool = False):
        """Plot all track segments as one gid-tagged LineCollection."""
        if self._x.size < 2:
            self._segments = np.empty((0, 2, 2), dtype=float)
            self._seg_mask = np.empty(0, dtype=bool)
            return

        plot_x, plot_y = self._plot_coords(polar=polar)
        segs = np.stack(
            (np.column_stack((plot_x[:-1], plot_y[:-1])),
             np.column_stack((plot_x[1:], plot_y[1:]))),
            axis=1,
        )
        self._seg_mask = self._same_track
        self._segments = segs[self._same_track]

        lc = LineCollection(
            self._segments,
            colors=self._segment_color_arg(self._same_track),
            linewidths=self.kwargs.get('lw', 1),
        )
        lc.set_gid('pg:tracks')
        ax.add_collection(lc)

    def _segment_color_arg(self, same: np.ndarray):
        if self._single_color is not None:
            return self._single_color
        if self._colors is None:
            return 'black'
        seg_colors = self._colors[:-1][same]
        if seg_colors.size and self._all_same_color(seg_colors):
            return seg_colors[0]
        return seg_colors

    def _segment_colors_for_mask(self, mask: np.ndarray):
        """Colors for the within-track segments selected by `mask` (animation)."""
        if self._single_color is not None:
            return self._single_color
        if self._colors is None:
            return 'black'
        return self._colors[:-1][self._seg_mask][mask]

    @staticmethod
    def _all_same_color(colors: np.ndarray) -> bool:
        first = colors[0]
        if colors.dtype != object:
            return bool((colors == first).all())
        return all(np.array_equal(c, first) for c in colors)

    def _head_markers(self, ax: plt.Axes, *, polar: bool = False):
        """Draw one gid-tagged marker at the last spot of each track."""
        ends = self._track_ends
        if ends.size == 0:
            return

        if self._single_color is not None:
            colors = self._single_color
        elif self._colors is not None:
            colors = self._colors[ends]
        else:
            colors = 'black'

        if polar:
            dx = self._x[ends] - self._x[self._track_starts]
            dy = self._y[ends] - self._y[self._track_starts]
            px, py = np.arctan2(dy, dx), np.hypot(dx, dy)
        else:
            px, py = self._x[ends], self._y[ends]

        outline = self.kwargs.get('outline_head', True)
        fill = self.kwargs.get('fill_head', False)
        scatter = ax.scatter(
            px, py,
            marker=self.kwargs.get('head_shape', 'o'),
            s=self.kwargs.get('head_size', 10),
            edgecolor=colors if outline else 'none',
            facecolor=colors if fill else 'none',
            linewidths=self.kwargs.get('head_outline_width', 1.0),
            zorder=12,
        )
        scatter.set_gid('pg:heads')

        if self.kwargs.get('show_ids', False):
            self._annotate_uids(ax, px, py, colors)

    def _annotate_uids(self, ax: plt.Axes, px, py, colors):
        label_colors = [colors] * len(self._track_uids) if isinstance(colors, str) else list(colors)
        for x, y, uid, col in zip(px, py, self._track_uids, label_colors):
            ax.annotate(
                str(uid), xy=(x, y),
                xytext=(self.kwargs.get('id_offset', 2),) * 2,
                textcoords='offset points',
                fontsize=self.kwargs.get('id_fontsize', 9),
                color=col, clip_on=True, zorder=14,
            )

    def _head_colors(self, has_pt: np.ndarray):
        """Per-track head colors, aligned to the tracks kept by `has_pt`."""
        if self._single_color is not None:
            return self._single_color
        if self._colors is None or self._track_ends.size == 0:
            return 'black'
        return np.asarray(self._colors)[self._track_ends][has_pt]

    # ---- styling -------------------------------------------------------------
    def _style_grid(self, ax: plt.Axes):
        grid_color = self.kwargs.get('grid_color', 'gainsboro')
        grid_lw = self.kwargs.get('grid_lw', 0.75)
        grid_ls = self.kwargs.get('grid_ls', '-')

        if not self.align_at_start:
            ax.grid(True, which='both', axis='both',
                    color=grid_color, linestyle=grid_ls, linewidth=grid_lw)
        else:
            ax.grid(True, lw=grid_lw, color=grid_color)

            get = lambda key, fallback: self.kwargs.get(key) or fallback
            for i, line in enumerate(ax.get_xgridlines()):
                kind = 'cardinal' if i % 2 == 0 else 'diagonal'
                line.set_linestyle(get(f'grid_ls_{kind}', grid_ls))
                line.set_color(get(f'grid_color_{kind}', grid_color))
            for line in ax.get_ygridlines():
                line.set_linestyle(get('grid_ls_radial', grid_ls))
                line.set_color(get('grid_color_radial', grid_color))

        for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            line.set_gid('pg:grid')

    @staticmethod
    def _html_color(color: Any) -> str:
        try:
            return to_hex(color, keep_alpha=True)
        except (TypeError, ValueError):
            return '#000000ff'


class InteractiveTracks:
    """
    Wraps the actual Matplotlib figure, exported as SVG markup with tagged
    (gid) element groups. The JS tooltip attaches to those groups and edits
    SVG attributes in place — nothing is re-rendered client-side.
    """

    def __init__(self, builder: ReconstructTracks):
        self._builder = builder

    @property
    def fig(self) -> plt.Figure:
        return self._builder.figure

    def animate(self, **kwargs):
        return self._builder.animate(**kwargs)

    def _figure_svg(self) -> str:
        buf = io.StringIO()
        self._builder.figure.savefig(buf, format='svg', bbox_inches='tight')
        svg = buf.getvalue()
        return svg[svg.find('<svg'):]  # strip XML prolog for inlining

    def _style_payload(self) -> dict[str, Any]:
        k = self._builder.kwargs
        hex_ = ReconstructTracks._html_color
        return {
            'lw': float(k.get('lw', 1.0)),
            'headSize': 1.0,
            'trackColor': hex_(k.get('color', 'black')),
            'faceColor': hex_(k.get('face_color', 'white')),
            'gridColor': hex_(k.get('grid_color', 'gainsboro')),
            'gridLw': float(k.get('grid_lw', 0.75)),
        }

    def to_html(self) -> str:
        js = tooltip_assets['js'].replace(
            '"__PEREGRIN_CSS__"', json.dumps(tooltip_assets['css']))
        payload = json.dumps({'style': self._style_payload()})
        return (
            f'<div class="peregrin-graph">{self._figure_svg()}</div>\n'
            f'<script>window.GRAPH_STATE = {payload};</script>\n'
            f'<script>{js}</script>'
        )

    def _repr_html_(self):
        return self.to_html()

    def save(self, path, **kwargs):
        path = Path(path)
        path.write_text(self.to_html(), encoding='utf-8')
        return path


reconstruct = ReconstructTracks().reconstruct