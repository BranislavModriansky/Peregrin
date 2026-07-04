from inspect import Arguments

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from PIL import Image

import warnings
from ...settings import params
from ..._pckg_exceptions._pckg_errors import *
from ..._pckg_exceptions._pckg_warnings import *

from ..categorizer import categorize
from ..painter import retrieve_palette, retrieve_lut

from ...various import Values, get_aliases
from io import BytesIO
from ..._infra._selections import Metrics
from ...compute.stats import Stats

class ReconstructTracks:

    ALIASES = {
        "smoothing_index": ["smoothing_index", "smooth_index", "smooth_window", "smooth_window_size"],
    }

    KEY_COLS = ['condition', 'replicate', 'track_id']
    
    def __init__(self): ...

    def reconstruct(
        self, 
        Spots_df: pd.DataFrame, 
        Tracks_df: pd.DataFrame = None, 
        *,
        conditions: list = [],
        replicates: list = [],
        align_origin: bool = False,
        **kwargs
    ) -> plt.Figure:
        
        self.Spots_df = Spots_df
        self.Tracks_df = Tracks_df
        self.conditions = conditions
        self.replicates = replicates
        self.align_origin = align_origin
        self.kwargs = get_aliases(kwargs, self.ALIASES)

        self._arrange_data()

        if not align_origin:
            return self.realistic()
        else:
            return self.anchored()

    def realistic(self) -> plt.Figure:

        self._arrange_data()    
        if self.kwargs.get('smoothing_index', None) is not None and self.kwargs.get('smoothing_index', None) > 0:
            self._smooth()
        self._assign_colors()
        self._color_tracks(polar=False)

        fig, ax = plt.subplots(figsize=(13, 10))
        if len(self.Spots):
            x = self.Spots['X coordinate'].to_numpy()
            y = self.Spots['Y coordinate'].to_numpy()
            ax.set_xlim(np.nanmin(x), np.nanmax(x))
            ax.set_ylim(np.nanmin(y), np.nanmax(y))

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X coordinate [µm]', color=self.text_color)
        ax.set_ylabel('Y coordinate [µm]', color=self.text_color)
        ax.set_title(self.title, fontsize=12, color=self.text_color)
        self._background_color(); ax.set_facecolor(self.face_color)

        # Ticks
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.tick_params(axis='both', which='major', labelsize=8, colors=self.text_color)

        if self.grid:
            self._grid_color(coord_system='cartesian')
            self._grid_style(coord_system='cartesian', ax=ax)
        else:
            ax.grid(False)
        
        self._color_segments(ax)
        
        if self.mark_heads:
            self._head_markers(ax, polar=False)

        if self.strip_backdrop:
            fig.set_facecolor('none')

        return plt.gcf()
    

    def Normalized(self, all_data: pd.DataFrame) -> plt.Figure:

        self._arrange_data()
        if self.kwargs.get('smoothing_index', None) is not None and self.kwargs.get('smoothing_index', None) > 0:
            self._smooth()
        self._assign_colors()
        self._convert_polar()
        self._get_radius(all_data)
        self._color_tracks(polar=True)

        fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})
        ax.set_title(self.title, fontsize=12, color=self.text_color)
        ax.set_ylim(0, self.y_max_global)        # <- global, consistent across subsets
        
        ax.spines['polar'].set_visible(False)

        self._background_color(); ax.set_facecolor(self.face_color)
        if self.grid:
            self._grid_color(coord_system='polar')
            self._grid_style(coord_system='polar', ax=ax)
        else:
            ax.grid(False)

        self._annotate_r_axis(ax)
        self._annotate_theta_axis(ax)

        self._color_segments(ax)

        if self.mark_heads:
            self._head_markers(ax, polar=True)

        if self.strip_backdrop:
            fig.set_facecolor('none')

        return plt.gcf()


    def ImageStack(
        self,
        frames_mode: str = "cumulative",  # 'cumulative' | 'per_frame'
        dpi: int = 100,
        size: tuple[int, int] = (975, 750),
    ) -> np.ndarray | None:
        """
        Build a stack of Cartesian frames (Realistic style), returning uint8 RGBA
        of shape (N, H, W, 4). Uses the same pipeline as Realistic():
        - category filtering via conditions/replicates
        - optional smoothing
        - color assignment (including LUT min/max, palettes)
        - background and grid settings
        - optional head markers per frame.
        """
        self._arrange_data()
        if self.kwargs.get('smoothing_index', None) is not None and self.kwargs.get('smoothing_index', None) > 0:
            self._smooth()
        self._assign_colors()

        Spots = self.Spots.copy()

        required = ["Time point", "X coordinate", "Y coordinate"]
        missing = [c for c in required if c not in Spots.columns]
        if missing:
            Reporter(Level.error, f"Cannot build animated track reconstruction. -> Missing required columns in Spots_df: {missing}.", noticequeue=self.noticequeue)
            return None

        if Spots.empty:
            return None

        # Global axes limits (fixed over time)
        x_all = Spots["X coordinate"].to_numpy(dtype=float, copy=False)
        y_all = Spots["Y coordinate"].to_numpy(dtype=float, copy=False)
        xlim = (np.nanmin(x_all), np.nanmax(x_all))
        ylim = (np.nanmin(y_all), np.nanmax(y_all))

        # Time points (sorted)
        time_points = np.unique(Spots["Time point"].to_numpy())
        time_points.sort()

        # Optional: clamp to number of frames if 'Frame' column is present
        if "Frame" in Spots.columns:
            frames = np.unique(Spots["Frame"].to_numpy())
            frames_size = frames.size
            if frames_size and len(time_points) > frames_size:
                time_points = time_points[:frames_size]

        W, H = size
        fig_w = W / float(dpi)
        fig_h = H / float(dpi)

        image_stack: list[np.ndarray] = []

        for t in time_points:
            if frames_mode == "per_frame":
                Spots_t = Spots.loc[Spots["Time point"] == t]
            else:  # cumulative
                Spots_t = Spots.loc[Spots["Time point"] <= t]

            if Spots_t.empty:
                continue

            segments, seg_colors = self._build_segments(Spots_t, polar=False)
            if not segments:
                continue

            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(f"X coordinate [µm]", color=self.text_color)
            ax.set_ylabel(f"Y coordinate [µm]", color=self.text_color)

            if self.title:
                ax.set_title(f"{self.title} | Time: {t} {Stats.t_unit}", fontsize=12, text_color=self.text_color)
            else:
                ax.set_title(f"Time: {t} {Stats.t_unit}", fontsize=12, color=self.text_color)

            self._background_color()
            ax.set_facecolor(self.face_color)

            if self.grid:
                self._grid_color(coord_system="cartesian")
                self._grid_style(coord_system="cartesian", ax=ax)
            else:
                ax.grid(False)

            # Ticks identical to Realistic
            ax.xaxis.set_major_locator(MultipleLocator(200))
            ax.yaxis.set_major_locator(MultipleLocator(200))
            ax.xaxis.set_minor_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(50))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.tick_params(axis="both", which="major", labelsize=8, colors=self.text_color)

            lc = LineCollection(segments, colors=seg_colors, linewidths=self.lw, zorder=10)
            ax.add_collection(lc)

            if self.mark_heads:
                self._head_markers(ax, polar=False, spots=Spots_t)

            if self.strip_backdrop:
                fig.set_facecolor('none')

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            im = Image.open(buf).convert("RGBA")
            image_stack.append(np.asarray(im, dtype=np.uint8))

        if not image_stack:
            return None

        return np.stack(image_stack, axis=0)

    
    def SaveAnimation(
        self,
        stack:  np.ndarray,
        path: str,
        fps: int = 30,
        codec: str = "libx264",
        crf: int | None = 18,
        pix_fmt: str = "yuv420p",
        background: tuple[int, int, int] = (255, 255, 255),
        bitrate: str | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Prepare an image stack for MP4 export.

        Returns
        -------
        rgb       : uint8 array (N, H, W, 3)
        out_params: list[str] ffmpeg parameters to pass to imageio / ffmpeg
        """

        if stack.ndim not in (3, 4):
            raise ValueError("stack must have shape (N,H,W) or (N,H,W,C)")
        if stack.ndim == 3:
            stack = stack[..., None]  # (N,H,W,1)

        N, H, W, C = stack.shape
        if C not in (1, 3, 4):
            raise ValueError("last dimension must be 1, 3, or 4 channels")

        # Ensure uint8
        if stack.dtype != np.uint8:
            stack = np.clip(stack, 0, 255).astype(np.uint8)

        # Expand to RGB
        if C == 1:
            rgb = np.repeat(stack, 3, axis=-1)
        elif C == 3:
            rgb = stack
        else:
            # RGBA -> RGB over background
            rgb = self._rgba_over_background(stack, background)

        # Build ffmpeg output params
        out_params: list[str] = ["-pix_fmt", pix_fmt]
        if crf is not None:
            out_params += ["-crf", str(crf)]
        if bitrate is not None:
            out_params += ["-b:v", str(bitrate)]

        # 'path' is kept for API compatibility; caller writes the file via imageio/ffmpeg.
        return rgb, out_params


    def GetLutMap(self, units: dict | None = None, _extend: bool = True) -> plt.Figure | None:
        """
        Create a color guide for the current color mode:

        - For continuous LUT modes: a scalar colorbar (same scaling as tracks).
        - For 'differentiate replicates' / 'differentiate conditions':
          a qualitative legend mapping category -> color.

        Parameters
        ----------
        units : dict, optional
            Mapping from metric name (e.g. 'Net distance') to unit string
            (e.g. 'μm'). For 'Speed instantaneous', the key should be the
            original name (e.g. 'Speed instantaneous'), not 'Distance'.
        _extend : bool, optional
            Whether to extend the colorbar at both ends (continuous modes).

        Returns
        -------
        matplotlib.figure.Figure or None
            The created figure, or None if no guide is appropriate.
        """
        units = units or {}

        # Ensure data and colors are ready
        if self.Spots is None or self.Tracks is None:
            self._arrange_data()
        if 'Track color' not in self.Tracks.columns and 'Spot color' not in self.Spots.columns:
            self._assign_colors()

        # Modes where a LUT / legend is not meaningful
        if self.c_mode in ['random colors', 'random greys', 'single color']:
            return None

        # Qualitative legend for categorical modes
        if self.c_mode in ['differentiate replicates', 'differentiate conditions']:
            if self.c_mode == 'differentiate replicates':
                category_col = 'Replicate'
            else:
                category_col = 'Condition'

            df = self.Tracks.reset_index()

            if category_col not in df.columns or 'Track color' not in df.columns:
                return None

            # Preserve first-seen order of categories
            seen: dict = {}
            for _, row in df[[category_col, 'Track color']].dropna().iterrows():
                key = row[category_col]
                if key not in seen:
                    seen[key] = row['Track color']

            if not seen:
                return None

            labels = list(seen.keys())
            colors = list(seen.values())

            fig, ax = plt.subplots(figsize=(2.5, 0.4 * len(labels) + 0.6))
            ax.axis('off')

            handles = [
                plt.Line2D([0], [0], color=c, lw=4)
                for c in colors
            ]
            ax.legend(
                handles,
                labels,
                loc='center left',
                frameon=False,
            )
            fig.tight_layout()
            return fig

        # Continuous LUT colorbar for other colormap modes
        norm, vals = Values.LutMapper(self.Tracks if self.lut_scaling_stat in self.Tracks.columns else self.Spots, self.lut_scaling_stat, min=self.lut_vmin, max=self.lut_vmax, noticequeue=self.noticequeue)
        if norm is None:
            return None

        colormap = self.painter.GetCmap(self.c_mode)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])

        fig_lut, ax_lut = plt.subplots(figsize=(2, 6))
        ax_lut.axis('off')
        cbar = fig_lut.colorbar(
            sm,
            ax=ax_lut,
            orientation='vertical',
            extend='both' if _extend else 'neither',
        )

        # Use the user-visible metric name for labeling (e.g. 'Speed instantaneous')
        label_metric = self.lut_scaling_stat
        unit = units.get(label_metric, "")
        if unit:
            cbar.set_label(f"{label_metric} {unit}", fontsize=10)
        else:
            cbar.set_label(f"{label_metric}", fontsize=10)

        return fig_lut


    @staticmethod
    def frame_interval_ms(fps: float) -> float:
        """Return the interval between frames in milliseconds."""
        if fps <= 0:
            raise ValueError("Frame rate must be positive.")
        return 1000.0 / fps
        

    def _arrange_data(self):
        Spots =  categorize(self.Spots_df, self.conditions, self.replicates)
        Tracks = categorize(self.Tracks_df, self.conditions, self.replicates)

        if not params.ignore_categories:
            self.Spots = Spots.sort_values(['condition', 'replicate', 'track_id', 'time_point'])
            self.Tracks = Tracks.sort_values(['condition', 'replicate', 'track_id'])
        else:
            self.Spots = Spots.sort_values(['track_uid', 'time_point'])
            self.Tracks = Tracks.sort_values(['track_uid'])


        if list(Spots.index.names) != self.KEY_COLS:
            Spots = Spots.set_index(self.KEY_COLS)
        if list(Tracks.index.names) != self.KEY_COLS:
            Tracks = Tracks.set_index(self.KEY_COLS)

        self.Spots = Spots
        self.Tracks = Tracks

    def _convert_polar(self):
        self.Spots['X coordinate'] = self.Spots['X coordinate'] - self.Spots.groupby(level=self.KEY_COLS)['X coordinate'].transform('first')
        self.Spots['Y coordinate'] = self.Spots['Y coordinate'] - self.Spots.groupby(level=self.KEY_COLS)['Y coordinate'].transform('first')

        self.Spots['r'] = np.sqrt(self.Spots['X coordinate']**2 + self.Spots['Y coordinate']**2)
        self.Spots['theta'] = np.arctan2(self.Spots['Y coordinate'], self.Spots['X coordinate'])

    def _get_radius(self, alldata: pd.DataFrame):
        """
        Compute global maximum radius from (X, Y) positions.

        Works with both:
        - MultiIndex with levels ['Condition', 'Replicate', 'Track ID']
        - Regular Index with those columns present.
        """
        if alldata.empty:
            self.y_max_global = 100.0
            self.y_max_label_global = "100 μm"
            return

        # Choose grouping mode based on index type
        if isinstance(alldata.index, pd.MultiIndex) and list(alldata.index.names) == self.KEY_COLS:
            group = alldata.groupby(level=self.KEY_COLS)
        else:
            # Fall back to grouping by columns
            missing = [c for c in self.KEY_COLS if c not in alldata.columns]
            if missing:
                Reporter(Level.error, f"Cannot compute radius for polar plot. -> Missing required grouping columns: {missing}.", noticequeue=self.noticequeue)

                self.y_max_global = 100.0
                self.y_max_label_global = "100 μm"
                return
            group = alldata.groupby(self.KEY_COLS)

        x = alldata['X coordinate']
        y = alldata['Y coordinate']
        x0 = group['X coordinate'].transform('first')
        y0 = group['Y coordinate'].transform('first')
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

        self.y_max_global = r.max() + 10.0
        self.y_max_label_global = f"{round(r.max()) + 10} μm"

        if not np.isfinite(self.y_max_global):
            Reporter(Level.warning, f"Invalid maximum radius computed for polar plot. Setting to default '100 μm'.", details=f"Maximum radius was not finite: {self.y_max_global}.", noticequeue=self.noticequeue)
            self.y_max_global = 100.0
        if not (self.y_max_global > 0):
            Reporter(Level.warning, 'Negative maximum radius. Setting to default "100 μm".', details=f"Maximum radius was a negative value: {self.y_max_global}.", noticequeue=self.noticequeue)
            self.y_max_global = 100.0
            self.y_max_label_global = "100 μm"
        
    def _smooth(self):
        if (isinstance(self.smoothing_index, (int, float))) and self.smoothing_index >= 1:
            _smoothing_window = self.smoothing_index
            if isinstance(_smoothing_window, float):
                _smoothing_window = round(self.smoothing_index)
                
                Reporter(Level.info, f"Smoothing index rounded to nearest integer: {_smoothing_window}.", f"Invalid smoothing index type: float ({self.smoothing_index}).", noticequeue=self.noticequeue)
            
            for col in ['X coordinate', 'Y coordinate']:
                self.Spots[col] = (
                    self.Spots.groupby(
                        level=self.KEY_COLS
                    )[col].transform(
                        lambda s: self._smooth_preserve_endpoints(s, _smoothing_window)
                    )
                )
        else:
            Reporter(Level.warning, f"Invalid smoothing index. No smoothing applied.", f"Smoothing index must be an integer type and must be greater than 1. {type(self.smoothing_index)}: {self.smoothing_index}.", noticequeue=self.noticequeue)

    @staticmethod
    def _smooth_preserve_endpoints(s: pd.Series, window: int) -> pd.Series:
        """Smooth a series with rolling mean, then linearly correct so endpoints are preserved."""
        if len(s) < 2:
            return s
        original_start = s.iloc[0]
        original_end = s.iloc[-1]
        smoothed = s.rolling(window, min_periods=1).mean()
        smoothed_start = smoothed.iloc[0]
        smoothed_end = smoothed.iloc[-1]
        n = len(smoothed)
        # Linear correction: ramp from (original_start - smoothed_start) to (original_end - smoothed_end)
        t = np.linspace(0, 1, n)
        correction = (original_start - smoothed_start) * (1 - t) + (original_end - smoothed_end) * t
        return smoothed + correction

    def _assign_colors(self):
        rng = np.random.default_rng(42)
        track_index = self.Tracks.index.unique()

        if self.c_mode == 'single color':
            self.Tracks['Track color'] = self._coerce_color(self.only_one_color)

        elif self.c_mode in ['random colors', 'random greys']:
            if self.c_mode == 'random colors':
                cols = [mcolors.to_hex(rng.random(3)) for _ in range(len(track_index))]
            else:
                cols = [mcolors.to_hex((float(g), float(g), float(g))) for g in rng.random(len(track_index))]
            lut = dict(zip(track_index, cols))
            self.Tracks['Track color'] = [self._coerce_color(lut.get(idx)) for idx in self.Tracks.index]

        elif self.c_mode in ['differentiate conditions', 'differentiate replicates']:
            category = 'Replicate' if self.c_mode == 'differentiate replicates' else 'Condition'
            df = self.Tracks.reset_index()
            categories = df[category].dropna().unique().tolist()

            if self.stock_palette is None:
                mp = self.painter.BuildQualPalette(data=df, tag=category, which=categories)
            else:
                mp = self.painter.StockQualPalette(
                    data=df,
                    tag=category,
                    palette=self.stock_palette,
                    which=categories
                )
                if not isinstance(mp, dict) or not mp:
                    mp = self.painter.BuildQualPalette(data=df, tag=category, which=categories)

            df['Track color'] = df[category].map(mp).apply(self._coerce_color)
            self.Tracks = df.set_index(['Condition', 'Replicate', 'Track ID'])

        else:
            cmap = self._resolve_cmap(self.c_mode)
            norm, vals = Values.LutMapper(
                self.Tracks if self.lut_scaling_stat in self.Tracks.columns else self.Spots,
                self.lut_scaling_stat,
                min=self.lut_vmin,
                max=self.lut_vmax,
                noticequeue=self.noticequeue
            )

            if not (norm is None or vals is None):
                if self.lut_scaling_stat in self.Tracks.columns:
                    self.Tracks['Track color'] = [self._coerce_color(mcolors.to_hex(cmap(norm(v)))) for v in vals]
                if self.lut_scaling_stat in self.Spots.columns:
                    self.Spots['Spot color'] = [self._coerce_color(mcolors.to_hex(cmap(norm(v)))) for v in vals]
            else:
                self.Tracks['Track color'] = mcolors.to_hex('black')

        if 'Track color' in self.Tracks.columns and 'Track color' not in self.Spots.columns:
            self.Spots = self.Spots.join(self.Tracks[['Track color']], how='left')
            self.Spots['Track color'] = self.Spots['Track color'].apply(self._coerce_color)

    def _build_segments(self, spots: pd.DataFrame, polar: bool = False):
        """
        Build segments and colors from a Spots-like dataframe.

        Returns
        -------
        segments : list of (N_i, 2) float arrays
        colors   : list of hex strings, one per segment
        """
        coord_cols = ('theta', 'r') if polar else ('X coordinate', 'Y coordinate')

        segments: list[np.ndarray] = []
        seg_colors: list[str] = []

        if 'Spot color' in spots.columns:
            # Per-spot coloring (e.g. instantaneous LUTs)
            for _, g in spots.groupby(level=self.KEY_COLS, sort=False):
                coords = g[list(coord_cols)].to_numpy(dtype=float, copy=False)
                cols = g['Spot color'].to_numpy()
                n = coords.shape[0]
                if n >= 2:
                    for i in range(1, n):
                        segments.append(coords[i - 1:i + 1])
                        seg_colors.append(self._coerce_color(cols[i]))
        elif 'Track color' in spots.columns:
            # Per-track coloring
            for _, g in spots.groupby(level=self.KEY_COLS, sort=False):
                coords = g[list(coord_cols)].to_numpy(dtype=float, copy=False)
                if coords.shape[0] >= 2:
                    segments.append(coords)
                    seg_colors.append(self._coerce_color(g['Track color'].iloc[0]))
        else:
            # Fallback: black
            default_col = mcolors.to_hex('black')
            for _, g in spots.groupby(level=self.KEY_COLS, sort=False):
                coords = g[list(coord_cols)].to_numpy(dtype=float, copy=False)
                if coords.shape[0] >= 2:
                    segments.append(coords)
                    seg_colors.append(default_col)

        return segments, seg_colors

    def _color_tracks(self, polar: bool = False):
        """
        Populate self.segments and self.segment_colors from self.Spots/self.Tracks.
        """
        self.segments, self.segment_colors = self._build_segments(self.Spots, polar=polar)

    def _background_color(self):
        mapping = {
            'white': 'white',
            'light': 'lightgrey',
            'mid': 'darkgrey',
            'dark': 'dimgrey',
            'black': 'black',
        }
        self.face_color = mapping.get(self.background, 'white')

    def _grid_color(self, coord_system: str = 'cartesian'):
        mapping = {
            'cartesian': {
                'white':    ('gainsboro', 0.5),
                'light':    ('silver', 0.5),
                'mid':      ('silver', 0.5),
                'dark':     ('grey', 0.5),
                'black':    ('dimgrey', 0.5),
            }, 
            'polar': {
                'white':    ('lightgrey', 0.7, 0.8),
                'light':    ('darkgrey', 0.7, 0.6),
                'mid':      ('dimgrey', 0.7, 0.5),
                'dark':     ('grey', 0.7, 0.6),
                'black':    ('dimgrey', 0.5, 0.4),
            }
        }

        if coord_system == 'cartesian':
            self.grid_color, self.grid_alpha = mapping['cartesian'].get(self.background, ('gainsboro', 0.5))

        elif coord_system == 'polar':
            self.grid_color, self.grid_a_alpha, self.grid_alpha = mapping['polar'].get(self.background, ('lightgrey', 0.7, 0.8))

    def _grid_style(self, ax: plt.Axes, coord_system: str = 'cartesian'):

        if coord_system == 'cartesian':
            self.grid_ls = '-.'
            ax.grid(True, which='both', axis='both', color=self.grid_color, linestyle=self.grid_ls, linewidth=1, alpha=self.grid_alpha)
        
        elif coord_system == 'polar':
            match self.gridstyle:
                case 'simple-1' | 'simple-2':
                    ax.xaxis.grid(True, color=self.grid_color, linestyle='-', linewidth=1, alpha=self.grid_a_alpha)
                    ax.yaxis.grid(False)

                    if self.gridstyle == 'simple-1':
                        for i, line in enumerate(ax.get_xgridlines()):
                            if i % 2 != 0:
                                line.set_color('none')
                    if self.gridstyle == 'simple-2':
                        for i, line in enumerate(ax.get_xgridlines()):
                            if i % 2 == 0:
                                line.set_color('none')

                case 'dartboard-1' | 'dartboard-2':
                    ax.grid(True, lw=0.75, color=self.grid_color, alpha=self.grid_a_alpha)
                    if self.gridstyle == 'dartboard-1':
                        for i, line in enumerate(ax.get_xgridlines()):
                            if i % 2 == 0:
                                line.set_linestyle('-.'); line.set_color(self.grid_color); line.set_linewidth(0.75), line.set_alpha(self.grid_a_alpha)
                        for line in ax.get_ygridlines():
                            line.set_linestyle('--'); line.set_color(self.grid_color); line.set_linewidth(0.75), line.set_alpha(self.grid_a_alpha)

                    if self.gridstyle == 'dartboard-2':
                        for i, line in enumerate(ax.get_xgridlines()):
                            if i % 2 != 0:
                                line.set_linestyle('-.'); line.set_color(self.grid_color); line.set_linewidth(0.75), line.set_alpha(self.grid_a_alpha)
                        for line in ax.get_ygridlines():
                            line.set_linestyle('--'); line.set_color(self.grid_color); line.set_linewidth(0.75), line.set_alpha(self.grid_a_alpha)
                case 'spindle':
                    ax.xaxis.grid(True, color=self.grid_color, linestyle='-', linewidth=1, alpha=self.grid_a_alpha)
                    ax.yaxis.grid(False)
                case 'radial':
                    ax.xaxis.grid(False)
                    ax.yaxis.grid(True, color=self.grid_color, linestyle='-', linewidth=1, alpha=self.grid_a_alpha)


    def _annotate_r_axis(self, ax: plt.Axes):
        match self.annotate_r:
            case 'minimal':
                ax.set_yticklabels([])

                # Scale indicator
                ax.scatter(0, self.y_max_global + 35, color=self.grid_color, marker='.', s=5, clip_on=False)
                ax.text(0, self.y_max_global + 50, self.y_max_label_global, va='center',
                        fontsize=10, color=self.grid_color, clip_on=False)

            case 'detailed':
                rlabels = ax.get_yticklabels()
                ax.set_yticklabels(rlabels, fontsize=10, color=self.grid_color)

            case _:
                ax.set_yticklabels([])

    
    def _annotate_theta_axis(self, ax: plt.Axes):
        if self.annotate_theta:
            tlabels = ax.get_xticklabels()
            ax.set_xticklabels(tlabels, fontsize=10, color=self.text_color)
        else:
            ax.set_xticklabels([])


    def _head_markers(self, ax, polar: bool = False, spots: pd.DataFrame | None = None):
        """
        Draw markers at track ends.

        polar=False -> use Cartesian coordinates (X/Y).
        polar=True  -> use polar coordinates (theta/r).

        If `spots` is provided, use that subset; otherwise use self.Spots.
        """
        spots_df = self.Spots if spots is None else spots

        x_coord, y_coord = ('theta', 'r') if polar else ('X coordinate', 'Y coordinate')

        ends = spots_df.groupby(level=self.KEY_COLS, sort=False).tail(1)
        if len(ends):
            xe = ends[x_coord].to_numpy(dtype=float, copy=False)
            ye = ends[y_coord].to_numpy(dtype=float, copy=False)

            cols = np.array([mcolors.to_hex('black')] * len(ends), dtype=object)
            if 'Spot color' in ends.columns:
                cols = np.array([self._coerce_color(c) for c in ends['Spot color'].to_numpy()], dtype=object)
            elif 'Track color' in ends.columns:
                cols = np.array([self._coerce_color(c) for c in ends['Track color'].to_numpy()], dtype=object)

            m = np.isfinite(xe) & np.isfinite(ye)
            if m.any():
                ax.scatter(
                    xe[m], ye[m],
                    marker=self.marker["symbol"],
                    s=self.marker_size,
                    edgecolor=cols[m],
                    facecolor=cols[m] if self.marker["fill"] else "none",
                    linewidths=self.lw,
                    zorder=12,
                )

    def _color_segments(self, ax):
        if not self.segments:
            return
        lc = LineCollection(self.segments, colors=self.segment_colors, linewidths=self.lw, zorder=10)
        ax.add_collection(lc)

    def _resolve_cmap(self, cmap_like):
        if isinstance(cmap_like, mcolors.Colormap):
            return cmap_like
        name = str(cmap_like).strip()
        if name.lower().endswith('lut'):
            name = name[:-3].strip()
        try:
            cmap = self.painter.GetCmap(name)
            if isinstance(cmap, mcolors.Colormap):
                return cmap
        except Exception:
            pass
        try:
            return plt.get_cmap(name)
        except Exception:
            return plt.get_cmap('viridis')

    def _coerce_color(self, value, fallback: str = '#000000') -> str:
        try:
            if pd.isna(value):
                return fallback
        except Exception:
            pass
        try:
            return mcolors.to_hex(value)
        except Exception:
            return fallback




reconstruct = ReconstructTracks().reconstruct