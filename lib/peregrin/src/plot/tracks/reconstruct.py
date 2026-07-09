from inspect import Arguments

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
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
from ..painter import retrieve_palette, retrieve_cmap, random_color, random_grey, cmap_lut, dyes, is_color_code

from ...various import Values, get_aliases, is_empty, clock
from io import BytesIO
from ..._infra._selections import Metrics
from ...compute.stats import Stats

import matplotlib

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

    _C_MODES = ['None', 'random', 'random_greys', 'categorical', 'numeric']
    _DYE_COLOR_SET = frozenset(dyes.colors.values())

    KEY_COLS = ['condition', 'replicate', 'track_id']
    REQUIRED_COLS = ['condition', 'replicate', 'track_id', 'track_uid' , 'time_point', 'x_coordinate', 'y_coordinate']
    
    def __init__(self): ...

    @clock
    def reconstruct(
        self, 
        spot_data: pd.DataFrame,
        track_data: pd.DataFrame = None,
        *,
        conditions: list = [],
        replicates: list = [],
        common_start: bool = False,
        ignore_categories: bool = False,
        **kwargs
    ) -> plt.Figure:
        
        self.spot_data = spot_data
        self.track_data = track_data
        self.conditions = conditions
        self.replicates = replicates
        self.common_start = common_start
        self.kwargs = get_aliases(kwargs, self.ALIASES)

        if params.ignore_categories or ignore_categories:
            self.REQUIRED_COLS.remove('condition', ' replicate')
        if ignore_categories and not params.ignore_categories:
            warnings.warn(message=f"ignore_categories is set to True for this function call, but the global setting is False. This function call will not ignore categories. To change the global setting, use `peregrin.settings(ignore_categories=True)`.", 
                          category=UserWarning, 
                          stacklevel=2)

        self._arrange_data()
        self._assign_color()

        if self.kwargs.get('smoothing_index', None) is not None and self.kwargs.get('smoothing_index', None) > 0:
            self._smooth()

        if not common_start:
            return self.cartesian()
        else:
            return self.polar()




    def cartesian(self) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(13, 10))

        ax = self._build_tracks(ax)

        
        if len(self.spot_data):
            x = self.spot_data.x_coordinate.to_numpy()
            y = self.spot_data.y_coordinate.to_numpy()
            ax.set_xlim(np.nanmin(x), np.nanmax(x))
            ax.set_ylim(np.nanmin(y), np.nanmax(y))

        ax.set_aspect('equal', adjustable='box')
        text_color = self.kwargs.get('text_color', 'black')
        ax.set_xlabel('x_coordinate [µm]', color=text_color)
        ax.set_ylabel('y_coordinate [µm]', color=text_color)
        ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=text_color)
        self._background_color(); ax.set_facecolor(self.face_color)

        # Ticks
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.tick_params(axis='both', which='major', labelsize=8, colors=text_color)

        # if self.grid:
        #     self._grid_color(coord_system='cartesian')
        #     self._grid_style(coord_system='cartesian', ax=ax)
        # else:
        #     ax.grid(False)

        ax.grid(False)
        
        if self.kwargs.get('show_heads', True):
            ax = self._head_markers(ax, polar=False)

        if self.kwargs.get('hide_backdrop', False):
            fig.set_facecolor('none')

        return plt.gcf()
    

    def polar(self) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})

        self._get_radius()
        self._convert_polar()

        text_color = self.kwargs.get('text_color', 'black')
        ax.set_title(self.kwargs.get('title', ''), fontsize=12, color=text_color)
        ax.set_ylim(0, self.y_max_global)        # <- global, consistent across subsets
        
        ax.spines['polar'].set_visible(False)

        ax = self._build_tracks(ax)

        self._background_color(); ax.set_facecolor(self.face_color)
        
        ax.grid(False)

        # if self.kwargs.get('grid', True):
        #     self._grid_color(coord_system='polar')
        #     self._grid_style(coord_system='polar', ax=ax)
        # else:
        #     ax.grid(False)

        self._annotate_r_axis(ax)
        self._annotate_theta_axis(ax)

        # self._color_segments(ax)

        if self.kwargs.get('show_heads', True):
            self._head_markers(ax, polar=True)

        if self.kwargs.get('hide_backdrop', False):
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
    

    # def _guard(self):
    #     missing_columns = [col for col in self.REQUIRED_COLS if col not in self.spot_data.columns]
    #     if missing_columns:
    #         raise MissingDataError(f"Missing required columns in spot_data: {missing_columns}")
        
    #     c_mode = self.kwargs.get('c_mode', None)

    #     if c_mode in ['categorical', 'quantitative']:
    #         color_by = self.kwargs.get('color_by', None)
    #         if color_by is None:
    #             raise ValueError("Missing 'color_by' parameter. Must be specified when c_mode is 'categorical' or 'quantitative'.")
    #         if color_by not in self.spot_data.columns:
    #             if is_empty(self.track_data):
    #                 raise MissingDataError(f"Failed to find: '{color_by}' data -> '{color_by}' data not found in spot_data columns -> parameter 'track_data' is empty or missing.")
    #             if color_by not in self.track_data.columns:
    #                 raise MissingDataError(f"Failed to find: '{color_by}' data -> '{color_by}' data not found in spot_data or track_data columns.")
        

    def _arrange_data(self):
        self.spot_data = categorize(self.spot_data, self.conditions, self.replicates, **self.kwargs)

        if not params.ignore_categories:
            self.spot_data = self.spot_data.sort_values(['condition', 'replicate', 'track_id', 'time_point'])
            if not is_empty(self.track_data):
                self.track_data = categorize(self.track_data, self.conditions, self.replicates, **self.kwargs)
                self.track_data = self.track_data.sort_values(['condition', 'replicate', 'track_id'])
        else:
            self.spot_data = self.spot_data.sort_values(['track_uid', 'time_point'])
            if not is_empty(self.track_data):
                self.track_data = self.track_data.sort_values(['track_uid'])


        if list(self.spot_data.index.names) != self.KEY_COLS:
            self.spot_data = self.spot_data.set_index(self.KEY_COLS)
            self.spot_data = self.spot_data
        if not is_empty(self.track_data) and list(self.track_data.index.names) != self.KEY_COLS:
            self.track_data = self.track_data.set_index(self.KEY_COLS)
            self.track_data = self.track_data
        

    def _convert_polar(self):
        self.spot_data.x_coordinate = self.spot_data.x_coordinate - self.spot_data.groupby(level=self.KEY_COLS).x_coordinate.transform('first')
        self.spot_data.y_coordinate = self.spot_data.y_coordinate - self.spot_data.groupby(level=self.KEY_COLS).y_coordinate.transform('first')

        self.spot_data['r'] = np.sqrt(self.spot_data.x_coordinate**2 + self.spot_data.y_coordinate**2)
        self.spot_data['theta'] = np.arctan2(self.spot_data.y_coordinate, self.spot_data.x_coordinate)


    def _get_radius(self):
        """
        Compute global maximum radius from (X, Y) positions.

        Works with both:
        - MultiIndex with levels ['Condition', 'Replicate', 'Track ID']
        - Regular Index with those columns present.
        """
        if is_empty(self.spot_data):
            self.y_max_global = 100.0
            self.y_max_label_global = "100 μm"
            return

        # Choose grouping mode based on index type
        if isinstance(self.spot_data.index, pd.MultiIndex) and list(self.spot_data.index.names) == self.KEY_COLS:
            group = self.spot_data.groupby(level=self.KEY_COLS)
        else:
            # Fall back to grouping by columns
            missing = [c for c in self.KEY_COLS if c not in self.spot_data.columns]
            if missing:
                # Reporter(Level.error, f"Cannot compute radius for polar plot. -> Missing required grouping columns: {missing}.", noticequeue=self.noticequeue)

                self.y_max_global = 100.0
                self.y_max_label_global = "100 μm"
                return
            group = self.spot_data.groupby(self.KEY_COLS)

        x = self.spot_data.x_coordinate
        y = self.spot_data.y_coordinate
        x0 = group.x_coordinate.transform('first')
        y0 = group.y_coordinate.transform('first')
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

        self.y_max_global = r.max() + 10.0
        self.y_max_label_global = f"{round(r.max()) + 10} μm"

        # if not np.isfinite(self.y_max_global):
        #     Reporter(Level.warning, f"Invalid maximum radius computed for polar plot. Setting to default '100 μm'.", details=f"Maximum radius was not finite: {self.y_max_global}.", noticequeue=self.noticequeue)
        #     self.y_max_global = 100.0
        # if not (self.y_max_global > 0):
        #     Reporter(Level.warning, 'Negative maximum radius. Setting to default "100 μm".', details=f"Maximum radius was a negative value: {self.y_max_global}.", noticequeue=self.noticequeue)
        #     self.y_max_global = 100.0
        #     self.y_max_label_global = "100 μm"
        
    def _smooth(self):
        if (isinstance(self.smoothing_index, (int))) and self.smoothing_index >= 1:
            _smoothing_window = self.smoothing_index
            if isinstance(_smoothing_window, float):
                _smoothing_window = round(self.smoothing_index)
                
            for col in ['X coordinate', 'Y coordinate']:
                self.Spots[col] = (
                    self.Spots.groupby(
                        level=self.KEY_COLS
                    )[col].transform(
                        lambda s: self.smooth_preserve_endpoints(s, _smoothing_window)
                    )
                )
        else:
            warnings.warn(message=f"Invalid 'smoothing_index': {self.smoothing_index} -> Must be a positive integer -> No smoothing applied.", 
                          category=UserWarning, 
                          stacklevel=2)
                          

    @staticmethod
    def smooth_preserve_endpoints(s: pd.Series, window: int) -> pd.Series:
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


    def _assign_color(self):

        if self.kwargs.get('color_by', None) is not None:
            color_by = self.kwargs.get('color_by', None)
            if self.kwargs.get('color', None) is not None:
                warnings.warn(message=f"Both 'color' and 'color_by' parameters are provided -> Parameter 'color' will be ignored -> Using 'color_by' for color assignment.",
                              category=ConflictingParametersWarning, 
                              stacklevel=2)

            datatype = None
            if isinstance(color_by, tuple) and len(color_by) == 2:
                color_by, datatype = color_by
                if datatype not in ['categorical', 'numeric']:
                    raise ValueError(f"Invalid datatype parameter '{datatype}' for color_by. Must be one of ['categorical', 'numeric'].")
            
            if datatype == 'categorical' or is_categorical_dtype(self.spot_data[color_by]):
                palette = self.kwargs.get('palette', None)

                if isinstance(palette, str):
                    categories = self.spot_data[color_by].dropna().unique().tolist()
                    palette = retrieve_palette(categories, palette)
                elif isinstance(palette, list):
                    categories = self.spot_data[color_by].dropna().unique().tolist()
                    palette = retrieve_palette(categories, palette)
                elif isinstance(palette, dict):
                    pass  # expected {category: color}, use as-is
                else:
                    raise PlottingError(f"Invalid palette type: {type(palette)}. Must be str, list, or dict.")

                self.spot_data['trackcolor'] = self.spot_data[color_by].map(palette).fillna("#000000FF")

            elif datatype == 'numeric' or is_numeric_dtype(self.spot_data[color_by]):
                cmap = retrieve_cmap(self.kwargs.get('cmap', None))
                norm, vals = cmap_lut(
                    self.spot_data[color_by],
                    min=self.kwargs.get('lut_vmin', None),
                    max=self.kwargs.get('lut_vmax', None),
                )

                try:
                    rgba = cmap(norm(np.asarray(vals, dtype=float)))
                    self.spot_data['trackcolor'] = list(rgba)
                except Exception as e:
                    raise PlottingError(f"Error applying quantitative colormap: '{cmap}' to data: {e}")
            
            else:
                raise InvalidParameterValueError(f"Invalid color_by value: '{color_by}'. Must be a column name in spot_data with categorical or numeric data or a tuple where the data type is specified (column_name, 'categorical'|'numeric').")

        else:
            c = self.kwargs.get('color', 'black')
            match c:
                case c if c in self._DYE_COLOR_SET or is_color_code(c):
                    self.spot_data['trackcolor'] = c
            
                case 'random_greys':
                    track_ids = self.spot_data['track_uid'].unique()
                    colors = random_grey(n=len(track_ids), code='hex', a=1.0)
                    color_map = dict(zip(track_ids, colors))
                    self.spot_data['trackcolor'] = self.spot_data['track_uid'].map(color_map)

                case c if c in ['random', 'random_colors', 'random_colours']:
                    track_ids = self.spot_data['track_uid'].unique()
                    colors = random_color(n=len(track_ids), code='hex', a=1.0)
                    color_map = dict(zip(track_ids, colors))
                    self.spot_data['trackcolor'] = self.spot_data['track_uid'].map(color_map)

                case _:
                    raise InvalidParameterValueError(f"Invalid color parameter: {self.kwargs.get('color')}. Must be a valid color name, hex code, or one of ['random', 'random_greys'].")


    def _build_tracks(
        self, 
        ax: plt.Axes | None = None, 
        *, 
        polar: bool = False
    ) -> plt.Axes | None:
        """
        Build and optionally plot track segments from self.spot_data efficiently.

        Tracks are rendered as one LineCollection instead of one line per track.
        This avoids Python-level plotting loops and is much faster for many tracks.
        """

        if is_empty(self.spot_data):
            raise MissingDataError("No spot data available to build tracks.")

        tuid = self.spot_data['track_uid'].to_numpy()
        if not np.all(tuid[1:] >= 0) or not (np.diff(pd.factorize(tuid)[0]) >= -0).all():
            self.spot_data = self.spot_data.sort_values(['track_uid', 'time_point'])

        if len(self.spot_data) < 2:
            self.segments = np.empty((0, 2, 2), dtype=float)
            self.segment_colors = np.empty((0,), dtype=object)
            return None

        group_codes = pd.factorize(self.spot_data['track_uid'], sort=False)[0]
        
        x = self.spot_data.x_coordinate.to_numpy(dtype=float, copy=False)
        y = self.spot_data.y_coordinate.to_numpy(dtype=float, copy=False)

        if polar:
            grouped = self.spot_data.groupby('track_uid', sort=False, observed=True)
            x0 = grouped.x_coordinate.transform("first").to_numpy(dtype=float, copy=False)
            y0 = grouped.y_coordinate.transform("first").to_numpy(dtype=float, copy=False)

            dx = x - x0
            dy = y - y0

            plot_x = np.arctan2(dy, dx)
            plot_y = np.sqrt(dx * dx + dy * dy)
        else:
            plot_x = x
            plot_y = y

        points = np.column_stack((plot_x, plot_y))

        valid = (
            (group_codes[1:] == group_codes[:-1])
            & (group_codes[1:] >= 0)
            & np.isfinite(points[:-1]).all(axis=1)
            & np.isfinite(points[1:]).all(axis=1)
        )

        self.segments = np.stack(
            (points[:-1][valid], points[1:][valid]),
            axis=1,
        )

        self.segment_colors = self.spot_data.trackcolor.to_numpy(dtype=object, copy=False)[:-1][valid]

        lc = LineCollection(
            self.segments,
            colors=self.segment_colors,
            linewidths=self.kwargs.get("lw", 1.0),
            zorder=10,
        )
        ax.add_collection(lc)

        return ax

    def _background_color(self):
        mapping = {
            'white': 'white',
            'light': 'lightgrey',
            'mid': 'darkgrey',
            'dark': 'dimgrey',
            'black': 'black',
        }
        self.face_color = mapping.get(self.kwargs.get('background', 'white'), 'white')

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
            self.grid_color, self.grid_alpha = mapping['cartesian'].get(self.kwargs.get('background', 'white'), ('gainsboro', 0.5))

        elif coord_system == 'polar':
            self.grid_color, self.grid_a_alpha, self.grid_alpha = mapping['polar'].get(self.kwargs.get('background', 'white'), ('lightgrey', 0.7, 0.8))

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
        match self.kwargs.get('r_axis', 'none'):
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
        if self.kwargs.get('theta_axis', 'none') == 'detailed':
            tlabels = ax.get_xticklabels()
            ax.set_xticklabels(tlabels, fontsize=10, color=self.text_color)
        else:
            ax.set_xticklabels([])


    def _head_markers(
        self,
        ax: plt.Axes,
        *,
        polar: bool = False
    ) -> plt.Axes:
        """
        Draw markers at track ends.

        polar=False -> use Cartesian coordinates (X/Y).
        polar=True  -> use polar coordinates (theta/r).

        If `spots` is provided, use that subset; otherwise use self.Spots.
        """
        x_coord, y_coord = ('theta', 'r') if polar else ('x_coordinate', 'y_coordinate')

        grouped = self.spot_data.groupby('track_uid', sort=False, observed=True)
        last_points = grouped.tail(1)

        ax.scatter(
            last_points[x_coord],
            last_points[y_coord],
            marker=self.kwargs.get("head_shape", "o"),
            s=self.kwargs.get("head_size", 10),
            edgecolor=last_points.trackcolor if self.kwargs.get("outline_head", True) else "none",
            facecolor=last_points.trackcolor if self.kwargs.get("fill_head", False) else "none",
            linewidths=self.kwargs.get("head_outline_width", 1.0),
            zorder=12
        )

        return ax

    def _color_segments(self, ax):
        if not self.segments:
            return
        lc = LineCollection(self.segments, colors=self.segment_colors, linewidths=self.lw, zorder=10)
        ax.add_collection(lc)

        
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