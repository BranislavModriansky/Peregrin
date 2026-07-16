import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Any
from scipy import stats

from ...categorizer import Painter, Categorizer
from ...compute.stats import Stats
from ...various import Values


_log = get_logger(__name__)


class PolarDataDistribute:

    n_line_points=1440

    def __init__(self, data: pd.DataFrame, conditions: list, replicates: list, *args, 
                 normalization: str = 'globally', cmap: str = 'plasma LUT', **kwargs):
        
        # - Shared arguments for all polar plots
        self.noticequeue = kwargs.get('noticequeue', None)

        self.data = data
        self.data_cache = data.copy()
        self.conditions = conditions
        self.replicates = replicates
        
        self.normalization = normalization
        self.weight_by = kwargs.get('weight_by', None)  # column name for weighting, e.g. 'Track displacement'

        self.cmap = Painter(noticequeue=self.noticequeue).GetCmap(cmap)

        self.kwargs = kwargs

        # - Keyword arguments and constants for Gaussian KDE Colormesh
        # bins and bandwidth already mentioned
        self.OUTER_RADIUS = 1.0
        self.INNER_RADIUS = 0.7
        self.WIDTH = self.OUTER_RADIUS - self.INNER_RADIUS
        self.D_THETA = 2 * np.pi / self.kwargs.get('bins', 16)

        self._check_input()

    def GaussianKDEColormesh(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

        bins = self.kwargs.get('bins', 16)
        self.kappa = self._bandwidth_to_kappa()

        self._arrange_data()
        self.theta, self.density = self._theta_density(self.angles, bins)
        self._density_norm()

        D_THETA = 2 * np.pi / bins

        # Use a 0-1 norm for color mapping based on normalized_density
        if self.normalization in ['locally', 'globally']:
            colormesh_norm = Normalize(self._min_density, self._max_density)
        else:
            colormesh_norm = Normalize(0.0, 1.0)

        self._plot_tiles(
            ax, 
            data=self.theta, 
            height=np.full_like(self.theta, self.WIDTH),
            widths=D_THETA * 1.1,
            bottom=self.INNER_RADIUS,
            color=self.cmap(colormesh_norm(self.normalized_density)),
            antialiased=False
        )

        theta_line = np.linspace(0, 2 * np.pi, 500)
        for circuit in (self.INNER_RADIUS, self.OUTER_RADIUS):
            ax.plot(theta_line, np.full_like(theta_line, circuit), color='black', linewidth=0.9)

        self._annotate_theta_axis(ax, create=True)
        if self.kwargs.get('title', None) is not None:
            ax.set_title(self.kwargs.get('title', ''), color=self.kwargs.get('text_color', 'black'), fontsize=14, pad=20)
        ax.set_axis_off()

        sm = plt.cm.ScalarMappable(norm=colormesh_norm, cmap=self.cmap)
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.115, fraction=0.045)
        cbar.set_ticks([])

        caps = {
            'min': self._min_density,
            'max': self._max_density
        }
        for cap, value in caps.items():
            if value == 0.0:
                display_value = '0'
            elif value == 1.0:
                display_value = '1'
            else:
                display_value = f'{value:.2f}'

            cbar.ax.text(0.015 if cap == 'min' else 0.985, -0.75, display_value, va='center', ha='center', color=self.kwargs.get('text_color', 'black'), transform=cbar.ax.transAxes, fontsize=9, fontstyle='italic')

        cbar.set_label("Density", labelpad=10, color=self.kwargs.get('text_color', 'black'))
        
        fig.set_facecolor(self.kwargs.get('facecolor', 'none'))

        return plt.gcf()

    def KDELinePlot(self) -> plt.Figure:

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        outline = self.kwargs.get('outline', False)
        outline_color = self.kwargs.get('outline_color', 'blend in')
        if outline_color == 'blend in':
            outline_color = self.kwargs.get('background', 'white')
        outline_width = self.kwargs.get('outline_width', 0)
        kde_fill = self.kwargs.get('kde_fill', True)
        kde_fill_color = self.kwargs.get('kde_fill_color', 'blue')
        kde_fill_alpha = self.kwargs.get('kde_fill_alpha', 0.4)

        self.kappa = self._bandwidth_to_kappa()

        self._arrange_data()

        self.theta, self.density = self._theta_density(self.angles, self.n_line_points)
        self._density_norm(num_points=self.n_line_points)

        _log.info(f"[DEBUG] KDE density range before normalization: [{np.nanmin(self.density)}, {np.nanmax(self.density)}]")
        _log.info(f"[DEBUG] KDE density range after normalization: [{np.nanmin(self.normalized_density)}, {np.nanmax(self.normalized_density)}]")
        _log.info(f"[DEBUG] theta range: [{self.theta[0]:.2f}, {self.theta[-1]:.2f}] with {self.theta.size} points\nData range: [{np.nanmin(self.angles):.2f}, {np.nanmax(self.angles):.2f}] with {self.angles.size} valid data points")
        _log.info(f"[DEBUG] data: {self.angles}")

        if outline:
            ax.plot(
                self.theta, 
                self.normalized_density, 
                color=outline_color,
                linewidth=outline_width,
                zorder=9 
            )
        
        if kde_fill:
            ax.fill_between(
                self.theta, 
                0, 
                self.normalized_density, 
                color=kde_fill_color, 
                alpha=kde_fill_alpha,
                zorder=5,
                linewidth=0
            )

        self._mean_dir_dial(ax)
        self._annotate_theta_axis(ax)
        self._annotate_dens_r_axis(ax)
        ax.set_facecolor(self.kwargs.get('background', 'white'))
        fig.set_facecolor(self.kwargs.get('facecolor', 'none'))

        if self.kwargs.get('title', None) is not None:
            ax.set_title(self.kwargs.get('title', ''), color=self.kwargs.get('text_color', 'black'), fontsize=14, pad=20)

        return plt.gcf()


    def RoseChart(self) -> plt.Figure:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        self._arrange_data()

        self._define_bins()
        self._polar_hist(ax, fig=fig)
        self._annotate_theta_axis(ax)
        self._annotate_hist_r_axis(ax)
        ax.set_facecolor(self.kwargs.get('background', 'white'))
        fig.set_facecolor(self.kwargs.get('facecolor', 'none'))
        if self.kwargs.get('title', None) is not None:
            ax.set_title(self.kwargs.get('title', ''), color=self.kwargs.get('text_color', 'black'), fontsize=14, pad=20)

        return plt.gcf()


    def get_density_range(self) -> tuple[float, float]:

        self.kappa = self._bandwidth_to_kappa()

        self._arrange_data()
        self.theta, self.density = self._theta_density(self.angles, self.n_line_points)
        self._density_norm()

        return self._min_density, self._max_density
    
    # TODO: Extend this method to check input parameters
    def _check_input(self) -> None:

        bins_raw = self.kwargs.get('bins', 16)
        ntiles_raw = self.kwargs.get('ntiles', 10)
        gap_raw = self.kwargs.get('gap', 0.1)
        r_loc_raw = self.kwargs.get('r_loc', 75)

        # sanitize integer-like inputs
        try:
            bins = int(bins_raw)
        except (TypeError, ValueError):
            bins = 16
            if self.noticequeue is not None:
                self.noticequeue.Report(Level.warning, "Invalid bins value. Resetting to 16.")
        if bins < 2:
            bins = 2
            if self.noticequeue is not None:
                self.noticequeue.Report(Level.error, "Number of bins must be at least 2.")
        self.kwargs['bins'] = bins

        try:
            ntiles = int(ntiles_raw)
        except (TypeError, ValueError):
            ntiles = 10
            if self.noticequeue is not None:
                self.noticequeue.Report(Level.warning, "Invalid ntiles value. Resetting to 10.")
        if ntiles < 1:
            ntiles = 1
            if self.noticequeue is not None:
                self.noticequeue.Report(Level.error, "Number of tiles must be at least 1.")
        self.kwargs['ntiles'] = ntiles

        # sanitize floats
        try:
            gap = float(gap_raw)
        except (TypeError, ValueError):
            gap = 0.1

        if not np.isfinite(gap):
            gap = 0.1

        if not 0.0 <= gap <= 1.0:
            gap = Values.Clamp01(gap)
            if self.noticequeue is not None:
                self.noticequeue.Report(Level.error, "Gap must be in range (0, 1). Clamped to: " + str(gap))
        self.kwargs['gap'] = gap

        try:
            r_loc = float(r_loc_raw)
        except (TypeError, ValueError):
            r_loc = 75.0

        if not np.isfinite(r_loc) or not (0 <= r_loc <= 360):
            r_loc = 75.0
            if self.noticequeue is not None:
                self.noticequeue.Report(Level.warning, "R axis labels position must be in range (0, 360). Resetting to 75.")
        self.kwargs['r_loc'] = r_loc


    def _arrange_data(self, wrap: bool = True) -> None:
        self.data = Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
            noticequeue=self.noticequeue
        )()

        angles = np.asarray(self.data['Direction mean'], dtype=float)
        angles = angles % (2 * np.pi) if wrap else angles

        valid_angles = np.isfinite(angles)
        if not np.all(valid_angles):
            dropped = int((~valid_angles).sum())
            self.data = self.data.iloc[valid_angles].copy()
            angles = angles[valid_angles]
            Reporter(Level.warning, f"Ignored {dropped} rows with invalid direction values.", noticequeue=self.noticequeue)

        self.angles = angles

        # Extract weights for the current subset
        if self.weight_by is not None and self.weight_by in self.data.columns:
            w = np.asarray(self.data[self.weight_by], dtype=float)
            w = np.where(np.isfinite(w), w, np.nan)
            w_sum = np.nansum(w)
            self.weights = w / w_sum if w_sum > 0 else np.ones(w.size) / max(w.size, 1)
        else:
            self.weights = None

    def _get_weights_for(self, data: pd.DataFrame) -> np.ndarray | None:
        """Extract and normalize weights from an arbitrary DataFrame (used for global normalization)."""
        if self.weight_by is not None and self.weight_by in data.columns:
            w = np.asarray(data[self.weight_by], dtype=float)
            w = np.where(np.isfinite(w), w, np.nan)
            w_sum = np.nansum(w)
            return w / w_sum if w_sum > 0 else np.ones(w.size) / max(w.size, 1)
        return None

    def _theta_density(self, a: np.ndarray, n: int = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
        -----------
        a: array-like
            Input angles in radians.
        n: int, optional
            Number of points to evaluate the density on. If None, defaults to 360 for smoothness.
        """

        check = np.isfinite(a)
        angles = a[check]

        n = 360 if n is None else n

        if angles.size < 2:
            theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
            density = np.zeros(n, dtype=float)
        else:
            theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

            _log.info(f"[DEBUG] theta range: [{theta[0]:.2f}, {theta[-1]:.2f}] with {n} points\nData range: [{np.nanmin(angles):.2f}, {np.nanmax(angles):.2f}] with {angles.size} valid data points")
        
            density = np.mean(
                stats.vonmises.pdf(theta[:, None], self.kappa, loc=angles),
                axis=1
            )

        return theta, density
    
    def _density_norm(self, wrap: bool = True, num_points: int = None) -> None:

        auto_lut_scale = self.kwargs.get('auto_lut_scale', True)

        if self.normalization in ['locally', 'globally']:
            _min_density = np.min(self.density)
            _max_density = np.max(self.density)
        else:
            # No normalization: normalize by global max but r-axis zooms to local peak
            all_angles_raw = np.asarray(self.data_cache['Direction mean'], dtype=float)
            all_angles_raw = all_angles_raw % (2 * np.pi) if wrap else all_angles_raw
            global_weights = self._get_weights_for(self.data_cache)
            _, all_density = self._theta_density(all_angles_raw, num_points)
            _min_density = np.min(all_density)
            _max_density = np.max(all_density)

        if not auto_lut_scale:
            _min_density = self.kwargs.get('min_density', None)
            _max_density = self.kwargs.get('max_density', None)
            if _min_density is None:
                Reporter(Level.warning, "Minimum density not provided -> setting 0.", noticequeue=self.noticequeue)
                _min_density = 0.0
            if _max_density is None:
                Reporter(Level.warning, "Maximum density not provided -> setting 1.", noticequeue=self.noticequeue)
                _max_density = 1.0
            if _min_density >= _max_density:
                Reporter(Level.error, "Minimum density must be less than maximum density -> resetting to 0 and 1.", noticequeue=self.noticequeue)
                _min_density = 0.0
                _max_density = 1.0
        
        self._min_density = _min_density
        self._max_density = _max_density
            
        self.norm = Normalize(self._min_density, self._max_density)
        if _max_density == 0:
            self.normalized_density = np.zeros_like(self.density)
        else:
            if self.normalization in ['locally', 'globally']:
                # Local: peak is always 1.0; r-axis goes to 1.0
                self.normalized_density = self.density
                _log.info(f"[DEBUG] Normalized density range after normalization: [{np.nanmin(self.normalized_density)}, {np.nanmax(self.normalized_density)}]")
            else:
                # No normalization: density is divided by global max, but r-axis zooms to local peak
                self.normalized_density = self.density
                _log.info(f"[DEBUG] Normalized density range after normalization: [{np.nanmin(self.normalized_density)}, {np.nanmax(self.normalized_density)}]")
        
    def _polar_hist(self, ax, *, fig = None) -> None:

        bins = self.kwargs.get('bins', 16)
        c_mode = self.kwargs.get('c_mode', 'single color')
        levels_count = self.kwargs.get('levels', 5)
        ntiles = self.kwargs.get('ntiles', 10)
        discretize_col = self.kwargs.get('discretize', None)
        default_colors = self.kwargs.get('default_colors', True)
        single_color = self.kwargs.get('single_color', 'coral')
        outline = self.kwargs.get('outline', False)

        match c_mode:

            # Computations are light, so performance should be acceptable
            # even with this amount of for cycles and clarity can be prioritized

            case 'single color':

                levels = 1
                colors = [single_color]
                heights = [[float(self.bin_counts[b])] for b in range(bins)]
                bottoms = [[0.0] for _ in range(bins)]

                bin_idx = 'pass'
                

            case 'level-based':

                levels = levels_count
                colors = self._slice_cmap(levels_count)
                
                level_height = max(self.bin_counts) / levels_count
                heights = [
                    [min((lvl + 1) * level_height, float(self.bin_counts[b])) - lvl * level_height 
                     for lvl in range(levels_count)] 
                    for b in range(bins)
                ]
                bottoms = [
                    [lvl * level_height 
                     for lvl in range(levels_count)] 
                    for _ in range(bins)
                ]

                bin_idx = 'pass'
                

            case 'n-tiles':

                colors = self._slice_cmap(ntiles)
                levels = ntiles

                bin_idx = np.digitize(self.angles, self.bin_edges, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, bins - 1)

                qs = np.linspace(0.0, 1.0, ntiles + 1)
                tile_spans = np.zeros(self.angles.size, dtype=int)

                if discretize_col is None or discretize_col not in self.data.columns:
                    global_edges = np.linspace(0.0, 1.0, ntiles + 1)
                    if self.noticequeue is not None:
                        Reporter(Level.error, "Missing/invalid discretization column for 'n-tiles'. Using a single fallback tile.", noticequeue=self.noticequeue)
                else:
                    discretize = pd.to_numeric(self.data[discretize_col], errors='coerce').to_numpy(dtype=float)
                    finite_mask = np.isfinite(discretize)

                    if not np.any(finite_mask):
                        global_edges = np.linspace(0.0, 1.0, ntiles + 1)
                        if self.noticequeue is not None:
                            Reporter(Level.warning, "Discretization column contains only NaN/invalid values. Using fallback scale.", noticequeue=self.noticequeue)
                    else:
                        global_edges = np.quantile(discretize[finite_mask], qs)

                        if np.allclose(global_edges, global_edges[0]):
                            if self.noticequeue is not None:
                                Reporter(Level.warning, "All values in the discretization column are identical. All data points will be assigned to the first tile.", noticequeue=self.noticequeue)
                        else:
                            tile_spans[finite_mask] = np.digitize(discretize[finite_mask], global_edges[1:-1], right=False)

                        if np.any(~finite_mask) and self.noticequeue is not None:
                            Reporter(Level.warning, "NaN/invalid discretization values were assigned to the first tile.", noticequeue=self.noticequeue)

                tile_counts = [
                    np.bincount(tile_spans[bin_idx == b], minlength=ntiles).astype(float)
                    for b in range(bins)
                ]

                heights = [
                    [tile_counts[b][n]
                     for n in range(ntiles)]
                    for b in range(bins)
                ]
                bottoms = [
                    [sum(tile_counts[b][0:n])
                     for n in range(ntiles)]
                    for b in range(bins)
                ]

                self._plot_cbar(fig, ax, colors, global_edges)


            case 'differentiate conditions':

                unique_cond, cond_idx = np.unique(self.data['Condition'], return_inverse=True)

                levels = len(unique_cond)
                
                if default_colors:
                    colors = self._slice_cmap(len(unique_cond))
                else:
                    try:
                        colors = [self.data[self.data['Condition'] == cond].iloc[0]['Condition color'] for cond in unique_cond]
                    except KeyError:
                        Reporter(Level.error, "Condition color information is missing in the data. Reverting to default colors.", noticequeue=self.noticequeue)
                        colors = self._slice_cmap(len(unique_cond))
                    except Exception as e:
                        Reporter(Level.error, f"An error occurred while retrieving condition colors: {e}. Reverting to default colors.", noticequeue=self.noticequeue)
                        colors = self._slice_cmap(len(unique_cond))
                
                bin_idx = np.digitize(self.angles, self.bin_edges, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, bins - 1)

                cond_score = [
                    np.bincount(cond_idx[bin_idx == b], minlength=unique_cond.size).astype(float)
                    for b in range(bins)
                ]

                heights = [
                    [cond_score[b][c]
                     for c in range(len(unique_cond))]
                    for b in range(bins)
                ]
                bottoms = [
                    [sum(cond_score[b][0:c])
                     for c in range(len(unique_cond))]
                    for b in range(bins)
                ]

                handles = [mpl.patches.Patch(color=colors[i], label=unique_cond[i]) for i in range(len(unique_cond))]
                
                self._plot_legend(ax, handles)

                
            case 'differentiate replicates':

                unique_repl, repl_idx = np.unique(self.data['Replicate'], return_inverse=True)

                levels = len(unique_repl)
                
                if default_colors:
                    colors = self._slice_cmap(len(unique_repl))
                else:
                    try:
                        colors = [self.data[self.data['Replicate'] == repl].iloc[0]['Replicate color'] for repl in unique_repl]
                    except KeyError:
                        Reporter(Level.error, "Replicate color information is missing in the data. Reverting to default colors.", noticequeue=self.noticequeue)
                        colors = self._slice_cmap(len(unique_repl))
                    except Exception as e:
                        Reporter(Level.error, f"An error occurred while retrieving replicate colors: {e}. Reverting to default colors.", noticequeue=self.noticequeue)
                        colors = self._slice_cmap(len(unique_repl))
                
                bin_idx = np.digitize(self.angles, self.bin_edges, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, bins - 1)

                repl_score = [
                    np.bincount(repl_idx[bin_idx == b], minlength=unique_repl.size).astype(float)
                    for b in range(bins)
                ]

                heights = [
                    [repl_score[b][r]
                     for r in range(len(unique_repl))]
                    for b in range(bins)
                ]
                bottoms = [
                    [sum(repl_score[b][0:r])
                     for r in range(len(unique_repl))]
                    for b in range(bins)
                ]

                handles = [mpl.patches.Patch(color=colors[i], label=unique_repl[i]) for i in range(len(unique_repl))]
                
                self._plot_legend(ax, handles)



        for b in range(bins):

            try:
                if bin_idx == 'pass':
                    pass
            except Exception:
                if not np.any(bin_idx == b):
                    continue

            for lvl in range(levels):

                if heights[b][lvl] <= 0:
                    continue

                self._plot_tiles(
                    ax,
                    data = self.bin_locs[b],
                    height = heights[b][lvl],
                    widths = self.bin_widths[b],
                    color = colors[lvl],
                    bottom = bottoms[b][lvl],
                    zorder = 10
                )
    

    def _plot_tiles(self, ax, data: np.ndarray, height: np.ndarray, widths: np.ndarray, 
                    bottom: float = 0.0, color: Any = None, **kwargs) -> None:
        
        alignment = self.kwargs.get('alignment', 'edge')
        outline = self.kwargs.get('outline', False)
        outline_color = self.kwargs.get('outline_color', 'blend in')
        if outline_color == 'blend in':
            outline_color = self.kwargs.get('background', 'white')
        outline_width = self.kwargs.get('outline_width', 0)

        ax.bar(
            data,
            height,
            width=widths,
            bottom=bottom,
            color=color,
            align=alignment,
            edgecolor=outline_color if outline else 'none',
            linewidth=outline_width if outline else 0,
            **kwargs
    )

    def _define_bins(self) -> None:
        bins = self.kwargs.get('bins', 16)
        gap = float(self.kwargs.get('gap', 0.1))

        # Bin edges
        self.bin_edges = np.linspace(0.0, 2 * np.pi, bins + 1)
        bin_width = np.diff(self.bin_edges)

        # Gaps between bars
        self.bin_locs = self.bin_edges[:-1] + (bin_width * gap / 2.0)
        self.bin_widths = bin_width * (1.0 - gap)

        self.bin_counts, _ = np.histogram(self.angles, bins=self.bin_edges)

    def _mean_dir_dial(self, ax) -> None:
        show_abs_average = self.kwargs.get('show_abs_average', True)
        mean_angle_color = self.kwargs.get('mean_angle_color', 'black')
        mean_angle_width = self.kwargs.get('mean_angle_width', 3)
        peak_direction_trend = self.kwargs.get('peak_direction_trend', False)
        peak_direction_trend_color = self.kwargs.get('peak_direction_trend_color', 'red')
        peak_direction_trend_width = self.kwargs.get('peak_direction_trend_width', 1)

        if show_abs_average:
            sin_sum = np.sum(np.sin(self.angles))
            cos_sum = np.sum(np.cos(self.angles))

            prominent_angle = np.arctan2(sin_sum, cos_sum)

            length = np.interp(prominent_angle, self.theta, self.normalized_density)

            ax.vlines(prominent_angle, 0, length, color=mean_angle_color, linewidth=mean_angle_width, zorder=7)
        
        if peak_direction_trend:
            peak_angle = self.theta[np.argmax(self.normalized_density)]

            ax.vlines(peak_angle, 0, self.normalized_density[np.argmax(self.normalized_density)], color=peak_direction_trend_color, linewidth=peak_direction_trend_width, zorder=6)

    def _plot_legend(self, ax, handles: list) -> None:

        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=len(handles),
            frameon=False,
            fontsize=9,
            labelcolor=self.kwargs.get('text_color', 'black'),
            handlelength=1.2,
            handletextpad=0.5,
            columnspacing=1.0
        )

    def _plot_cbar(self, fig, ax, colors, global_edges) -> None:
        """
        Discrete horizontal colorbar for n-tiles.

        IMPORTANT: Uses fig.colorbar(..., ax=ax) instead of fig.add_axes(...)
        to avoid Shiny coordmap crashing on Axes without SubplotSpec.
        """
        if fig is None:
            fig = ax.figure

        ntiles = self.kwargs.get('ntiles', 10)
        outline = self.kwargs.get('outline', False)
        outline_color = self.kwargs.get('outline_color', 'blend in')
        if outline_color == 'blend in':
            outline_color = self.kwargs.get('background', 'white')
        outline_width = self.kwargs.get('outline_width', 0)
        discretize_col = self.kwargs.get('discretize', None)

        n = int(ntiles)
        if n < 1:
            return

        edges = np.asarray(global_edges, dtype=float)
        if edges.size != (n + 1):
            edges = np.linspace(0.0, 1.0, n + 1)

        # Discrete mapping: bins [0..n] with n colors
        cmap = mpl.colors.ListedColormap(list(colors))
        boundaries = np.arange(n + 1)
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            pad=0.10,
            fraction=0.045,
            ticks=boundaries,
            spacing="uniform",
            drawedges=bool(outline),
        )

        cbar.ax.tick_params(axis="x", length=0, pad=4, colors=self.kwargs.get('text_color', 'black'))
        units = Stats().stat_units(discretize_col)
        units = "" if units is None else f' [{units}]'
        cbar.set_label(f'{discretize_col}{units}', color=self.kwargs.get('text_color', 'black'), fontsize=10, labelpad=10)

        finite_edges = edges[np.isfinite(edges)]
        span = float(finite_edges.max() - finite_edges.min()) if finite_edges.size >= 2 else 0.0

        def fmt_val(v):
            if not np.isfinite(v):
                return "NA"
            if not np.isfinite(span) or span < 10:
                return f"{v:.2g}"
            return f"{int(np.rint(v))}"

        cbar.ax.set_xticklabels([fmt_val(v) for v in edges], fontsize=9, color=self.kwargs.get('text_color', 'black'))

        if not outline:
            cbar.outline.set_visible(False)
        else:
            cbar.outline.set_edgecolor(outline_color)
            cbar.outline.set_linewidth(outline_width)

    def _annotate_theta_axis(self, ax, create: bool = False) -> None:

        if self.kwargs.get('label_theta', True):
            if create:
                for angle in range(0, 360, 45):
                    ax.text(np.deg2rad(angle), self.OUTER_RADIUS + 0.1, f"{angle}°", 
                            ha='center', va='center', fontsize=10, color=self.kwargs.get('text_color', 'black'), fontweight="medium", zorder=500)
            else:
                ax.tick_params(axis="x", labelcolor=self.kwargs.get('text_color', 'black'))
        elif not self.kwargs.get('label_theta', True):
            ax.set_xticklabels([])
        
    def _annotate_dens_r_axis(self, ax) -> None:
        r_loc = self.kwargs.get('r_loc', 75)

        if self.kwargs.get('label_r', True):
            ax.tick_params(axis="y", labelcolor=self.kwargs.get('text_color', 'black'))

            if self.normalization in ['locally', 'globally']:
                # Global: r-axis always 0 to 1.0; weaker subsets don't reach the top
                max = self._max_density
                ax.set_ylim(0, max * 1.1)
                ax.set_yticks(np.linspace(0, max, 5))

            else:
                # No normalization: density is divided by global max,
                # but r-axis zooms to the local subset's peak.
                # Labels show the actual normalized density values.
                max = 1
                ax.set_ylim(0, 1.1)
                ax.set_yticks(np.linspace(0, 1, 5))
            
            ax.set_yticklabels(['', f'{max * 0.25:.2f}', f'{max * 0.50:.2f}', f'{max * 0.75:.2f}', f'{max:.2f}'])
            ax.set_rlabel_position(r_loc)
            ax.yaxis.label.set_color(self.kwargs.get('label_r_color', 'lightgrey'))
            ax.yaxis.label.set_fontweight("medium")
        else:
            ax.set_yticklabels([])

    def _annotate_hist_r_axis(self, ax) -> None:
        r_loc = self.kwargs.get('r_loc', 75)

        peak = float(np.nanmax(self.bin_counts)) if np.size(self.bin_counts) else 0.0
        if not np.isfinite(peak) or peak <= 0:
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_yticklabels([])
            return

        ax.set_ylim(0, peak * 1.05)
        ax.set_yticks(np.linspace(0, peak, 6))
        ax.set_yticklabels([])

        if self.kwargs.get('label_r', True):
            max_val = int(np.ceil(peak))
            step = max(1, int(np.ceil(max_val / 5)))
            for r in range(step, max_val + 1, step):
                ax.text(
                    np.deg2rad(r_loc),
                    r,
                    str(r),
                    ha="center",
                    va="center",
                    fontsize=10,
                    zorder=100,
                    color=self.kwargs.get('label_r_color', 'black'),
                    fontweight="medium",
                )

    def _slice_cmap(self, slices: int) -> list:
        return self.cmap(np.linspace(0.05, 0.95, slices))

    def _bandwidth_to_kappa(self, kappa_min=0, kappa_max=1e4) -> float:
        
        if self.kwargs.get('bw', None) is not None:
            return np.clip(1.0 / (self.kwargs.get('bw', 1.0) ** 2), kappa_min, kappa_max)

        else:
            return np.clip(self.kwargs.get('kappa', 25), kappa_min, kappa_max)

