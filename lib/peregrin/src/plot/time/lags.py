import traceback

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from typing import *

from ..categorizer import Categorizer, Painter
from ...various import is_empty
from ...compute.stats import Stats


class MSD:
    """
    #### *Mean or Median Squared Displacement analysis and visualization class.*
    
    Provides methods for computing and plotting MSD statistics across time lags,
    with support for multiple conditions, linear fitting, and various scaling options.
    """

    # Constants for color adjustments in linear fits
    SATURATION_SCALE = 0.7
    SATURATION_MIN = 0.02
    SATURATION_MAX = 1.0
    BRIGHTNESS_SCALE = 0.8
    BRIGHTNESS_MIN = 0.06


    def __init__(self, data: pd.DataFrame, conditions: list, replicates: list, 
                 *, log_transform: bool = False, level: str = 'Condition', **kwargs):
        
        self.data = data
        self.conditions = conditions
        self.replicates = replicates
        
        self.level = level
        self.disaggregate = False
        self.log_transform = log_transform

        self.color = kwargs.get('color', None)
        self.stock_palette = kwargs.get('stock_palette', None)
        self.palette = kwargs.get('palette', None)
        self.noticequeue = kwargs.get('noticequeue', None)

        self.instance_kwargs = kwargs

        self.painter = Painter(noticequeue=self.noticequeue)

        self._check_errors()


    def plot(self,
             statistic: str = 'mean',
             linear_fit: bool = False,
             disper: Optional[Literal['sd', 'sem', 'min-max', 'ci']] = None,
             *,
             line: bool = True,
             scatter: bool = False,
             **kwargs) -> plt.Figure:

        from ...compute.stats import Stats
        
        if self.level == 'Condition':
            prefix = '{per condition}'
            if not any('{per condition}' in col for col in self.data.columns.to_list()):
                Reporter(Level.warning, "Expected aggregated data per condition but no columns with '{per condition}' found. Going to use '{per replicate}' data instead.", noticequeue=self.noticequeue)
                prefix = '{per replicate}'
        else:
            prefix = '{per replicate}'

        fig, ax = plt.subplots(figsize=(kwargs.get('fig_width', 11), kwargs.get('fig_height', 7.5)))

        if self.log_transform:
            ax.set_xscale('log')
            ax.set_yscale('log')

        # ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
        # ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs='auto', numticks=100))
        # ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        # ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        # ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
        # ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs='auto', numticks=100))
        # ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        # ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        self.agg_dict, cols = self._build_agg_dict(prefix, Stats.CI_STATISTIC, Stats.CONFIDENCE_LEVEL)
        
        # Extract colors BEFORE aggregation drops the color columns
        color_map = self._get_colors()
        
        self._arrange_data()

        mi = cols['mean'] if statistic == 'mean' else cols['median']

        self._set_axis_labels(ax)
            
        # Plot each condition
        for idx, condition in enumerate(self.conditions):
            cond_data = self.data[self.data['Condition'] == condition]

            if self.level == 'Condition':
                # grouped replicates
                groups = [(condition, None, cond_data)]
            else:
                # separate replicates
                groups = []
                for rep, rep_df in cond_data.groupby('Replicate'):
                    groups.append((condition, rep, rep_df))

            for g_idx, (cond_name, rep_name, gdata) in enumerate(groups):

                x_data = gdata['Time lag'].values
                y_data = gdata[f'{prefix} MSD {statistic}'].values
              
                match disper:

                    case 'sd' | 'std':

                        if statistic != 'mean':
                            Reporter(Level.warning, "Error band with 'sd' is only meaningful when plotting the mean. Ignoring error band.", noticequeue=self.noticequeue)
                            disper = False
                        else:
                            err_data = gdata[f'{prefix} MSD sd'].values / 2

                    case 'sem':

                        if f'{prefix} MSD sem' not in gdata.columns:
                            Reporter(Level.error, "SEM data not available for error band. Make sure to compute SEM in the statistics step. Ignoring error band.", noticequeue=self.noticequeue)
                            disper = False

                        else:
                            if statistic != 'mean':
                                Reporter(Level.warning, "Error band with 'sem' is only meaningful when plotting the mean. Ignoring error band.", noticequeue=self.noticequeue)
                                disper = False
                            else:
                                err_data = gdata[f'{prefix} MSD sem'].values

                    case 'min-max':
                        err_data = (gdata[f'{prefix} MSD max'].values, gdata[f'{prefix} MSD min'].values)

                    case 'ci':
                        
                        if f'{prefix} MSD {Stats.CI_STATISTIC} ci{Stats.CONFIDENCE_LEVEL} low' not in gdata.columns or f'{prefix} MSD {Stats.CI_STATISTIC} ci{Stats.CONFIDENCE_LEVEL} high' not in gdata.columns:
                            Reporter(Level.error, "Confidence interval data not available for error band. Make sure to compute confidence intervals in the statistics step. Ignoring error band.", noticequeue=self.noticequeue)
                            disper = False

                        else:
                            if statistic != Stats.CI_STATISTIC:
                                Reporter(Level.warning, f"Confidence interval was coputed for {Stats.CI_STATISTIC}, not {statistic}. Ignoring error band.", noticequeue=self.noticequeue)
                                disper = False
                            else:
                                err_data = (gdata[f'{prefix} MSD {Stats.CI_STATISTIC} ci{Stats.CONFIDENCE_LEVEL} high'].values, gdata[f'{prefix} MSD {Stats.CI_STATISTIC} ci{Stats.CONFIDENCE_LEVEL} low'].values)

                    case 'none' | None | False:
                        disper = False

                    case _:
                        Reporter(Level.error, f"Invalid 'disper' (dispersion) type '{disper}'.", noticequeue=self.noticequeue)
                        disper = False

                if disper:
                    if isinstance(err_data, tuple):
                        band_top_y, band_bottom_y = err_data
                    else:
                        band_bottom_y = np.maximum(y_data - err_data, 0.0)
                        band_top_y = y_data + err_data
                
                color = color_map.get(cond_name) if self.level == 'Condition' else color_map.get(rep_name)
                color = self._resolve_color(color, idx if self.level == 'Condition' else g_idx)

                # label: show replicate name only once in legend when not grouped
                if self.level == 'Condition':
                    label = cond_name
                else:
                    label = f"{cond_name} | {rep_name}"

                # Plot error band
                if disper:
                    mask = ~np.isnan(band_bottom_y) & ~np.isnan(band_top_y)
                    if np.any(mask):
                        ax.fill_between(
                            x_data[mask],
                            band_bottom_y[mask],
                            band_top_y[mask],
                            color=color,
                            alpha=0.08 if self.level == 'Condition' else 0.05,
                            linewidth=0,
                            zorder=2
                        )
                
                # Plot main line
                if line:
                    pass
                    ax.plot(
                        x_data,
                        y_data,
                        marker='none',
                        label=label,
                        linestyle='-',
                        color=color,
                        alpha=1 if self.level == 'Condition' else 0.8,
                        zorder=6
                    )
                
                # Plot scatter markers
                if scatter:
                    ax.plot(
                        x_data,
                        y_data,
                        marker='o',
                        markersize=6,
                        label=None,
                        linestyle='none',
                        color=color,
                        zorder=5
                    )
                
                # Add linear fit (per replicate when not grouped)
                if linear_fit:
                    # pass a unique condition key for annotation placement
                    fit_key = f"{condition}" if self.level == 'Condition' else f"{condition}-{rep_name}"
                    self._add_linear_fit(
                        ax,
                        x_data,
                        y_data,
                        fit_key,
                        color,
                        idx if self.level == 'Condition' else g_idx,
                        len(self.conditions) if self.level == 'Condition' else len(groups)
                    )
        
        # Set y-limits
        self._set_ylim(ax, self.data[mi].values)

        if kwargs.get('title', None):
            ax.set_title(
                kwargs.get('title', None), 
                color=kwargs.get('text_color', 'black'), 
                fontsize=kwargs.get('title_fontsize', 14), 
                fontweight=kwargs.get('title_fontweight', 'bold')
            )
        
        if kwargs.get('grid', False):
            ax.grid(True, color='whitesmoke', zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(frameon=False)

        fig.set_facecolor(kwargs.get('fig_background', 'white'))
        
        return plt.gcf()
    
    
    def _arrange_data(self) -> pd.DataFrame:
        self.data = Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
            aggby=['Condition', 'Time lag'] if self.level == 'Condition' else ['Condition', 'Replicate', 'Time lag'],
            aggdict=self.agg_dict,
            noticequeue=self.noticequeue
        )()

    
    def _get_colors(self) -> dict:

        tags = self.conditions if self.level == 'Condition' else self.replicates

        if self.stock_palette:
            try:
                return self.painter.StockQualPalette(self.data, self.level, self.palette, which=tags)
            except Exception as e:
                Reporter(Level.error, f"Failed to apply palette '{self.palette}'. Falling back to default colors.", details=traceback.format_exc(), noticequeue=self.noticequeue)
                pass

        else:
            try:
                return self.painter.BuildQualPalette(self.data, self.level, which=tags)
            except Exception as e:
                Reporter(Level.error, f"Failed to build color palette for tag '{self.level}'. Falling back to default colors. Error: {e}", details=traceback.format_exc(), noticequeue=self.noticequeue)
                return self.painter.StockQualPalette(self.data, self.level, self.palette, which=tags)
            
        


    def _resolve_color(self, color: Optional[str], idx: int = 0) -> str:
        if color is not None and mcolors.is_color_like(color):
            return color
        return f"C{idx % 10}"

    def _compute_fit_color(self, base_color: str) -> str:
        safe_color = self._resolve_color(base_color, 0)
        base_rgb = mcolors.to_rgb(safe_color)
        hsv = mcolors.rgb_to_hsv(np.array(base_rgb))

        hsv[1] = np.clip(hsv[1] * self.SATURATION_SCALE, self.SATURATION_MIN, self.SATURATION_MAX)
        hsv[2] = np.clip(hsv[2] * self.BRIGHTNESS_SCALE, self.BRIGHTNESS_MIN, hsv[2])

        return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))
    
    def _set_axis_labels(self, ax: plt.Axes):
        ax.set_xlabel(f'Time lag [{Stats.t_unit}]', fontsize=12)
        ax.set_ylabel('MSD [μm²]', fontsize=12)
    
    def _set_ylim(self, ax: plt.Axes, y_vals: np.ndarray):
        miny, maxy = np.nanmin(y_vals), np.nanmax(y_vals)
        lower = miny - 0.5 * abs(miny)
        upper = maxy + 0.05 * abs(maxy) if maxy != 0 else 1.0
        ax.set_ylim(lower, upper)
    
    
    def _add_linear_fit(self, ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                       tag: str, color: str, idx: int, n_tags: int):
        
        # valid, finite, strictly positive for log10
        valid = (
            np.isfinite(x_data) & np.isfinite(y_data) &
            (x_data > 0) & (y_data > 0)
        )
        xv = x_data[valid]
        yv = y_data[valid]
        
        if xv.size < 2:
            return

        # Fit in log10 space: log10(MSD) = a * log10(t) + b
        if self.log_transform:
            xv = np.log10(xv)
            yv = np.log10(yv)

            a, b = np.polyfit(xv, yv, 1)  # slope=a, intercept=b

            # Power-law line in original space
            x_fit = np.logspace(xv.min(), xv.max(), 200)
            y_fit = (10.0 ** b) * (x_fit ** a)
        else:
            a, b = np.polyfit(xv, yv, 1)  # slope=a, intercept=b

            x_fit = np.linspace(xv.min(), xv.max(), 200)
            y_fit = a * x_fit + b

        fit_color = self._compute_fit_color(color)

        ax.plot(
            x_fit, y_fit,
            # linestyle=(0, (3, 1, 1, 1, 1, 1)),
            linestyle='-.',
            color=fit_color,
            linewidth=2,
            zorder=7,
            alpha=0.8
        )

        # Annotation placed using log-domain offsets so it behaves on log axes
        try:
            if self.log_transform:
                lxrange = xv.max() - xv.min()
                lyrange = yv.max() - yv.min()
                if not np.isfinite(lxrange) or lxrange <= 0:
                    lxrange = 1.0
                if not np.isfinite(lyrange) or lyrange <= 0:
                    lyrange = 1.0

                # Place a bit left from xmax (so it's inside the plot)
                x_text_log = xv.max() - 0.03 * lxrange
                # Base y at fitted value at xmax, add small stacked offset per series
                y_base_log = b + a * xv.max()
                v_offset_log = ((idx - (n_tags - 1) / 2.0) * 0.03 * lyrange)
                y_text_log = y_base_log + v_offset_log

                x_text = 10.0 ** x_text_log
                y_text = 10.0 ** y_text_log
            
            else:
                x_text = xv.max() - 0.03 * (xv.max() - xv.min())
                y_base = a * xv.max() + b
                v_offset = ((idx - (n_tags - 1) / 2.0) * 0.03 * (yv.max() - yv.min()))
                y_text = y_base + v_offset


            # Show α; if close to Brownian (α≈1), estimate D from intercept: MSD ≈ 4 D t ⇒ D = 10^b / 4
            slope_text = f"D = {a:.2f} [μm²·s⁻¹]"

            ax.text(
                x_text, y_text, slope_text,
                color=color,
                fontsize=7,
                fontweight='bold',
                verticalalignment='center',
                horizontalalignment='left',
                bbox=dict(facecolor='none', alpha=1, edgecolor='none'),
                zorder=7
            )
        except Exception:
            pass

    def _build_agg_dict(self, prefix: str, ci_stat: str, ci_lvl: int) -> dict:
        

        min     = f"{prefix} MSD min"
        max     = f"{prefix} MSD max"
        mean    = f"{prefix} MSD mean"
        median  = f"{prefix} MSD median"
        sd      = f"{prefix} MSD sd"
        sem     = f"{prefix} MSD sem"
        ci_low  = f"{prefix} MSD {ci_stat} ci{ci_lvl} low"
        ci_high = f"{prefix} MSD {ci_stat} ci{ci_lvl} high"

        agg_dict = {
            min:     'min',
            max:     'max',
            mean:    'mean',
            sd:      'mean',
            median:  'median',
        }

        # Preserve color columns through aggregation
        if f'{self.level} color' in self.data.columns:
            agg_dict[f'{self.level} color'] = 'first'

        if sem in self.data.columns:
            agg_dict.update({
                sem: 'mean',
            })

        if ci_low in self.data.columns and ci_high in self.data.columns:
            agg_dict.update({
                ci_low:  'mean',
                ci_high: 'mean'
            })

        cols = {
            'mean': mean,
            'median': median,
        }

        return agg_dict, cols


    def _check_errors(self) -> None:
        if not self.level == 'Replicate':
            if 'Replicate' not in self.data.columns or not any('{per replicate}' in col for col in self.data.columns.to_list()):
                Reporter(Level.error, f"Cannot separate replicates <- missing 'Replicate' data.", noticequeue=self.noticequeue)
                self.level = 'Condition'
                self.disaggregate = False
        
        if self.level == 'Condition':
            if 'Condition' not in self.data.columns or not any('{per condition}' in col for col in self.data.columns.to_list()):
                Reporter(Level.error, f"Cannot aggregate per condition <- missing 'Condition' data.", noticequeue=self.noticequeue)
                self.level = 'Replicate'
                self.disaggregate = True


def TurnAnglesHeatmap(data: pd.DataFrame, condition: str, replicates: list[str], *, angle_range: int = 15, tlag_range: int = 1, cmap="plasma", **kwargs) -> plt.Figure:
    """ Plot directional change (turning angle) over time lags as a colormesh. """

    noticequeue = kwargs.get('noticequeue', None)
    text_color = kwargs.get('text_color', 'black')
    title = kwargs.get('title', '')

    fig, ax = plt.subplots(figsize=(kwargs.get('figsize', (6, 6))))

    if is_empty(data):
        Reporter(Level.warning, "No data available for plotting.", noticequeue=noticequeue)
        return None

    if isinstance(condition, list | tuple):
        condition = condition[0]
        if len(condition) > 1:
            Reporter(Level.info, "Multiple conditions selecting for colormesh. ", noticequeue=noticequeue)
            condition = condition[0]

    data = Categorizer(
        data=data,
        conditions=[condition],
        replicates=replicates,
        noticequeue=noticequeue
    )()

    cmap = Painter(noticequeue=noticequeue).GetCmap(cmap)

    xvals = data['{per replicate} Directional change mean'].to_numpy()
    yvals = data['Time lag'].to_numpy()
    lags = data['Time lag'].unique()

    if lags.size < 2:
        return None
    
    tlag_range = lags[1] - lags[0]
    
    n = len(replicates)

    x_bins = np.arange(0, 181, angle_range)
    y_bins = np.arange(0, lags.max(), tlag_range)

    H, xe, ye = np.histogram2d(xvals, yvals, bins=[x_bins, y_bins])

    pcm = ax.pcolormesh(
        xe, ye, H.T / n,
        cmap=cmap, shading='auto',
        norm=mcolors.Normalize(vmin=0, vmax=np.nanmax(H / n)),
    )

    # Set axis labels, limits, ticks, and title
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

    # Add colorbar with custom ticks and labels (0–1 fractions)
    cbar = plt.colorbar(pcm, ax=ax, aspect=25, pad=0.04)
    cbar.set_label('Fraction of replicates', color=text_color)
    cbar.ax.tick_params(colors=text_color, width=0.5)

    for spine in cbar.ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(0.5)

    return plt.gcf()