from __future__ import annotations
from typing import *
from itertools import chain
from scipy.stats import gaussian_kde
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import stats
import seaborn as sns
import matplotlib.lines as mlines

from ...categorizer import Painter, Categorizer
from ...various import is_empty
from ...compute.stats import Stats



class SuperPlots:
    """
    Complex superplotting class with methods for visualizing data and its distribution across conditions and replicates.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to plot. Must include columns for conditions, replicates, and the statistic of interest.

    statistic : str
        Name of the column in `data` that contains the values to be plotted.

    conditions : list, optional
        List of conditions for which the data will be plotted.

    replicates : list, optional
        List of replicates for which the data will be plotted.

    ignore_categories : bool, optional
        Whether to ignore the categories (conditions and replicates) when plotting. 
        When set to True, all categories are included in the graph.

    ci_statistic : str, optional
        One of 'mean' or 'median', specifying for which statistic will the confidence interval be computed.

    bootstrap_method : str, optional
        Method for computing bootstrap confidence intervals. One of 'percentile' (default), 'basic', 'BCa', or 'studentized'.

    confidence_level : float, optional
        The confidence level for the confidence intervals. Default is 95.

    n_resamples : int, optional
        The number of bootstrap resamples to use for computing confidence intervals. Default is 1000.
    """


    def __init__(
            self, 
            data: pd.DataFrame, 
            statistic: str, 
            conditions: list = None, 
            replicates: list = None, 
            *,
            ignore_categories: bool = True,
            **kwargs
        ):

        data = data.copy()
        self.data = data.reset_index(drop=True)
        self.statistic = statistic
        self.conditions = conditions
        self.replicates = replicates

        self.noticequeue = kwargs.get('noticequeue', None)

        if self._guard():
            return
        
        self.monochrome = kwargs.get('monochrome', False)
        self.monochrome_color = kwargs.get('monochrome_color', 'cornflowerblue')

        if not ignore_categories:
            self._arrange_data() 

        self.ci_statistic = kwargs.get('ci_statistic', 'mean')
        self.bootstrap_method = kwargs.get('bootstrap_method', 'percentile')
        self.confidence_level = kwargs.get('confidence_level', 95)
        self.n_resamples = kwargs.get('n_resamples', 1000)

        self.stats = Stats()

        self.instance_kwargs = kwargs


    def hybrid(self, **kwargs) -> plt.Figure:

        if kwargs.get('error_type', 'sd') == 'std':
            kwargs['error_type'] = 'sd'

        fig, ax, kwargs = self._init_orientation(**kwargs)

        categories = self._spacing(kde=kwargs.get('kde', False))
        self.condition_stats, self.replicate_stats = self._get_statistics(categories, err_agg_stat=kwargs.get('error_type', 'sd'))

        palette = self._make_palette(use_stock_palette=kwargs.get('use_stock_palette', True), palette=kwargs.get('palette', 'tab10'))

        fwd_kwargs = {k: v for k, v in kwargs.items() if k not in ('palette',)}

        data_vals = self.data[self.statistic].dropna()
        if not data_vals.empty:
            data_min, data_max = data_vals.min(), data_vals.max()
            margin = (data_max - data_min) * 0.1 if data_max != data_min else 1
            if kwargs.get('orient', 'v') == 'v':
                ax.set_ylim(data_min - margin, data_max + margin)
                ax.set_xlim(-0.75, len(categories) - 0.25)
            else:
                ax.set_xlim(data_min - margin, data_max + margin)
                ax.set_ylim(-0.75, len(categories) - 0.25)
                

        if fwd_kwargs.get('scatter', True):
            if fwd_kwargs.get('scatter_style', 'sina') == 'sina':
                self._sina(ax, palette, categories, **fwd_kwargs)
            else:
                self._swarms(ax, palette, **fwd_kwargs)

        if fwd_kwargs.get('violins', True):
            self._violins(ax, **fwd_kwargs)

        if fwd_kwargs.get('kde', False):
            self._kdes(ax, palette, categories, **fwd_kwargs)

        if fwd_kwargs.get('skeleton', True):
            self._skeleton(ax, categories, **fwd_kwargs)

        if not self.monochrome:
            if fwd_kwargs.get('rep_means', True):
                self._rep_markers(ax, palette, stat='mean', **fwd_kwargs)

            if fwd_kwargs.get('rep_medians', True):
                self._rep_markers(ax, palette, stat='median', **fwd_kwargs)

        if fwd_kwargs.get('legend', True):
            self._legend(ax, palette, **fwd_kwargs)
        else:
            try:
                plt.legend().remove()
            except Exception:
                pass

        plt.title(fwd_kwargs.get('title', ''), fontsize=fwd_kwargs.get('title_fontsize', 14), color=fwd_kwargs.get('text_color', 'black'), **fwd_kwargs.get('title_kwargs', {}))
        plt.xlabel(fwd_kwargs.get('xlabel'), fontsize=fwd_kwargs.get('label_fontsize', 12), color=fwd_kwargs.get('text_color', 'black'), **fwd_kwargs.get('xlabel_kwargs', {}))
        plt.ylabel(fwd_kwargs.get('ylabel'), fontsize=fwd_kwargs.get('label_fontsize', 12), color=fwd_kwargs.get('text_color', 'black'), **fwd_kwargs.get('ylabel_kwargs', {}))
        sns.despine(top=fwd_kwargs.get('despine', True), right=fwd_kwargs.get('despine', True), bottom=False, left=False)
        plt.tick_params(axis='y', **fwd_kwargs.get('ytickparams', {}))
        plt.tick_params(axis='x', **fwd_kwargs.get('xtickparams', {}))

        ax.set_facecolor(fwd_kwargs.get('background', 'white'))
        fig.set_facecolor(fwd_kwargs.get('figure_background', 'whitesmoke'))
        fig.tight_layout()

        if fwd_kwargs.get('grid', False):
            plt.grid(True, axis=fwd_kwargs.get('grid_axis', 'y'), color=fwd_kwargs.get('grid_color', 'lightgrey'), linewidth=fwd_kwargs.get('grid_linewidth', 1), alpha=fwd_kwargs.get('grid_alpha', 0.25), linestyle=fwd_kwargs.get('grid_linestyle', '-'))
        else:
            plt.grid(False)
        
        return plt.gcf()

    def superviolins(self, **kwargs) -> plt.Figure:
        """
        Superviolin plot: KDE-stacked stripes per replicate within each condition,
        with skeleton summary stats overlay.

        kwargs
        ------
        orient : str
            'v' (vertical, default) or 'h' (horizontal).
        palette : str
            Stock qualitative palette name (default 'Accent').
        use_stock_palette : bool
            Use a stock palette (True) or build from assigned colors (False).
        bw : float | None
            KDE bandwidth (Scott's rule if None).
        total_width : float
            Half-width of each superviolin (default 0.8).
        outline_lw : float
            Outline linewidth of each superviolin (default 1).
        sep_lw : float
            Separating line width between replicate stripes (default 0).
        bullet_lw : float
            Replicate bullet outline width (default 0.75).
        bullet_size : int
            Replicate bullet size (default 5, internally scaled ×10).
        middle_vals : str
            Replicate central measure: 'mean' or 'median' (default 'mean').
        error_type : str
            Error bar method: 'sd', 'sem', or 'ci' (default 'sd').
        cond_mean : bool
            Show condition mean line (default True).
        cond_median : bool
            Show condition median line (default False).
        error : bool
            Show error bars (default True).
        legend : bool
            Show legend (default True).
        All other kwargs from hybrid (_init_orientation, _legend, _skeleton) are supported.
        """

        if kwargs.get('error_type', 'sd') == 'std':
            kwargs['error_type'] = 'sd'

        fig, ax, kwargs = self._init_orientation(**kwargs)

        categories = self.conditions  # no spacers for superviolins
        self.data['Condition'] = pd.Categorical(
            values=self.data['Condition'], categories=categories, ordered=True
        )

        self.condition_stats, self.replicate_stats = self._get_statistics(
            categories, err_agg_stat=kwargs.get('error_type', 'sd')
        )

        palette = self._make_palette(
            use_stock_palette=kwargs.get('use_stock_palette', True),
            palette=kwargs.get('palette', 'Accent')
        )

        # Build color list indexed by replicate order
        unique_reps = list(self.data['Replicate'].unique())
        colors = [palette.get(r, Painter().GenerateRandomColor()) for r in unique_reps]

        bw = kwargs.get('bw', None)
        total_width = kwargs.get('total_width', 0.8)
        outline_lw = kwargs.get('outline_lw', 1)
        sep_lw = kwargs.get('sep_lw', 0)
        bullet_lw = kwargs.get('bullet_lw', 0.75)
        bullet_size = kwargs.get('bullet_size', 5) * 10
        middle_vals = kwargs.get('middle_vals', 'mean').lower()
        rep_markers = kwargs.get('rep_markers', True)
        orient = kwargs.get('orient', 'v')

        # Compute KDE data for each condition
        subgroups = list(categories)
        subgroup_dict = {g: {"norm_wy": [], "px": []} for g in subgroups}
        self._sv_compute_kde(subgroups, unique_reps, subgroup_dict, bw)

        # Plot each condition's stacked stripes
        _on_legend = []
        tick_positions = []
        tick_labels = []

        for i, cond in enumerate(subgroups):
            axis_point = i * 2
            tick_positions.append(axis_point)
            tick_labels.append(cond)

            sub = self.data[self.data['Condition'] == cond]
            if sub.empty:
                continue

            mid_vals = sub.groupby('Replicate', as_index=False).agg(
                {self.statistic: middle_vals}
            )

            self._sv_single_subgroup(
                ax, cond, axis_point, mid_vals,
                unique_reps, subgroup_dict, colors,
                total_width=total_width, outline_lw=outline_lw,
                sep_lw=sep_lw, bullet_lw=bullet_lw,
                bullet_size=bullet_size, orient=orient,
                _on_legend=_on_legend, rep_markers=rep_markers
            )

        # Skeleton overlay — remap condition positions to 0,2,4,... scale
        # We build a temporary condition_stats aligned to the 0,2,4... x-axis
        self._sv_skeleton(
            ax, subgroups, tick_positions, kwargs
        )

        # Tick labels
        if orient == 'v':
            plt.xticks(tick_positions, tick_labels, rotation=345)
        else:
            plt.yticks(tick_positions, tick_labels, rotation=0)

        # Legend (reuses the shared _legend method)
        if kwargs.get('legend', True):
            kwargs_for_legend = {k: v for k, v in kwargs.items() if k != 'palette'}
            self._legend(ax, palette, **kwargs_for_legend)
        else:
            try:
                plt.legend().remove()
            except Exception:
                pass

        # Cosmetics
        plt.title(kwargs.get('title', ''), fontsize=kwargs.get('title_fontsize', 14),
                  color=kwargs.get('text_color', 'black'), **kwargs.get('title_kwargs', {}))
        plt.xlabel(kwargs.get('xlabel'), fontsize=kwargs.get('label_fontsize', 12),
                   color=kwargs.get('text_color', 'black'), **kwargs.get('xlabel_kwargs', {}))
        plt.ylabel(kwargs.get('ylabel'), fontsize=kwargs.get('label_fontsize', 12),
                   color=kwargs.get('text_color', 'black'), **kwargs.get('ylabel_kwargs', {}))
        sns.despine(top=kwargs.get('despine', True), right=kwargs.get('despine', True),
                    bottom=False, left=False)
        plt.tick_params(axis='y', **kwargs.get('ytickparams', {}))
        plt.tick_params(axis='x', **kwargs.get('xtickparams', {}))

        ax.set_facecolor(kwargs.get('background', 'white'))
        fig.set_facecolor(kwargs.get('figure_background', 'whitesmoke'))
        fig.tight_layout()

        if kwargs.get('grid', False):
            plt.grid(True, axis=kwargs.get('grid_axis', 'y'),
                     color=kwargs.get('grid_color', 'lightgrey'),
                     linewidth=kwargs.get('grid_linewidth', 1),
                     alpha=kwargs.get('grid_alpha', 0.25),
                     linestyle=kwargs.get('grid_linestyle', '-'))
        else:
            plt.grid(False)

        return plt.gcf()


    # ================================================================
    # SUPERVIOLINS — internal helpers
    # ================================================================

    def _sv_compute_kde(self, subgroups, unique_reps, subgroup_dict, bw):
        """Fit KDEs per replicate per condition, stack and normalize."""

        for group in subgroups:
            px = []
            norm_wy = []
            min_cuts = []
            max_cuts = []

            for r in unique_reps:
                sub = self.data[
                    (self.data['Replicate'] == r) & (self.data['Condition'] == group)
                ][self.statistic].dropna()
                if sub.empty:
                    continue
                min_cuts.append(sub.min())
                max_cuts.append(sub.max())

            # If no valid data for this group, fill with empties and skip
            if not min_cuts or not max_cuts:
                for r in unique_reps:
                    norm_wy.append(np.array([]))
                    px.append(np.array([]))
                subgroup_dict[group]["norm_wy"] = norm_wy
                subgroup_dict[group]["px"] = px
                continue

            min_cuts = sorted(min_cuts)
            max_cuts = sorted(max_cuts)

            points1 = list(np.linspace(np.nanmin(min_cuts), np.nanmax(max_cuts), num=128))
            points = sorted(list(set(min_cuts + points1 + max_cuts)))

            for r in unique_reps:
                try:
                    sub = self.data[
                        (self.data['Replicate'] == r) & (self.data['Condition'] == group)
                    ][self.statistic]
                    arr = np.array(sub)
                    arr = arr[~np.isnan(arr)]

                    kde = gaussian_kde(arr, bw_method=bw)
                    kde_points = kde.evaluate(points)

                    idx_min = min_cuts.index(arr.min())
                    idx_max = max_cuts.index(arr.max())
                    if idx_min > 0:
                        for p in range(idx_min):
                            kde_points[p] = 0
                    for idx in range(idx_max - len(max_cuts), 0):
                        if idx_max - len(max_cuts) != -1:
                            kde_points[idx + 1] = 0

                    kde_wo_nan = self._sv_interpolate_nan(kde_points)
                    points_wo_nan = self._sv_interpolate_nan(np.array(points, dtype=float))

                    area = 30
                    kde_wo_nan = (area / sum(kde_wo_nan)) * kde_wo_nan

                    norm_wy.append(kde_wo_nan)
                    px.append(points_wo_nan)
                except ValueError:
                    norm_wy.append([])
                    px.append(self._sv_interpolate_nan(np.array(points, dtype=float)))

            px = np.array(px)

            length = max([len(e) for e in norm_wy])
            norm_wy = [a if len(a) > 0 else np.zeros(length) for a in norm_wy]
            norm_wy = np.array(norm_wy)
            norm_wy = np.cumsum(norm_wy, axis=0)
            try:
                norm_wy = norm_wy / np.nanmax(norm_wy)
            except ValueError:
                pass

            subgroup_dict[group]["norm_wy"] = norm_wy
            subgroup_dict[group]["px"] = px


    @staticmethod
    def _sv_interpolate_nan(arr):
        """Interpolate NaN values in a KDE coordinate array."""
        arr = np.asarray(arr, dtype=float)
        nan_idx = np.where(np.isnan(arr))
        if nan_idx[0].size == 0:
            return arr

        # If the array has fewer than 2 elements, we can't compute diffs
        if len(arr) < 2:
            arr[np.isnan(arr)] = 0.0
            return arr

        diffs = np.diff(arr, axis=0)
        median_val = np.nanmedian(diffs)
        # If median_val is itself NaN (all diffs are NaN), fall back to 0
        if np.isnan(median_val):
            median_val = 0.0

        idx = nan_idx[0][0]
        if idx + 1 < len(arr) and not np.isnan(arr[idx + 1]):
            arr[idx] = arr[idx + 1] - median_val
        elif idx - 1 >= 0 and not np.isnan(arr[idx - 1]):
            arr[idx] = arr[idx - 1] + median_val
        else:
            arr[idx] = 0.0
        return arr


    @staticmethod
    def _sv_find_nearest(array, value):
        """Find the value in array nearest to value."""
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    def _sv_single_subgroup(
        self, ax, group, axis_point, mid_df,
        unique_reps, subgroup_dict, colors,
        *, total_width, outline_lw, sep_lw,
        bullet_lw, bullet_size, orient, _on_legend,
        rep_markers
    ):
        """Plot stacked KDE stripes for one condition."""

        norm_wy = subgroup_dict[group]["norm_wy"]
        px = subgroup_dict[group]["px"]

        if len(norm_wy) == 0 or len(px) == 0:
            return

        # Guard: if the last KDE row is all zeros or empty, skip this subgroup
        if np.asarray(norm_wy[-1]).size == 0 or np.nanmax(np.abs(np.asarray(norm_wy[-1]))) == 0:
            return

        right_sides = np.array([norm_wy[-1] * -1 + i * 2 for i in norm_wy])

        new_wy = []
        for i in range(len(px)):
            if i == 0:
                newline = np.append(norm_wy[-1] * -1, np.flipud(right_sides[i]))
            else:
                newline = np.append(right_sides[i - 1], np.flipud(right_sides[i]))
            new_wy.append(newline)
        new_wy = np.array(new_wy)

        outline_vals = np.append(px[-1], np.flipud(px[-1]))
        append_param = np.flipud(norm_wy[-1]) * -1
        outline_pos = np.append(norm_wy[-1], append_param) * total_width + axis_point

        # Guard: if outline arrays ended up empty, skip
        if outline_pos.size == 0 or outline_vals.size == 0:
            return

        # Close the outline if endpoints don't match
        if outline_pos[0] != outline_pos[-1]:
            xval = round(outline_pos[0], 4)
            yval = outline_vals[0]
            outline_pos = np.insert(outline_pos, 0, xval)
            outline_pos = np.insert(outline_pos, outline_pos.size, xval)
            outline_vals = np.insert(outline_vals, 0, yval)
            outline_vals = np.insert(outline_vals, outline_vals.size, yval)

        for i, rep in enumerate(unique_reps):
            reshaped_vals = np.append(px[i - 1], np.flipud(px[i - 1]))
            mid_val = mid_df[mid_df['Replicate'] == rep][self.statistic].values
            reshaped_pos = new_wy[i] * total_width + axis_point

            lbl = "" if rep in _on_legend else rep
            if lbl:
                _on_legend.append(rep)

            # Draw stripe and separator
            if orient == 'v':
                ax.plot(reshaped_pos, reshaped_vals, c="k", linewidth=sep_lw)
                ax.fill(reshaped_pos, reshaped_vals, color=colors[i], label=lbl, linewidth=sep_lw)
            else:
                ax.plot(reshaped_vals, reshaped_pos, c="k", linewidth=sep_lw)
                ax.fill(reshaped_vals, reshaped_pos, color=colors[i], label=lbl, linewidth=sep_lw)

            # Replicate bullet
            if rep_markers:
                if mid_val.size > 0:
                    rx = pd.to_numeric(np.asarray(reshaped_vals).reshape(-1), errors='coerce')
                    arr_finite = rx[np.isfinite(rx)]
                    if arr_finite.size == 0:
                        continue
                    nearest = self._sv_find_nearest(arr_finite, float(mid_val[0]))
                    idx = np.where(rx == nearest)
                    if idx[0].size < 2:
                        j = int(np.argmin(np.abs(rx - nearest)))
                        idx = (np.array([max(j - 1, 0), j]),)
                    pos_vals = reshaped_pos[idx]
                    pos_val = pos_vals[0] + ((pos_vals[1] - pos_vals[0]) / 2)

                    if orient == 'v':
                        ax.scatter(pos_val, mid_val[0], facecolors=colors[i],
                                edgecolors="black", linewidth=bullet_lw,
                                zorder=10, marker="o", s=bullet_size)
                    else:
                        ax.scatter(mid_val[0], pos_val, facecolors=colors[i],
                                edgecolors="black", linewidth=bullet_lw,
                                zorder=10, marker="o", s=bullet_size)

        # Draw outline
        if orient == 'v':
            ax.plot(outline_pos, outline_vals, color="black", linewidth=outline_lw)
        else:
            ax.plot(outline_vals, outline_pos, color="black", linewidth=outline_lw)


    def _sv_skeleton(self, ax, subgroups, tick_positions, kwargs):
        """Draw mean/median lines and error bars on the superviolin x-axis (0, 2, 4, ...)."""

        orient = kwargs.get('orient', 'v')
        error_type = kwargs.get('error_type', 'sd')

        for i, cond in enumerate(subgroups):
            pos = tick_positions[i]
            row = self.condition_stats[self.condition_stats['Condition'] == cond]
            if row.empty:
                continue

            cond_mean = row['mean'].values[0]
            cond_median = row['median'].values[0]
            # cond_count = row['count'].values[0]

            # Error value
            if error_type == 'sem' and 'sem' in row.columns:
                err = row['sem'].values[0]
            elif error_type == 'ci' and 'ci' in row.columns:
                err = row['ci'].values[0]
            else:
                err = row['std'].values[0]

            median_width = 0.4  # half-span for skeleton lines

            # Pick the central value for error bars
            central = cond_mean  # default
            if error_type == 'ci' and self.ci_statistic == 'median':
                central = cond_median

            if orient == 'v':
                lines_fn = ax.hlines
                errbar_ax = 'y'
            else:
                lines_fn = ax.vlines
                errbar_ax = 'x'

            # Mean line
            if kwargs.get('cond_mean', True) and np.isfinite(cond_mean):
                span = kwargs.get('mean_span', median_width / 1.5)
                lines_fn(
                    cond_mean, pos - span, pos + span,
                    colors=kwargs.get('mean_color', 'black'),
                    linestyles=kwargs.get('mean_ls', '-'),
                    linewidths=kwargs.get('line_width', 1),
                    zorder=20
                )

            # Median line
            if kwargs.get('cond_median', False) and np.isfinite(cond_median):
                span = kwargs.get('median_span', median_width / 1.5)
                lines_fn(
                    cond_median, pos - span, pos + span,
                    colors=kwargs.get('median_color', 'black'),
                    linestyles=kwargs.get('median_ls', '--'),
                    linewidths=kwargs.get('line_width', 1),
                    zorder=20
                )

            # Error bars
            if kwargs.get('error', True) and np.isfinite(err) and np.isfinite(central):
                upper = central + err
                lower = central - err
                cap_span = kwargs.get('errorbar_capsize_span', median_width / 4.5)
                elw = float(kwargs.get('line_width', 1))
                ecolor = kwargs.get('errorbar_color', 'black')

                if orient == 'v':
                    # Caps
                    ax.hlines([upper, lower], pos - cap_span, pos + cap_span,
                              colors=ecolor, linewidths=elw, zorder=20)
                    # Vertical connector
                    ax.vlines(pos, lower, upper, colors=ecolor, linewidths=elw, zorder=20)
                else:
                    ax.vlines([upper, lower], pos - cap_span, pos + cap_span,
                              colors=ecolor, linewidths=elw, zorder=20)
                    ax.hlines(pos, lower, upper, colors=ecolor, linewidths=elw, zorder=20)


    # ================================================================
    # SHARED INFRASTRUCTURE (unchanged from your original)
    # ================================================================

    def _init_orientation(self, **kwargs) -> tuple[plt.Figure, plt.Axes]:

        m_label = Stats().stat_units(col=self.statistic, time_unit=kwargs.get('t_unit', None))
        if m_label is not None:
            m_label = f" [{m_label}]"
        else:
            m_label = ""

        if kwargs.get('orient', 'v') == 'v':
            height = kwargs.get('fig_height', 9)
            width = kwargs.get('fig_width', 15)
            xlabel = kwargs.get('xlabel', 'Condition')
            ylabel = kwargs.get('ylabel', f"{self.statistic}{m_label}")
            xtickparams = dict(length=7, width=1.5, direction='out', color='black', rotation=345)
            ytickparams = dict(length=5, width=1.5, direction='out', color='black')
        else:
            height = kwargs.get('fig_width', 15)
            width = kwargs.get('fig_height', 9)
            xlabel = kwargs.get('xlabel', f"{self.statistic}{m_label}")
            ylabel = kwargs.get('ylabel', 'Condition')
            xtickparams = dict(length=5, width=1.5, direction='out', color='black')
            ytickparams = dict(length=7, width=1.5, direction='out', color='black', rotation=45)

        kwargs['xlabel'] = xlabel
        kwargs['ylabel'] = ylabel
        kwargs['xtickparams'] = xtickparams
        kwargs['ytickparams'] = ytickparams

        fig, ax = plt.subplots(figsize=(width, height))
  
        return fig, ax, kwargs
  

    def _guard(self):

        if is_empty(self.data):
            Reporter(Level.warning, "Empty DataFrame.", details="The input DataFrame is empty. No plot will be generated.")
            return True
        
        if self.statistic not in self.data.columns:
            Reporter(Level.error, "Missing statistic column.", details=f"The specified statistic '{self.statistic}' is not a column in the input DataFrame.")
            return True
        
        return False
    

    def _arrange_data(self):
        
        self.data = Categorizer(self.data, self.conditions, self.replicates)()

        self.condition_stats = self.data.groupby('Condition', observed=False)[self.statistic].agg(['mean', 'median', 'count']).reset_index()

        if not self.monochrome:
            self.replicate_stats = self.data.groupby(['Condition', 'Replicate'], observed=False)[self.statistic].agg(['mean', 'median']).reset_index()

    
    def _spacing(self, kde: bool = False) -> list:

        if kde:
            categories = ["spacer_0"] + list(
                chain.from_iterable(
                    (cond, f"spacer_{i+1}") if i < len(self.conditions) - 1 else (cond,)
                     for i, cond in enumerate(self.conditions)
                ))
            
            self.data['Condition'] = pd.Categorical(values=self.data['Condition'], categories=categories, ordered=True)
            return categories
        
        else:
            self.data['Condition'] = pd.Categorical(values=self.data['Condition'], categories=self.conditions, ordered=True)
            return self.conditions
        

    def _get_statistics(self, categories: list, *, err_agg_stat: str = 'sd') -> tuple[pd.DataFrame, pd.DataFrame | None]:
        cond_stats = self.data.groupby('Condition', observed=False)[self.statistic].agg(
            ['mean', 'median', 'std', 'count']
        ).reindex(categories).reset_index()

        if err_agg_stat == 'sem':
            cond_stats['sem'] = cond_stats['std'] / np.sqrt(cond_stats['count'])
        elif err_agg_stat == 'ci':
            ci_results = []
            for cond in categories:
                group = self.data[self.data['Condition'] == cond][self.statistic].dropna().values
                if len(group) >= 2:
                    ci = self.stats.ci(
                        group,
                        bootstrap_method=self.bootstrap_method,
                        n_resamples=self.n_resamples,
                        confidence_level=self.confidence_level,
                        ci_statistic=self.ci_statistic
                    )
                    ci_results.append(ci)
                else:
                    ci_results.append((np.nan, np.nan))
            cond_stats['ci_low'] = [c[0] for c in ci_results]
            cond_stats['ci_high'] = [c[1] for c in ci_results]
            cond_stats['ci'] = (cond_stats['ci_high'] - cond_stats['ci_low']) / 2

        if not self.monochrome and 'Replicate' in self.data.columns:
            rep_stats = (
                self.data.groupby(['Condition', 'Replicate'], observed=False)[self.statistic]
                .agg(['mean', 'median']).reset_index()
            )
        else:
            rep_stats = None

        return (cond_stats, rep_stats)


    def _make_palette(self, use_stock_palette: bool = True, palette: str = 'tab10') -> dict:

        painter = Painter(noticequeue=self.noticequeue)

        if self.monochrome:
            color = self.monochrome_color
            return {cond: color for cond in self.conditions}

        if use_stock_palette: 
            return painter.StockQualPalette(self.data, tag='Replicate', palette=palette)
        else:
            return painter.BuildQualPalette(self.data, tag='Replicate')


    def _sina(self, ax: plt.Axes, palette: dict, categories: list, **kwargs):

        condition_data = {}
        global_density_max = 0.0

        cond_counts = {}
        for cond in categories:
            group = self.data[self.data['Condition'] == cond]
            cond_counts[cond] = group[self.statistic].dropna().shape[0]
        max_count = max(cond_counts.values()) if cond_counts else 0

        def _draw_scatter(cat_pos, rep_vals, jitter, color):
            scatterx = cat_pos + jitter if kwargs.get('orient', 'v') == 'v' else rep_vals
            scattery = rep_vals if kwargs.get('orient', 'v') == 'v' else cat_pos + jitter

            ax.scatter(
                scatterx, scattery,
                s=kwargs.get('scatter_size', 1),
                color=color,
                edgecolor=kwargs.get('scatter_outline_color', 'black') if kwargs.get('swarm_outline', True) else 'none',
                linewidth=kwargs.get('scatter_outline_width', 0.5) if kwargs.get('swarm_outline', True) else 0,
                alpha=kwargs.get('scatter_alpha', 0.75),
                zorder=1,
            )

        for i, cond in enumerate(categories):
            group = self.data[self.data['Condition'] == cond]
            if group.empty or len(group) < 3:
                continue

            vals = group[self.statistic].dropna().values
            if len(vals) < 2:
                continue

            kde = gaussian_kde(vals)
            densities = kde(vals)

            scaled_densities = densities
            global_density_max = max(global_density_max, scaled_densities.max())

            local_max = scaled_densities.max()
            condition_data[i] = (vals, scaled_densities, group, local_max)

        for i, (vals, scaled_densities, group, local_max) in condition_data.items():

            if kwargs.get('density_norm', 'area') == 'area':
                scaled_densities = (scaled_densities / global_density_max)
                width_scale = 1.0
            else:
                scaled_densities = (scaled_densities / (local_max))
                cond = categories[i]
                width_scale = (cond_counts.get(cond, 0) / max_count) if max_count > 0 else 1.0

            base_width   = kwargs.get('sina_width', 0.5)
            violin_width = kwargs.get('violin_width', 1.0)
            eff_width    = base_width * width_scale * (violin_width / 1.0)
            eff_range    = kwargs.get('sina_range', 0.6) * width_scale * (violin_width / 1.0)

            if not self.monochrome:
                for rep in group['Replicate'].unique():
                    rep_mask = group['Replicate'].values == rep
                    rep_vals = vals[rep_mask[:len(vals)]]
                    rep_scaled = scaled_densities[rep_mask[:len(vals)]]

                    if len(rep_vals) == 0:
                        continue

                    jitter = np.random.uniform(-eff_width, eff_width, size=len(rep_vals)) * rep_scaled
                    jitter = np.clip(jitter, -eff_range, eff_range)

                    color = palette.get(rep, 'grey')

                    _draw_scatter(i, rep_vals, jitter, color)
            else:
                jitter = np.random.uniform(-eff_width, eff_width, size=len(vals)) * scaled_densities
                jitter = np.clip(jitter, -eff_range, eff_range)

                cond = categories[i]
                color = palette.get(cond, self.monochrome_color)

                _draw_scatter(i, vals, jitter, color)
     

    def _swarms(self, ax: plt.Axes, palette: dict, **kwargs):

        if self.monochrome:
            color = self.monochrome_color
            hue = None
            palette = None
        else:
            color = None
            hue = 'Replicate'
            palette = palette
            
        sns.swarmplot(
            data=self.data,
            x='Condition' if kwargs.get('orient', 'v') == 'v' else self.statistic,
            y=self.statistic if kwargs.get('orient', 'v') == 'v' else 'Condition',
            color=color,
            hue=hue,
            palette=palette,
            warn_thresh=0.1,
            orient=kwargs.get('orient', 'v'),
            size=kwargs.get('scatter_size', 2.5),
            edgecolor=kwargs.get('scatter_outline_color', 'black') if kwargs.get('swarm_outline', True) else None,
            alpha=kwargs.get('scatter_alpha', 0.5),
            legend=False, zorder=1, ax=ax
        )


    def _violins(self, ax: plt.Axes, **kwargs):

        sns.violinplot(
            data=self.data,
            x='Condition' if kwargs.get('orient', 'v') == 'v' else self.statistic,
            y=self.statistic if kwargs.get('orient', 'v') == 'v' else 'Condition',
            orient=kwargs.get('orient', 'v'),
            color=kwargs.get('violin_fill_color', 'whitesmoke') if kwargs.get('violin_fill', True) else 'none',
            edgecolor=kwargs.get('violin_outline_color', 'lightgrey') if kwargs.get('violin_outline', True) else None,
            linewidth=kwargs.get('violin_outline_width', 1) if kwargs.get('violin_outline', True) else 0,
            inner=None, gap=0.1, cut=kwargs.get('violin_cut', 0),
            width=kwargs.get('violin_width', 1),
            alpha=kwargs.get('violin_alpha', 0.5),
            density_norm=kwargs.get('density_norm', 'count'),
            zorder=0, ax=ax
        )

    
    def _kdes(self, ax: plt.Axes, palette: dict, categories: list, **kwargs):

        scale_ax_min, scale_ax_max = ax.get_ylim() if kwargs.get('orient', 'v') == 'v' else ax.get_xlim()

        for i, cond in enumerate(self.conditions):
            group_df = self.data[self.data['Condition'] == cond]
            if group_df.empty:
                continue

            cat_pos = 2 * i
            offset = 0.25
            inset_height = scale_ax_max - (scale_ax_max - group_df[self.statistic].max()) + abs(scale_ax_min * 2)

            if kwargs.get('orient', 'v') == 'v':
                _bounds = [cat_pos - offset, scale_ax_min, kwargs.get('kde_inset_width', 0.5), inset_height]
            else:
                _bounds = [scale_ax_min, cat_pos - offset, inset_height, kwargs.get('kde_inset_width', 0.5)]

            inset_ax = ax.inset_axes(_bounds, transform=ax.transData, zorder=0, clip_on=True)
            
            if self.monochrome:
                color = self.monochrome_color
                hue = None
                palette = None
            else:
                color = None
                hue = 'Replicate'
                palette = palette


            sns.kdeplot(
                data=group_df, 
                y=self.statistic if kwargs.get('orient', 'v') == 'v' else None,
                x=None if kwargs.get('orient', 'v') == 'v' else self.statistic,
                hue=hue, palette=palette, color=self.monochrome_color,
                fill=kwargs.get('kde_fill', False), alpha=kwargs.get('kde_alpha', 1) if kwargs.get('kde_fill', True) else 1,
                lw=kwargs.get('kde_outline_lw', 1.5) if kwargs.get('kde_outline', True) else 0,
                ax=inset_ax, legend=False, zorder=0, clip=(scale_ax_min, scale_ax_max)
            )

            inset_ax.invert_xaxis() if kwargs.get('orient', 'v') == 'v' else None
            inset_ax.set_xticks([]); inset_ax.set_yticks([])
            inset_ax.set_xlabel(''); inset_ax.set_ylabel('')
            sns.despine(ax=inset_ax, left=True, bottom=True, top=True, right=True)

        ticks = [i for i, lbl in enumerate(categories) if not str(lbl).startswith('spacer')]
        labels = [categories[i] for i in ticks]
        
        if kwargs.get('orient', 'v') == 'v': 
            plt.xticks(ticks=ticks, labels=labels)
            plt.xlim(-0.75, len(categories))
        else:
            plt.yticks(ticks=ticks, labels=labels)
            plt.ylim(-0.75, len(categories))


    def _rep_markers(self, ax: plt.Axes, palette: dict, stat: str = 'median', **kwargs):

        if self.monochrome or self.replicate_stats is None:
            return

        plot_data = self.replicate_stats.copy()
        plot_data['Replicate'] = plot_data['Replicate'].astype(str)

        palette_norm = {str(k): v for k, v in palette.items()}
        for rep in plot_data['Replicate'].unique():
            if rep not in palette_norm:
                palette_norm[rep] = 'grey'

        sns.scatterplot(
            data=plot_data,
            x='Condition' if kwargs.get('orient', 'v') == 'v' else stat,
            y=stat if kwargs.get('orient', 'v') == 'v' else 'Condition',
            hue='Replicate', palette=palette_norm,
            edgecolor=kwargs.get(f'rep_{stat}_outline_color', 'black') if kwargs.get(f'rep_{stat}_outline', True) else None,
            s=kwargs.get(f'rep_{stat}_size', 90),
            alpha=kwargs.get(f'rep_{stat}_alpha', 1),
            linewidth=kwargs.get(f'rep_{stat}_outline_width', 0.75) if kwargs.get(f'rep_{stat}_outline', True) else 0,
            legend=False, zorder=4, ax=ax
        )

        if all([not kwargs.get('kde', False), kwargs.get(f'connect_rep_{stat}', True)]):
            self._connect_rep_markers(ax, palette_norm, stat, **kwargs)


    def _connect_rep_markers(self, ax: plt.Axes, palette: dict, stat: str, **kwargs):

        if self.monochrome or self.replicate_stats is None:
            return

        rep_df = self.replicate_stats.copy()
        rep_df['Replicate'] = rep_df['Replicate'].astype(str)
        palette_norm = {str(k): v for k, v in palette.items()}

        orient = kwargs.get('orient', 'v')
        line_alpha = kwargs.get(f'connect_rep_{stat}_alpha', 0.75)
        line_width = kwargs.get(f'connect_rep_{stat}_lw', 1.0)
        line_style = kwargs.get(f'connect_rep_{stat}_ls', '-')

        ordered_conditions = [c for c in self.conditions]
        cond_to_pos = {}
        cat_codes = rep_df['Condition'].cat.categories.tolist()
        for cond in ordered_conditions:
            if cond in cat_codes:
                cond_to_pos[cond] = cat_codes.index(cond)

        for rep in rep_df['Replicate'].unique():
            rep_data = rep_df[rep_df['Replicate'] == rep].copy()
            rep_data = rep_data[rep_data['Condition'].isin(ordered_conditions)]
            rep_data = rep_data.sort_values(
                'Condition',
                key=lambda col: col.map({c: i for i, c in enumerate(ordered_conditions)})
            )

            if len(rep_data) < 2:
                continue

            positions = [cond_to_pos[c] for c in rep_data['Condition']]
            values = rep_data[stat].values
            color = palette_norm.get(str(rep), 'grey')

            for j in range(len(positions) - 1):
                if orient == 'v':
                    plotx = [positions[j], positions[j + 1]]
                    ploty = [values[j], values[j + 1]]
                else:
                    plotx = [values[j], values[j + 1]]
                    ploty = [positions[j], positions[j + 1]] 

                ax.plot(plotx, ploty, color=color, alpha=line_alpha,
                        linewidth=line_width, linestyle=line_style, zorder=3)


    def _skeleton(self, ax: plt.Axes, categories: list, **kwargs):

        n = len(categories)
        centers = np.arange(n)

        is_spacer = self.condition_stats['Condition'].astype(str).str.startswith('spacer')
        valid = ~is_spacer

        cond_mean = self.condition_stats.loc[valid, 'mean'].to_numpy()
        cond_median = self.condition_stats.loc[valid, 'median'].to_numpy()

        error_type = kwargs.get('error_type', 'sd')
        error_col_map = {'sd': 'std', 'sem': 'sem', 'ci': 'ci'}
        error_col = error_col_map.get(error_type, 'std')
        if error_col in self.condition_stats.columns:
            cond_err = self.condition_stats.loc[valid, error_col].to_numpy()
        else:
            Reporter(Level.warning, f"Requested error type '{error_type}' not found in condition statistics. Defaulting to standard deviation.", noticequeue=self.noticequeue)
            cond_err = self.condition_stats.loc[valid, 'std'].to_numpy()

        center_valid = centers[valid.to_numpy()]

        if kwargs.get('orient', 'v') == 'v':
            lines = ax.hlines
        else:
            lines = ax.vlines

        if any(cond_mean) and kwargs.get('cond_mean', True):
            span_l = center_valid - kwargs.get('mean_span', 0.16)
            span_r = center_valid + kwargs.get('mean_span', 0.16)
            lines(cond_mean, span_l, span_r,
                      colors=kwargs.get('mean_color', 'black'), linestyles=kwargs.get('mean_ls', '-'),
                      linewidths=kwargs.get('line_width', 1), zorder=3, label='mean')
            
        if any(cond_median) and kwargs.get('cond_median', True):
            span_l = center_valid - kwargs.get('median_span', 0.12)
            span_r = center_valid + kwargs.get('median_span', 0.12)
            lines(cond_median, span_l, span_r,
                      colors=kwargs.get('median_color', 'black'), linestyles=kwargs.get('median_ls', '--'),
                      linewidths=kwargs.get('line_width', 1), zorder=3, label='median')
            
        if any(cond_mean) and kwargs.get('error', True):
            if kwargs.get('orient', 'v') == 'v':
                if kwargs.get('error_type', 'sd') != 'ci' or ((kwargs.get('error_type', 'ci') == 'ci' and self.ci_statistic == 'mean')):
                    y = cond_mean
                elif kwargs.get('error_type', 'ci') == 'ci' and self.ci_statistic == 'median':
                    y = cond_median
                else:
                    y = cond_mean
                pos_args = dict(x=center_valid, y=y, yerr=cond_err)
            else:
                if kwargs.get('error_type', 'sd') != 'ci' or ((kwargs.get('error_type', 'ci') == 'ci' and self.ci_statistic == 'mean')):
                    x = cond_mean
                elif kwargs.get('error_type', 'ci') == 'ci' and self.ci_statistic == 'median':
                    x = cond_median
                else:
                    x = cond_mean
                pos_args = dict(x=x, y=center_valid, xerr=cond_err)

            ax.errorbar(
                **pos_args, fmt='none', 
                color=kwargs.get('errorbar_color', 'black'), alpha=kwargs.get('errorbar_alpha', 0.8),
                linewidth=kwargs.get('line_width', 2), capsize=kwargs.get('errorbar_capsize', 5),
                zorder=3, label=(kwargs.get('error_type', 'sd'))
            )


    def _cond_error(self, valid: pd.Series, error_type: str = 'sd') -> np.ndarray:

        match error_type:
            case 'sd':
                return self.condition_stats.loc[valid, 'std'].to_numpy()
            case 'sem':
                return self.condition_stats.loc[valid, 'sem'].to_numpy()
            case 'ci':
                return self.condition_stats.loc[valid, 'ci'].to_numpy()
            case _:
                Reporter(Level.warning, f"Unknown error type '{error_type}'. Defaulting to standard deviation.", noticequeue=self.noticequeue)
                return self.condition_stats.loc[valid, 'std'].to_numpy()


    def _legend(self, ax: plt.Axes, palette: dict, **kwargs):

        handles, labels = [], []
        err_lbl = self._error_label(**kwargs)
        palette_norm = {str(k): v for k, v in palette.items()}

        if not self.monochrome and 'Replicate' in self.data.columns:
            for r in self.data['Replicate'].astype(str).unique().tolist():
                c = palette_norm.get(r, 'grey')
                handles.append(mlines.Line2D([], [], linestyle='None',
                                            marker='o', markersize=8,
                                            markerfacecolor=c,
                                            markeredgecolor='black',
                                            label=(str(r))))
                labels.append(str(r))

        match (kwargs.get('cond_mean', True), kwargs.get('cond_median', True), kwargs.get('error', True)):

            case (_, _, False):

                if kwargs.get('cond_mean', True):
                    handles.append(mlines.Line2D([], [], color=kwargs.get('mean_color', 'black'),
                                   linestyle=kwargs.get('mean_ls', '-'), linewidth=kwargs.get('line_width', 2),
                                   label='Mean'))
                    labels.append('Mean')

                if kwargs.get('cond_median', True):
                    handles.append(mlines.Line2D([], [], color=kwargs.get('median_color', 'black'),
                                   linestyle=kwargs.get('median_ls', '--'), linewidth=kwargs.get('line_width', 2),
                                   label='Median'))
                    labels.append('Median')

            case (_, _, True):

                match (kwargs.get('cond_mean', True), kwargs.get('cond_median', True)):

                    case (True, False):

                        if kwargs.get('error_type', 'sd') != 'ci' or (kwargs.get('error_type', 'ci') == 'ci' and self.ci_statistic == 'mean'):
                            handles.append(mlines.Line2D([], [], color=kwargs.get('errorbar_color', 'black'),
                                           linestyle='-', linewidth=kwargs.get('line_width', 2),
                                           marker='_', markersize=10,
                                           label=f'Mean ± {err_lbl}'))
                            labels.append(f'Mean ± {err_lbl}')

                        elif kwargs.get('error_type', 'ci') == 'ci' and self.ci_statistic != 'mean':
                            handles.append(mlines.Line2D([], [], color=kwargs.get('mean_color', 'black'),
                                           linestyle=kwargs.get('mean_ls', '-'), linewidth=kwargs.get('line_width', 2),
                                           label='Mean'))
                            labels.append('Mean')
                            handles.append(mlines.Line2D([], [], color=kwargs.get('errorbar_color', 'black'),
                                           linestyle='-', linewidth=kwargs.get('line_width', 2),
                                           marker='_', markersize=10,
                                           label=f'{err_lbl}'))
                            labels.append(f'{err_lbl}')
                    
                    case (False, True):

                        if kwargs.get('error_type', 'ci') == 'ci' and self.ci_statistic == 'median':
                            handles.append(mlines.Line2D([], [], color=kwargs.get('errorbar_color', 'black'),
                                           linestyle='-', linewidth=kwargs.get('line_width', 2),
                                           marker='_', markersize=10,
                                           label=f'Median ± {err_lbl}'))
                            labels.append(f'Median ± {err_lbl}')

                        else:
                            handles.append(mlines.Line2D([], [], color=kwargs.get('median_color', 'black'),
                                           linestyle=kwargs.get('median_ls', '--'), linewidth=kwargs.get('line_width', 2),
                                           label='Median'))
                            labels.append('Median')
                            handles.append(mlines.Line2D([], [], color=kwargs.get('errorbar_color', 'black'),
                                           linestyle='-', linewidth=kwargs.get('line_width', 2),
                                           marker='_', markersize=10,
                                           label=f'{err_lbl}'))
                            labels.append(f'{err_lbl}')

                    case (False, False):
                        handles.append(mlines.Line2D([], [], color=kwargs.get('errorbar_color', 'black'),
                                       linestyle='-', linewidth=kwargs.get('line_width', 2),
                                       marker='_', markersize=10,
                                       label=f'{err_lbl}'))
                        labels.append(f'{err_lbl}')

                    case (True, True):

                        if kwargs.get('error_type', 'sd') != 'ci' or ((kwargs.get('error_type', 'ci') == 'ci' and self.ci_statistic == 'mean')):
                            handles.append(mlines.Line2D([], [], color=kwargs.get('errorbar_color', 'black'),
                                           linestyle='-', linewidth=kwargs.get('line_width', 2),
                                           marker='_', markersize=10,
                                           label=f'Mean ± {err_lbl}'))
                            labels.append(f'Mean ± {err_lbl}')
                            handles.append(mlines.Line2D([], [], color=kwargs.get('median_color', 'black'),
                                           linestyle=kwargs.get('median_ls', '--'), linewidth=kwargs.get('line_width', 2),
                                           label='Median'))
                            labels.append('Median')

                        elif kwargs.get('error_type', 'ci') == 'ci' and self.ci_statistic == 'median':
                            handles.append(mlines.Line2D([], [], color=kwargs.get('errorbar_color', 'black'),
                                           linestyle='-', linewidth=kwargs.get('line_width', 2),
                                           marker='_', markersize=10,
                                           label=f'Median ± {err_lbl}'))
                            labels.append(f'Median ± {err_lbl}')
                            handles.append(mlines.Line2D([], [], color=kwargs.get('mean_color', 'black'),
                                           linestyle=kwargs.get('mean_ls', '-'), linewidth=kwargs.get('line_width', 2),
                                           label='Mean'))
                            labels.append('Mean')
                        
                    case _:
                        Reporter(Level.warning, "Unexpected combination of cond_mean and cond_median with error.", noticequeue=self.noticequeue)

        if not handles:
            return

        ax.legend(handles, labels, title='Legend', title_fontsize=kwargs.get('legend_title_fontsize', 12), fontsize=kwargs.get('legend_fontsize', 9),
                  loc='upper right', frameon=kwargs.get('legend_frame', True), framealpha=kwargs.get('legend_frame_alpha', 0.8), edgecolor=kwargs.get('legend_edge_color', 'black'),
                  facecolor=kwargs.get('legend_face_color', 'white'))
        
        try:
            if plt.gca().get_legend() is not None:
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        except Exception:
            try:
                ax = plt.gca()
                if plt.gca().get_legend() is not None:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            except Exception:
                pass


    def _error_label(self, **kwargs):
        if kwargs.get('error', True):
            error_label = None
            if kwargs.get('error_type', 'sd') != 'ci':
                error_label = f"{kwargs.get('error_type', 'sd').upper()}"
            else:
                match (self.ci_statistic, kwargs.get('cond_mean', True), kwargs.get('cond_median', True)):
                    case ('mean', True, _):
                        error_label = f"CI{self.confidence_level}%"
                    case ('median', _, True):
                        error_label = f"CI{self.confidence_level}%"
                    case (a, _, _):
                        error_label = f"{a.capitalize()} CI{self.confidence_level}%"

            return error_label