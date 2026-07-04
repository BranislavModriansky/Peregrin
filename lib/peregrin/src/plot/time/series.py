from __future__ import annotations
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..categorizer import Categorizer, Painter
from ...various import is_empty
from ...compute.stats import Stats



class TSeries:

    FILL_ALPHA = 0.18
    REPLICATES_ALPHA = 0.3

    STAT_COLS = ['mean', 'median', 'min', 'max']
    DISPER_COLS  = ['sem', 'sd', 'min-max', 'ci', 'iqr', 'none', None]


    def __init__(self, data: pd.DataFrame, conditions: list = None, replicates: list = None, metric: str = None,
                 *args, stat: str = 'mean', disper: str = None, log_transform_x: bool = False, level: str = 'Condition', ignore_categories: bool = False, **kwargs) -> None:
        
        self.data = data
        self.conditions = conditions
        self.replicates = replicates
        self.metric = metric
        self.log_transform_x = log_transform_x
        self.stat = stat
        self.disper = disper if disper != 'std' else 'sd'
        self.ignore_categories = ignore_categories

        match level:
            case 'Condition':
                self.level  = ['Condition']
                self.prefix = '{per condition} '
            case 'Replicate':
                self.level  = ['Condition', 'Replicate']
                self.prefix = '{per replicate} '
            case _:
                Reporter(Level.warning, "Unknown level argument {level}. Level must be either 'Condition' or 'Replicate'. Defaulting to 'Condition'.", noticequeue=self.noticequeue)
                self.level  = ['Condition']
                self.prefix = '{per condition} '

        if ignore_categories:
            self.level  = None
            self.prefix = None
        
        self._assign_kwargs(kwargs)

        self._error = False
        self._check_errors()


    def _assign_kwargs(self, kwargs):
        """Assign keyword arguments to instance variables with defaults."""
        self.noticequeue = kwargs.get('noticequeue', None)
        self.color = kwargs.get('color', None)
        self.palette = kwargs.get('palette', None)
        self.stock_palette = kwargs.get('stock_palette', 'tab10')  # <-- added default
        self.painter = Painter(noticequeue=self.noticequeue)
        self.figsize = kwargs.get('figsize', (8, 5))
        self.xscale = kwargs.get('xscale', 'literal')
        self.time_units = kwargs.get('time_units', None)
        self.yscale = kwargs.get('yscale', 'literal')
        self.title = kwargs.get('title', None)
        self.darkmode = kwargs.get('darkmode', False)

        self.ci_statistic = Stats.CI_STATISTIC
        self.confidence_level = Stats.CONFIDENCE_LEVEL


    def _check_errors(self):
        """Validate input data"""

        if is_empty(self.data):
            Reporter(Level.error, "Input data is empty.", noticequeue=self.noticequeue)
            self._error = True

        data_cols = self.data.columns.tolist()

        if self.stat not in self.STAT_COLS:
            Reporter(Level.error, f"Unexpected statistic '{self.stat}' for metric '{self.metric}'.", noticequeue=self.noticequeue)
            self._error = True

        elif self.disper is not None and self.disper not in self.DISPER_COLS:
            Reporter(Level.error, f"Unexpected dispersion type '{self.disper}' for metric '{self.metric}'.", noticequeue=self.noticequeue)
            self._error = True

        elif 'Condition' not in data_cols:
            Reporter(Level.error, "Column 'Condition' not found in data.", noticequeue=self.noticequeue)
            self._error = True

        elif self.level == ['Condition', 'Replicate'] and 'Replicate' not in data_cols:
            Reporter(Level.error, "Column 'Replicate' not found in data.", noticequeue=self.noticequeue)
            self._error = True

        elif f'{self.prefix}{self.metric} {self.stat}' not in data_cols:
            Reporter(Level.error, f"Column '{self.prefix} {self.metric} {self.stat}' not found.", noticequeue=self.noticequeue)
            self._error = True

        elif is_empty(self.data[f'{self.prefix}{self.metric} {self.stat}']):
            Reporter(Level.error, f"Column '{self.prefix} {self.metric} {self.stat}' is empty.", noticequeue=self.noticequeue)
            self._error = True


    def plot(self) -> plt.Figure:
        if self._error: return

        fig, ax = plt.subplots(figsize=self.figsize)
        
        if self.log_transform_x:
            ax.set_xscale('log')

        x_col      = self._get_x_col()
        y_col      = f'{self.prefix}{self.metric} {self.stat}'
        disper_col = self._get_disper_col()

        if not self.ignore_categories:
            self._arrange_data()

        colors = self._get_palette()
        color_list = list(colors.values())

        # Build groups
        if self.ignore_categories:
            iter_groups = [(None, self.data)]
        else:
            iter_groups = self.data.groupby(self.level, sort=False)

        for idx, (key, group) in enumerate(iter_groups):
            group = group.sort_values('Frame')

            x_vals = group[x_col].to_numpy()
            y_vals = group[y_col].to_numpy()

            if disper_col is None:
                disper_vals = None
            elif isinstance(disper_col, list):
                disper_vals = [group[c].to_numpy() for c in disper_col]
            else:
                disper_vals = group[disper_col].to_numpy()

            # Resolve color: explicit > palette-by-key > cycle through list
            if self.color:
                c = self.color
            elif self.level and key is not None:
                group_label = key if isinstance(key, str) else key[0]
                c = colors.get(group_label, color_list[idx % len(color_list)])
            else:
                c = color_list[idx % len(color_list)]

            label = None
            if self.level:
                label = group[self.level[0]].iloc[0]

            ax.plot(x_vals, y_vals, color=c, linewidth=2, label=label)

            if disper_vals is not None:
                if isinstance(disper_vals, list) and len(disper_vals) == 2:
                    ax.fill_between(x_vals, disper_vals[0], disper_vals[1],
                                    color=c, alpha=self.FILL_ALPHA, linewidth=0)
                elif isinstance(disper_vals, list) and len(disper_vals) == 1:
                    ax.fill_between(x_vals, y_vals - disper_vals[0], y_vals + disper_vals[0],
                                    color=c, alpha=self.FILL_ALPHA, linewidth=0)
                elif not isinstance(disper_vals, list):
                    ax.fill_between(x_vals, y_vals - disper_vals, y_vals + disper_vals,
                                    color=c, alpha=self.FILL_ALPHA, linewidth=0)

        # Y units
        time_unit = Stats.t_unit if self.time_units is None else self.time_units
        y_units = Stats().stat_units(self.metric, time_unit=time_unit, time_data=True)
        if y_units is not None:
            y_units = f' [{y_units}]'
        else:
            y_units = ''

        # Labels and title
        ax.set_xlabel(f"Time [{time_unit}]" if x_col == 'Time point' else 'Frame')
        ax.set_ylabel(f"{y_col}{y_units}")
        ax.set_title(self.title)
        ax.legend(loc='best', frameon=True, framealpha=0.9)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        
        fig.set_facecolor('whitesmoke') if not self.darkmode else fig.set_facecolor("#262626")
        if self.darkmode: 
            ax.set_facecolor("#161616")
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('white')
            legend = ax.get_legend()
            if legend:
                frame = legend.get_frame()
                frame.set_facecolor('#2e2e2e')
                for text in legend.get_texts():
                    text.set_color('white')

        fig.tight_layout()
        return plt.gcf()


    def _arrange_data(self):
        """Arrange data according to conditions and replicates."""
        self.data = Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
        )()


    def _get_dispersion(self) -> np.ndarray | None:
        """Detect the appropriate error column in the data based on self.disper."""
        if self.disper is None:
            return None

        err_col = f'{self.prefix}{self.stat} {self.disper}'
        if err_col not in self.data.columns:
            Reporter(Level.warning, f"Dispersion column '{err_col}' not found in data -> No error bars will be plotted.")
            return None
        
        if is_empty(self.data[err_col]):
            Reporter(Level.warning, f"Dispersion column '{err_col}' is empty -> No error bars will be plotted.")
            return None

        return err_col


    def _get_disper_col(self) -> str:
        match self.disper:
            case 'sem'         : return [f'{self.prefix}{self.metric} sem']
            case 'sd'          : return [f'{self.prefix}{self.metric} sd']
            case 'min-max'     : return [f'{self.prefix}{self.metric} min', f'{self.prefix}{self.metric} max']
            case 'iqr'         : return [f'{self.prefix}{self.metric} q25', f'{self.prefix}{self.metric} q75']
            case 'ci'          : return [f'{self.prefix}{self.metric} {self.ci_statistic} ci{self.confidence_level} low', f'{self.prefix}{self.metric} {self.ci_statistic} ci{self.confidence_level} high']
            case 'none' | None : return  None
            case _:
                Reporter(Level.warning, f"Unknown dispersion type '{self.disper}' -> No error bars will be plotted.")
                return None


    def _get_palette(self) -> dict:
        """ Get color palette for the plot. """
        if self.level == ['Condition']:
            if not self.palette:
                return self.painter.BuildQualPalette(self.data, 'Condition', which=self.conditions)
            else:
                return self.painter.StockQualPalette(self.data, 'Condition', self.stock_palette, which=self.conditions)
        elif self.level == ['Condition', 'Replicate']:
            if not self.palette:
                return self.painter.BuildQualPalette(self.data, 'Replicate', which=self.replicates)
            else:
                return self.painter.StockQualPalette(self.data, 'Replicate', self.stock_palette, which=self.replicates)
        else:
            # ignore_categories / fallback
            return {'_default': self.color or '#1f77b4'}


    def _get_x_col(self) -> str:
        """Detect the x-axis column in the data."""
        match self.xscale:
            case 'literal' | 'time':
                return 'Time point'
            case 'relative' | 'frame':
                return 'Frame'
            case _:
                Reporter(Level.warning, f"Unknown xscale '{self.xscale}'. Defaulting to 'relative'.")
                return 'Frame'
    