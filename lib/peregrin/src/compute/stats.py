from __future__ import annotations

import traceback
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Callable, Literal, Optional, Tuple, Dict, List, Union

from ..various import Values, is_empty
from ..settings import params
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *



class Stats:
    """
    A class with methods for computing trajectory statistics at various levels of aggregation:
    spots (per-trajectory-point), tracks (per-whole-trajectory), frames (per-time-point),
    time intervals (per-time-interval).

    Calling this method initializes statistical configuration.

    Parameters
    ----------
    cat_descr : bool, default True
        If True, descriptive statistics (min, max, mean, median, q25, q75) will be computed for categories.
    
    cat_descr_err : bool, default True
        If True, descriptive error statistics (std) will be computed.

    cat_infer_err : bool, default False
        If True, inferative statistics (sem, ci) will be computed.

    bootstrap_ci : bool, default False
        If True, ci will be computed when the `cat_infer_err` is set to True.


    Attributes
    ----------
    significant_figures : int, optional
        *If specified, all values are going to be rounded to the given number of significant figures.*

    decimal_places : int, optional
        *If specified, all floating-point values are going to be rounded to the given number of decimal places.*

    BOOTSTRAP_RESAMPLES : int, default 1000
        *A number of resamples to perform when calculating bootstrap confidence intervals.*

    CONFIDENCE_LEVEL : int, default 95
        *Confidence level (%) to use when calculating confidence intervals.*
    
    CI_STATISTIC : str, default 'mean'
        *Statistic to calculate confidence intervals for (e.g. 'mean', 'median').*

    
    Methods
    -------
    GetAllData(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]

    """

    ignore_categories: bool = params.ignore_categories

    t_step: Optional[float] = None
    t_unit: str = 's'
    significant_figures: Optional[int] = None
    decimal_places: Optional[int] = None

    DEFAULT_CATEGORIES = ['track_uid', 'subsubgroup', 'subgroup', 'group', 'subset', 'set']

    BOOTSTRAP_RESAMPLES: int = 1000
    CONFIDENCE_LEVEL: float = 95
    CI_STATISTIC: str = 'mean'
    _ci_method_used: str = 'BCa'

    _PANDAS_BUILTINS = frozenset({
        'mean', 'median', 'std', 'count', 'sum', 'min', 'max',
        'first', 'last', 'var', 'prod', 'size', 'nunique',
    })

    _EXCLUDE_SUFFIXES = set([
        'track_id', 'track_uid', 'time_point', 'frame', 'time_lag', 'frame_lag', 'sd', 'var', 'sem', 'q25', 'q75'
    ])

    # Class-level defaults for stat categories
    _DESCR = ['min', 'max', 'mean', 'median', 'q25', 'q75']
    _DESCR_ERR = ['std']
    _INFER_ERR = ['sem']

    COLUMNS = {
        'SPOTS': [
            'condition', 'replicate', 'track_id', 'track_uid',
            'time_point', 'frame', 'x_coordinate', 'y_coordinate', 'distance',
            'cum_track_length', 'cum_track_displacement', 'cum_straightness_ratio',
            'cum_speed_mean', 'cum_mean_straight_line_speed',
            'cum_forward_progression_linearity', 'direction', 'directional_change',
            'cum_sum_directional_change', 'cum_mean_directional_change',
            'cum_mean_directional_change_rate',
            'cum_direction_mean', 'cum_direction_var'
        ],
        'TRACKS': [
            'condition', 'replicate', 'track_id', 'track_uid',
            'y_location', 'x_location',
            'track_length', 'track_displacement', 'straightness_ratio',
            'speed_min', 'speed_max', 'speed_mean', 'speed_sd', 'speed_median',
            'mean_straight_line_speed', 'forward_progression_linearity',
            'max_distance_reached', 'track_start_frame', 'track_end_frame',
            'direction_mean', 'direction_var', 'mean_directional_change', 'mean_directional_change_rate'
        ],
        'FRAMES': ['condition', 'replicate', 'time_point', 'frame'],
        'TIMEINTERVALS': ['condition', 'replicate', 'time_lag', 'frame_lag']
    }

    def __init__(
        self,
        *,
        cat_descr: bool = True,
        cat_descr_err: bool = True,
        cat_infer_err: bool = False,
        bootstrap_ci: bool = False,
        **kwargs
    ) -> None:

        # self.tier: List[str] = ['set', 'subset', 'group', 'subgroup', 'subsubgroup']
        self.tier = None

        self.cat_descr = cat_descr
        self.cat_descr_err = cat_descr_err
        self.cat_infer_err = cat_infer_err

        # Create per-instance copies to avoid shared mutable state
        self.DESCR: List[str] = list(self._DESCR) if cat_descr else []
        self.DESCR_ERR: List[str] = list(self._DESCR_ERR) if cat_descr_err else []
        self.INFER_ERR: List[str] = []

        if cat_infer_err:
            self.INFER_ERR = list(self._INFER_ERR)
            if bootstrap_ci:
                self.INFER_ERR.append('ci')
        else:
            self.INFER_ERR = []
        
        # Custom aggregation functions
        self.CUSTOM_AGG_FUNCTIONS: Dict[str, Callable] = {
            'q25': self._q25,
            'q75': self._q75,
            'ci': self.ci,
            'sem': self.sem,
            'circ_mean': self._circ_mean,
            'circ_var': self._circ_var,
        }

        # Validate CI_STATISTIC
        if self.CI_STATISTIC not in ['mean', 'median']:
            raise Warning(
                f"CI_STATISTIC '{self.CI_STATISTIC}' may not be meaningful; "
                f"consider using 'mean' or 'median'."
            )

    # def get_all(
    #     self, df: pd.DataFrame,
    #     **kwargs
    # ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     """
    #     Compute trajectory data at all levels of aggregation 
    #     (`spots`, `tracks`, `frames`, `time_intervals`) from input spot data.

    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         Input DataFrame must contain these columns:
    #         - *`condition`*
    #         - *`replicate`*
    #         - `track_id`
    #         - `x_coordinate`
    #         - `y_coordinate`
    #         - `time_point`

    #     ignore_categories : bool, optional
    #         If True, the `condition` and `replicate` columns will be ignored in the computation, and all data will be treated as a single group.
    #         If not specified, the default value is taken from the package settings. 
    #         To change the default configuration and behavior throughout all computations, use `peregrin.settings(ignore_categories=...)`

    #     Returns
    #     -------
    #     tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] : `stats.Spots_df`, `stats.Tracks_df`, `stats.Frames_df` and `stats.TimeIntervals_df` DataFrames.

    #     See also
    #     --------
    #     `stats.spots()`- 
    #     computes per-trajectory-point statistics, both local (previous -> current position) and cumulative (start -> current position).

    #     `stats.tracks()`- 
    #     computes per-whole-trajectory statistics from the Spots_df.

    #     `stats.frames()`- 
    #     computes per-time-point statistics from the Spots_df.

    #     `stats.time_intervals()`- 
    #     computes per-time-interval statistics from the Spots_df.

    #     Documentation
    #     -------------
    #     links..

    #     """

    #     # self.Input = df
    #     spots_df = self.spots(df, **kwargs)

    #     return spots_df, self.tracks(spots_df, **kwargs), self.frames(spots_df, **kwargs), self.time_intervals(spots_df, **kwargs)


    def spots(
        self, 
        df: pd.DataFrame,
        *,
        to_disk: bool = ...,
        **kwargs
    ) -> pd.DataFrame:
        """ Computes per-trajectory-point statistics, both local (previous -> current position) and cumulative (start -> current position).

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame must contain these columns:
            - `track_id`
            - `x_coordinate`
            - `y_coordinate`
            - `time_point`

        ignore_categories : bool, optional
            If True, the `condition` and `replicate` columns will be ignored in the computation, and all data will be treated as a single group.
            If not specified, the default value is taken from the package settings. 
            To change the default configuration and behavior throughout all computations, use `peregrin.settings(ignore_categories=...)`

        Returns
        -------
        pd.DataFrame
            The computed DataFrame containing these columns:

            - `track_id`
            - `track_uid`
            - `time_point`
            - `frame`
            - *`condition`*
            - *`replicate`*

            - `x_coordinate`
            - `y_coordinate`

            - `distance` = 
            euclidean distance between consecutive (previous -> current) positions

            - `cum_track_length` = 
            cumulative sum of `distance` along the track up to the current position

            - `cum_track_displacement` = 
            euclidean distance from the starting position of the track to the current position

            - `cum_straightness_ratio = 
            cum_track_displacement / cum_track_length`

            - `cum_speed_mean` = 
            `cum_track_length / frame`

            - `cum_mean_straight_line_speed` =
            calculated as `cum_track_displacement / (current track point count ⋅ t_step)`.

            - `cum_forward_progression_linearity` =
            calculated as `cum_mean_straight_line_speed` / `cum_speed_mean`

            - `direction` = 
            instantaneous direction of motion in radians `np.arctan2(Δy, Δx)`. Calculated between the previous and current positions.

            - `directional_change` = 
            absolute turning angle (degrees) between consecutive directions, calculated as the angular difference between the current and previous `direction` values, wrapped to the range [-180°, 180°].

            - `cum_sum_directional_change` =
            cumulative sum of `directional_change` along the track up to the current position.

            - `cum_mean_directional_change` = 
            mean of all absolute `directional_change` values along the track up to the current position

            - `cum_mean_directional_change_rate` = 
            mean of all absolute `directional_change` values / (current track point count * `t_step`).

            - `cum_direction_mean` = 
            mean of directions of motion from the starting to the current position.

            - `cum_direction_var` = 
            cumulative direction variance from the starting to the current position.

            - *`other`* = 
            *any additional columns from the input DataFrame that are not part of the above list will be retained in the output if they contain any non-NA values; otherwise, they will be dropped.*

            \n See documentation: links..

        See also
        --------
        `stats.get_all()` - 
        computes and returnes all DataFrames (Spots_df, Tracks_df, Frames_df, TimeIntervals_df) from input spot data in one call.

        `stats.tracks()` - 
        computes per-whole-trajectory statistics from the Spots_df.

        `stats.frames()` - 
        computes per-time-point statistics from the Spots_df.

        `stats.time_intervals()` - 
        computes per-time-interval statistics from the Spots_df.

        Documentation
        -------------

        links..

        """

        df = df.copy()

        if df.empty:
            warnings.warn(message="Input DataFrame is empty. No computation performed.", 
                          category=DataFrameWarning, 
                          stacklevel=2)
            
            return pd.DataFrame(columns=self.COLUMNS['SPOTS'])

        # grouping_cols = self._get_grouping_level(df.columns, grouping_level)
        grouping_cols = [col for col in self.DEFAULT_CATEGORIES if col in df.columns]

        # Create a unique track identifier (track_uid) based on the grouping columns
        df = self._assign_track_uid(df)

        df.sort_values(grouping_cols + ['time_point'], inplace=True)

        if self.t_step is None:
            _t = df.copy()
            _t.sort_values('time_point', inplace=True)
            t_steps = np.diff(_t['time_point'].unique())

            if np.all(t_steps == t_steps[0]):
                t_step = float(t_steps[0])
            else:
                t_step = float(np.median(t_steps))
                warnings.warn(message=f"Time points are not uniformly spaced -> this will most probably lead to incorrect data computation. (spot stats + time stats)\nObserved time steps:\n{t_steps}\nUsing: {t_step}", 
                                category=TimePointWarning, 
                                stacklevel=2)
        else:
            t_step = self.t_step

        grp = df.groupby(level='track_uid', sort=False)

        df['frame'] = grp['time_point'].rank(method='dense').astype('Int64') - 1

        bad = df.groupby(grouping_cols + ['time_point'], sort=False)['frame'].nunique(dropna=True).max()
        if bad and bad > 1:
            raise TimePointError(f"Multiple frames assigned to the same {grouping_cols} × time_point combination. Multiplicates of time_point values within the data. Max frames per time point: {bad}.")

        # distance between the previous and the current position
        df['distance'] = np.hypot(
            grp['x_coordinate'].diff(),
            grp['y_coordinate'].diff()
        ).fillna(np.nan)

        # cum_track_length
        df['cum_track_length'] = grp['distance'].cumsum()

        # cum_track_displacement -> straight-line distance (start -> current position)
        start = grp[['x_coordinate', 'y_coordinate']].transform('first')
        df['cum_track_displacement'] = np.hypot(
            (df['x_coordinate'] - start['x_coordinate']),
            (df['y_coordinate'] - start['y_coordinate'])
        ).replace(0, np.nan)

        # Straightness index: Track displacement vs. actual trajectory length ratio
        # Avoid division by zero by replacing zeros with NaN, then fill
        df['cum_straightness_ratio'] = (df['cum_track_displacement'] / df['cum_track_length'].replace(0, np.nan)).fillna(np.nan)

        # cum_speed_mean -> mean speed from the starting to the current position
        track_start_time = grp['time_point'].transform('first')
        elapsed = (df['time_point'] - track_start_time).replace(0, np.nan)
        df['cum_speed_mean'] = df['cum_track_length'] / elapsed

        cumulative_count = (grp.cumcount() + 1).values  # strip index → plain numpy array
        df['cum_mean_straight_line_speed'] = df['cum_track_displacement'] / (cumulative_count * t_step)
        df['cum_forward_progression_linearity'] = df['cum_mean_straight_line_speed'] / df['cum_speed_mean']

        # Instantaneous direction of motion (rad) -> difference between the previous and current position
        dy = grp['y_coordinate'].diff()
        dx = grp['x_coordinate'].diff()
        df['direction'] = np.arctan2(dy, dx).fillna(np.nan)

        # Directional change (turning angle) -> angular difference between consecutive directions, wrapped to [-π, π], then absolute, converted to degrees
        raw_dir_change = grp['direction'].diff()
        # Wrap to [-π, π] first, then take absolute value, then convert to degrees
        wrapped_dir_change = (raw_dir_change + np.pi) % (2 * np.pi) - np.pi
        df['directional_change'] = np.rad2deg(wrapped_dir_change.abs()).fillna(np.nan)

        # Mean directional change -> vectorized cumulative sum/mean over the same track_uid grouping.
        # NaN entries (first two points of each track) must be excluded from the running count,
        # matching the previous expanding().mean() behavior which skipped NaNs.
        dc = df['directional_change']
        df['cum_sum_directional_change'] = grp_dc_sum = df.groupby(level='track_uid', sort=False)['directional_change'].cumsum(skipna=True)

        # Running count of non-NaN directional_change values within each track
        valid = dc.notna()
        valid_count = valid.groupby(level='track_uid', sort=False).cumsum()
        with np.errstate(invalid='ignore', divide='ignore'):
            df['cum_mean_directional_change'] = df['cum_sum_directional_change'] / valid_count.replace(0, np.nan)

        # First two points of each track have no directional change; set them to NaN
        df.loc[df['directional_change'].isna(), ['cum_mean_directional_change']] = np.nan
        df['cum_mean_directional_change_rate'] = df['cum_mean_directional_change'] / (cumulative_count * t_step)
        
        # Cumulative direction of motion (circular mean and variance) calculations
        # Reuse already-computed dy/dx instead of recomputing arctan2.
        _dir = np.arctan2(dy, dx)
        _sin = np.sin(_dir)
        _cos = np.cos(_dir)

        # Cumulative sums respect track boundaries via grp; NaN at track starts contribute 0.
        cum_sin = _sin.groupby(level='track_uid', sort=False).cumsum()
        cum_cos = _cos.groupby(level='track_uid', sort=False).cumsum()

        # Number of contributing angles (0 at first timepoint, 1 at second, ...)
        n_angles = cumulative_count - 1

        # cum_direction_mean and circular variance
        df['cum_direction_mean'] = np.arctan2(cum_sin, cum_cos)

        with np.errstate(invalid='ignore', divide='ignore'):
            R = np.hypot(cum_sin, cum_cos) / pd.Series(n_angles, index=df.index).replace(0, np.nan)
        df['cum_direction_var'] = 1.0 - R

        df.loc[n_angles == 0, 'cum_direction_var'] = np.nan
        df.loc[n_angles == 1, 'cum_direction_var'] = 0.0

        # Drop all-NaN columns (if any are present)
        df.dropna(how='all', axis='columns', inplace=True)

        if self.significant_figures:
            df = self.signify(df)
        if self.decimal_places:
            df = self.norm_decimals(df)

        return df


    def tracks(
        self, 
        df: pd.DataFrame,
        *,
        to_disk: bool = ...,
        **kwargs
    ) -> pd.DataFrame:
        """ Computes a comprehensive DataFrame of track-level statistics for each trajectory of the input Spots_df.

        Parameters
        ----------
        df : pd.DataFrame
            This method expects the dataframe returned from `stats.spots()`. The input DataFrame must contain these columns:
            - `track_id`
            - `track_uid`
            - `frame`
            - `x_coordinate`
            - `y_coordinate`
            - `distance`
            - `cum_track_displacement`
            - `direction`

        ignore_categories : bool, optional
            If True, the `condition` and `replicate` columns will be ignored in the computation, and all data will be treated as a single group.
            If not specified, the default value is taken from the package settings. 
            To change the default configuration and behavior throughout all computations, use `peregrin.settings(ignore_categories=...)`

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per unique track, containing the following columns:

            - `track_id`
            - `track_uid`

            - `y_location`-
            The mean Y position of the track's starting position.

            - `x_location`-
            The mean X position of the track's starting position.
            
            - `track_length`- 
            Total length of the track (sum of `distance`).

            - `speed_min`, `speed_max`, `speed_mean`, `speed_sd`, `speed_median` - 
            of the `Distances` between consecutive points (step lengths).

            - `max_distance_reached`- 
            Maximum Euclidean distance from the starting position reached at any point along the track.

            - `track_start_frame`- 
            frame number of the first point in the track.

            - `track_end_frame`- 
            frame number of the last point in the track.

            - `track_displacement`- 
            Euclidean distance from the starting position to the end position of the track.

            - `straightness_ratio`- 
            Track's straigtness calculated as `track_displacement` / `track_length`.

            - `mean_straight_line_speed`-
            Calculated as `track_displacement` / (`track_points` * `t_step`).

            - `forward_progression_linearity`-
            Calculated as `mean_straight_line_speed` / `speed_mean`

            - `track_points`- 
            The number of points the trajectory is comprised of.

            - `direction_mean`- 
            Circular mean of the `direction` values.

            - `mean_directional_change`- 
            Mean of absolute directional changes per track (degrees).

            - `mean_directional_change_rate`-
            `mean_directional_change` / (`track_points` * `t_step`)

            - `direction_var`- 
            Circular variance of the `direction` values.

            - *`other`* - 
            any additional columns from the input DataFrame that are not part of the above list will be retained in the output if they contain any non-NA values; otherwise, they will be dropped.*

            \n See documentation: links..

        See also
        --------
        `stats.get_all()`- 
        computes all DataFrames (Spots_df, Tracks_df, Frames_df, TimeIntervals_df) from raw spot data in one call.

        `stats.frames()`- 
        computes per-time-point statistics from the Spots_df.

        `stats.time_intervals()`- 
        computes per-time-interval statistics from the Spots_df.

        """

        # Work on a copy to avoid mutating the caller's DataFrame
        df = df.copy()

        if df.empty:
            warnings.warn(message="Input DataFrame is empty. No computation performed.", 
                          category=DataFrameWarning, 
                          stacklevel=2)
            
            return pd.DataFrame(columns=self.COLUMNS['TRACKS'])

        
        # grouping_cols = self._get_grouping_level(df.columns, grouping_level)
        
        grouping_cols = [col for col in self.DEFAULT_CATEGORIES if col in df.columns]

        df = self._assign_track_uid(df)
        
        if self.t_step is None:
            _t = df.copy()
            _t.sort_values('time_point', inplace=True)
            t_steps = np.diff(_t.time_point.unique())
            
            if np.all(t_steps == t_steps[0]):
                t_step = float(t_steps[0])
            else:
                t_step = float(np.median(t_steps))
                warnings.warn(message=f"Time points are not uniformly spaced -> this will most probably lead to incorrect data computation. (track stats)\nObserved time steps:\n{t_steps}\nUsing: {t_step}",
                                category=TimePointWarning,
                                stacklevel=2)
        else:
            t_step = self.t_step

        # Stash the categorical identifiers for merging them back into the aggregated result DataFrame
        stash = df[grouping_cols].drop_duplicates()

        grp = df.groupby(level='track_uid', sort=False)

        speed_agg_spec = {
            'speed_min':    lambda x: x.min() / t_step,
            'speed_max':    lambda x: x.max() / t_step,
            'speed_mean':   lambda x: x.mean() / t_step,
            'speed_sd':     lambda x: x.std() / t_step,
            'speed_median': lambda x: x.median() / t_step,
        }

        # Build the named agg dict: each entry is (column, func)
        agg_spec = {
            'track_length': ('distance', 'sum'),
        }
        for label, func in speed_agg_spec.items():
            agg_spec[label] = ('distance', func) 

        # Add coordinates first/last for displacement calculation
        agg_spec['start_x'] = ('x_coordinate', 'first')
        agg_spec['end_x']   = ('x_coordinate', 'last')
        agg_spec['start_y'] = ('y_coordinate', 'first')
        agg_spec['end_y']   = ('y_coordinate', 'last')

        agg_spec['x_location'] = ('x_coordinate', 'mean')
        agg_spec['y_location'] = ('y_coordinate', 'mean')

        agg_spec['max_distance_reached'] = ('cum_track_displacement', 'max')

        agg_spec['track_start_frame'] = ('frame', 'min')
        agg_spec['track_start_time'] = ('time_point', 'min')
        agg_spec['track_end_frame'] = ('frame', 'max')
        agg_spec['track_end_time'] = ('time_point', 'max')
        
        agg_spec['mean_straight_line_speed'] = ('cum_mean_straight_line_speed', 'last')
        agg_spec['forward_progression_linearity'] = ('cum_forward_progression_linearity', 'last')

        agg_spec['direction_mean'] = ('cum_direction_mean', 'last')
        agg_spec['direction_var'] = ('cum_direction_var', 'last')
        agg_spec['mean_directional_change'] = ('cum_mean_directional_change', 'last')
        agg_spec['mean_directional_change_rate'] = ('cum_mean_directional_change_rate', 'last')

        agg = grp.agg(**agg_spec)
        
        # If colors were assigned, carry them over
        for col in df.columns:
            if col.endswith('color'):
                colors = grp[col].first()
                agg = agg.merge(colors, left_index=True, right_index=True)

        # Displacement and straightness
        agg['track_displacement'] = np.hypot(agg['end_x'] - agg['start_x'], agg['end_y'] - agg['start_y'])
        agg['straightness_ratio'] = (agg['track_displacement'] / agg['track_length'])
        agg = agg.drop(columns=['start_x','end_x','start_y','end_y'])

        # Points/ per track
        n = grp.size().rename('track_points')
        agg = agg.merge(n, left_index=True, right_index=True)
        df = df.merge(agg, left_index=True, right_index=True, how='right')
        df = df.drop(grouping_cols, axis=1, errors='ignore')

        # Drop any columns that also live in `stash` to avoid _x/_y suffix collisions
        # when merging the categorical identifiers back in.
        overlap = [c for c in stash.columns if c in df.columns]
        df = df.drop(columns=overlap, errors='ignore')

        df = stash.merge(df, left_index=True, right_index=True, how='right')

        for col in self.COLUMNS['SPOTS']:
            if col in df.columns and col not in self.COLUMNS['TRACKS']:
                df = df.drop(columns=[col])

        df.drop_duplicates(inplace=True)
        
        if self.significant_figures:
            df = self.signify(df)
        if self.decimal_places:
            df = self.norm_decimals(df)
        
        return df
    

    def frames(
        self, 
        df: pd.DataFrame,
        *,
        grouping_level: Literal['highest', 'lowest'] | str | int | list = 'highest',
        to_disk: bool = ...,
        **kwargs
    ) -> pd.DataFrame:
        
        """ 
        Computes time point statistics for each category (group). Specifically for example:

        - per replicate - across all trajectories of the same `replicate`
        - per condition - across all trajectories of the same `condition`

        Parameters
        ----------
        df : pd.DataFrame
            This method expects the dataframe acquired by `stats.spots()`. The input DataFrame must contain these columns:
            - `time_point`
            - `frame`
            - `distance`
            - `cum_track_length`
            - `cum_track_displacement`
            - `cum_straightness_ratio`
            - `cum_speed_mean`
            - `direction`
            - `cum_direction_mean`

        ignore_categories : bool, optional
            If True, the `condition` and `replicate` columns will be ignored in the computation, and all data will be treated as a single group.
            If not specified, the default value is taken from the package settings. 
            To change the default configuration and behavior throughout all computations, use `peregrin.settings(ignore_categories=...)`

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per unique combination of `condition` × `replicate` × `time_point` 
            if `ignore_categories` is False, otherwise a single row for all data. It contains the following columns:

            - `time_point`
            - `frame`

            \n`per_(category)_(metric)_`...
                - ***descriptive base statistics:***  `min`, `max`, `mean`, `median`, `q25`, `q75` (iqr) if `cat_descr` is set `True` when initializing the `stats` class
                - ***descriptive error statistics:*** `std` if `descr_descr_err` is set `True` when initializing the `stats` class
                - ***inferative error statistics:***  `sem`, if `descr_infer_err` is set `True` when initializing the `stats` class, 
                `(CI_STATISTIC)_ci(CONFIDENCE_LEVEL)_low` and `(CI_STATISTIC)_ci(CONFIDENCE_LEVEL)_high` (confidence interval) if both `descr_infer_err` and `bootstrap_ci` are set `True`
                - ***circular statistics:*** `mean` (circular mean) and `var` (circular variance) calculated for each of `direction` and `cum_direction`.
            
            - for each of these metrics
                - `cum_track_length`
                - `cum_track_displacement`
                - `cum_straightness_ratio`
                - `instantaneous_speed`
                - `cum_speed_mean`
                - `cum_mean_straight_line_speed`
                - `cum_forward_progression_linearity`
                - `instantaneous_direction`
                - `cum_direction`
                - `cum_sum_directional_change`
                - `cum_mean_directional_change`

        See also
        --------
        `Stats.get_all()`- 
        computes all DataFrames (Spots_df, Tracks_df, Frames_df, TimeIntervals_df) from raw spot data in one call.

        `Stats.tracks()`- 
        computes per-whole-trajectory statistics from the Spots_df.

        `Stats.time_intervals()`- 
        computes per-time-interval statistics from the Spots_df.

        """

        # Work on a copy to avoid mutating the caller's DataFrame
        df = df.copy()
        
        grouping_set = []

        if (isinstance(grouping_level, list) 
            and (all(isinstance(g, list) for g in grouping_level)
                 or not any(g in df.columns for g in grouping_level))):

                for g in grouping_level:
                    grouping_cols = self._get_grouping_level(df.columns, g, exclude='track_uid')
                    grouping_set.append(grouping_cols)
                grouping_cols = max(grouping_set, key=len)
        else:
            grouping_cols = self._get_grouping_level(df.columns, grouping_level, exclude='track_uid')
            grouping_set = [grouping_cols]
        
        df = self._assign_track_uid(df)

        # Stash color columns if present, to carry them over to the output
        _color_cols = [c for c in df.columns if c.endswith('color')]
        _color_stash = None
        if _color_cols:
            # Build a lookup keyed by the replicate/condition grouping columns
            _stash_keys = grouping_cols
            _color_stash = df[_stash_keys + _color_cols].drop_duplicates(subset=_stash_keys)

        group_cols = [grouping_cols[-1]] + ['time_point', 'frame']

        # Expected metrics to compute stats for (their input df labels)
        metrics = [
            'cum_track_length',
            'cum_track_displacement',
            'cum_straightness_ratio',
            'cum_speed_mean',
            'distance',
            'cum_mean_straight_line_speed',
            'cum_forward_progression_linearity',
            'cum_sum_directional_change',
            'cum_mean_directional_change',
        ]
        # (output df labels)
        metric_out = {m: m for m in metrics}
        metric_out['distance'] = 'instantaneous_speed'

        def _compute_level(source: pd.DataFrame, by_cols: list[str], prefix: str) -> pd.DataFrame:
            grp = source.groupby(by_cols, sort=False, observed=True)
            out = pd.DataFrame(index=grp.size().index)

            # Batch scalar statistics via a single named-agg call
            for m in metrics:
                mout = metric_out[m]
                sgrp = grp[m]
                if self.cat_descr:
                    out[f'{mout}_min'] = sgrp.min()
                    out[f'{mout}_max'] = sgrp.max()
                    out[f'{mout}_mean'] = sgrp.mean()
                    out[f'{mout}_median'] = sgrp.median()
                    # out[f'{mout}_q25'] = sgrp.agg(self._q25)
                    # out[f'{mout}_q75'] = sgrp.agg(self._q75)
                if self.cat_descr_err:
                    out[f'{mout}_sd'] = sgrp.std(ddof=1)
                if self.cat_infer_err:
                    out[f'{mout}_sem'] = sgrp.agg(self.sem)
                    if 'ci' in self.INFER_ERR:
                        ci_series = sgrp.agg(self.ci)
                        ci_unpacked = pd.DataFrame(
                            ci_series.tolist(),
                            index=ci_series.index,
                            columns=[
                                f'{mout}_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}_low',
                                f'{mout}_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}_high',
                            ],
                        )
                        for c in ci_unpacked.columns:
                            out[c] = ci_unpacked[c]

            # Circular statistics: compute sin/cos
            dir_vals = pd.to_numeric(source['direction'], errors='coerce').to_numpy(dtype=float)
            cum_vals = pd.to_numeric(source['cum_direction_mean'], errors='coerce').to_numpy(dtype=float)

            sin_dir = np.sin(dir_vals)
            cos_dir = np.cos(dir_vals)
            sin_cum = np.sin(cum_vals)
            cos_cum = np.cos(cum_vals)

            # Build a temporary DataFrame for circular stats aggregation
            circ_df = pd.DataFrame({
                '_sin_dir': sin_dir,
                '_cos_dir': cos_dir,
                '_sin_cum': sin_cum,
                '_cos_cum': cos_cum,
            }, index=source.index)
            for col in by_cols:
                circ_df[col] = source[col].values

            circ = circ_df.groupby(by_cols, sort=False, observed=True).agg(
                sin_dir=('_sin_dir', 'mean'),
                cos_dir=('_cos_dir', 'mean'),
                sin_cum=('_sin_cum', 'mean'),
                cos_cum=('_cos_cum', 'mean'),
            )

            out[f'instantaneous_direction_mean'] = np.arctan2(circ['sin_dir'], circ['cos_dir'])
            out[f'instantaneous_direction_var'] = 1.0 - np.hypot(circ['sin_dir'], circ['cos_dir'])
            out[f'cum_direction_mean'] = np.arctan2(circ['sin_cum'], circ['cos_cum'])
            out[f'cum_direction_var'] = 1.0 - np.hypot(circ['sin_cum'], circ['cos_cum'])
            out[f'cum_mean_directional_change_mean'] = source.groupby(by_cols, sort=False)['cum_mean_directional_change'].mean().values
            return out.reset_index()

        # Compute one frame-stats block per grouping set, then stack them.
        level_frames = []
        for grouping_cols in grouping_set:
            group_cols = [grouping_cols[-1]] + ['time_point', 'frame']

            # Stash color columns if present, to carry them over to the output
            _color_cols = [c for c in df.columns if c.endswith('color')]
            _color_stash = None
            if _color_cols:
                _stash_keys = grouping_cols
                _color_stash = df[_stash_keys + _color_cols].drop_duplicates(subset=_stash_keys)

            # Stash higher-level grouping columns so each lower-level group can be
            # attributed back to its (unambiguous) parent group(s).
            _parent_cols = grouping_cols[:-1]
            _parent_stash = None
            if _parent_cols:
                _key = grouping_cols[-1]
                _parent_stash = (
                    df[[_key] + _parent_cols]
                    .drop_duplicates(subset=[_key])
                    .copy()
                )
                # Normalize key dtype to avoid silent merge failures
                # (e.g. categorical vs object mismatches producing NaNs).
                _parent_stash[_key] = _parent_stash[_key].astype('object')

            level_df = _compute_level(df, group_cols, grouping_cols[-1])

            # Re-attach higher-level grouping columns (parents of the lowest level)
            if _parent_stash is not None:
                _key = grouping_cols[-1]
                level_df[_key] = level_df[_key].astype('object')
                level_df = level_df.merge(_parent_stash, on=_key, how='left')

            # Tag the grouping level so stacked rows remain distinguishable
            level_df['grouping_level'] = str(grouping_cols[0])

            # Re-attach color columns if they were present on input
            if _color_stash is not None:
                level_df = level_df.merge(_color_stash, on=_stash_keys, how='left')

            level_frames.append(level_df)

        if not level_frames:
            return pd.DataFrame(columns=self.COLUMNS['TIMEINTERVALS'])

        df = pd.concat(level_frames, ignore_index=True)

        # JSON-safe cleanup for Shiny/front-end serializers (no NaN/Inf in strict JSON)
        df = df.replace([np.inf, -np.inf], np.nan)

        if self.significant_figures:
            df = self.signify(df)
        if self.decimal_places:
            df = self.norm_decimals(df)

        return df
    
    
    def time_intervals(
        self, 
        df: pd.DataFrame,
        *,
        grouping_level: Literal['highest', 'lowest'] | str | int | list | None = 'highest',
        to_disk: bool = ...,
        **kwargs
    ) -> pd.DataFrame:
        """ 
        Computes per-time-interval statistics.

        For each `frame_lag` value (1, 2, …, maximum), squared displacements and turning angles are computed across trajectories
        and then processed (e.g. averaged) for each category (group). Specifically for example:
        
        - per replicate - across all trajectories of the same `replicate`
        - per condition - across all trajectories of the same `condition`


        Example
        -------

        Trajectories A, B, and C comprised of consecutive points and their positions:
        ```
        pa1 ─ pa2 ─ pa3 ─ pa4 ─ pa5 ─ pa6 ─ pa7
              pb1 ─ pb2 ─ pb3 ─ pb4 ─ pb5 ─ pb6
              pc1 ─ pc2 ─ pc3 ─ pc4 ─ pc5
        ```

        Valid position pairs for the interval (lag) of three frames:
        ```
        pa1 ───────────── pa4
              pa2 ───────────── pa5
              pb1 ───────────── pb4
              pc1 ───────────── pc4
                    pa3 ───────────── pa6
                    pb2 ───────────── pb5
                    pc2 ───────────── pc5
                          pa4 ───────────── pa7
                          pb3 ───────────── pb6
        ```

        MSD formula for a given time lag *k* for a given trajactory *i* with trajectory point positions *p* at a time position *t* :
        ```
        MSDᵢ(k) = ||pᵢ(t+k) - pᵢ(t)||²
        ```
        \n The per-track MSD values are then aggregated across tracks within each of unique `time_lag` × grouping. In case that `ignore_categories` is set to `True`, the aggregation will be performed across all tracks for each unique of `time_lag`.
        
            
        Turning angle formula for a given time lag *k* for a trajectory *i* with trajectory point positions *y* and *x* at a time position *t* :
        ```
        Δθᵢ(k) = ||θᵢ(Δy(t+k), Δx(t+k)) - θᵢ(Δy(t), Δx(t))||
        ```
        \n The angular difference in the direction of motion is then wrapped to [−π, π]. 
        Per-track turning angles values are then aggregated (circular mean and variance) across tracks within each of unique `time_lag`.



        Parameters
        ----------
        df : pd.DataFrame
            This method expects the dataframe returned from `stats.spots()`. The input DataFrame must contain these columns:
            - `track_uid`
            - `time_point`
            - `x_coordinate`
            - `y_coordinate`

        ignore_categories : bool, optional
            If True, the `condition` and `replicate` columns will be ignored in the computation, and all data will be treated as a single group.
            If not specified, the default value is taken from the package settings. 
            To change the default configuration and behavior throughout all computations, use `peregrin.settings(ignore_categories=...)`

        Returns
        -------
        pd.DataFrame
            *A DataFrame with one row per unique combination of category(ies) × `frame_lag`, containing the following columns:*

            - **category(ies)**- categories for the given row.
            - **`frame_lag`**- Integer lag in frames (1, 2, 3, …).
            - **`time_lag`**- Corresponding time lag computed as `frame_lag` × time step.

            \n **`per category`**
                - **`tracks_contributing`**- the number of tracks that contributed data at a given time lag for a given category
                - **`position_pairs_contributing`**- the number of position pairs that contributed data at a given time lag for a given category
                - **`directional_change_mean`**- mean absolute turning angle in degrees
                - **`directional_change_var`**- circular variance of turning angles
                - **`MSD`**
                - **`MSD_sd`**
                - **`MSD_sem`** if `cat_infer_err` is set to `True`
                - **`MSD_{CI_STATISTIC}_ci{CONFIDENCE_LEVEL}_low`** and **`MSD_{CI_STATISTIC}_ci{CONFIDENCE_LEVEL}_high`** (confidence interval) if both `cat_infer_err` and `bootstrap_ci` are set to `True`


            \n Sets `BaseDataInventory.TimeIntervals` to the computed DataFrame.

        Notes
        -----
        - Tracks with fewer than 2 points are excluded from computation.
        - If fewer than 2 unique time points exist in the input, an empty DataFrame is returned.

        See also
        --------
        `stats.get_all()`- 
        computes all DataFrames (Spots_df, Tracks_df, Frames_df, TimeIntervals_df) from raw spot data in one call.

        `stats.spots()`- 
        computes per-trajectory-point statistics, both local (previous -> current position) and cumulative (start -> current position).

        `stats.tracks()`- 
        computes per-whole-trajectory statistics from the Spots_df.

        `stats.frames()`- 
        computes per-time-point statistics from the Spots_df.

        """

        df = df.copy()

        if df.empty: 
            return pd.DataFrame(columns=self.COLUMNS['TIMEINTERVALS'])
        
        grouping_set = []

        if (isinstance(grouping_level, list) 
            and (all(isinstance(g, list) for g in grouping_level)
                 or not any(g in df.columns for g in grouping_level))):

                for g in grouping_level:
                    grouping_cols = self._get_grouping_level(df.columns, g, exclude='track_uid')
                    grouping_set.append(grouping_cols)
                grouping_cols = max(grouping_set, key=len)
        else:
            grouping_cols = self._get_grouping_level(df.columns, grouping_level, exclude='track_uid')
            grouping_set = [grouping_cols]

        # Stash color columns if present, to carry them over to the output
        _color_cols = [c for c in df.columns if c.endswith('color')]
        _color_stash = None
        if _color_cols:
            # Build a lookup keyed by the replicate/condition grouping columns
            _stash_keys = grouping_cols
            _color_stash = df[_stash_keys + _color_cols].drop_duplicates(subset=_stash_keys)

        # Ensure track_uid is available as a column
        if 'track_uid' not in df.columns:
            if df.index.name == 'track_uid':
                df = df.reset_index(drop=False)
            else:
                df = self._assign_track_uid(df).reset_index(drop=False)

        # Unique time points
        t_unique = np.sort(df['time_point'].unique())

        if t_unique.size < 2:
            return pd.DataFrame(columns=self.COLUMNS['TIMEINTERVALS'])

        # Unique time steps (time interval)
        if self.t_step is None:
            _t = df.copy()
            _t.sort_values('time_point', inplace=True)
            t_steps = np.diff(_t['time_point'].unique())

            if np.all(t_steps == t_steps[0]):
                t_step = float(t_steps[0])
            else:
                t_step = float(np.median(t_steps))
                warnings.warn(message=f"Time points are not uniformly spaced -> this will most probably lead to incorrect data computation. (time interval stats)\nObserved time steps:\n{t_steps}\nUsing: {t_step}",
                                category=TimePointWarning,
                                stacklevel=2)
            
        else:
            t_step = self.t_step

        def _agg_msd_turn(msd_src: pd.DataFrame, turn_src: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
            """Aggregate pooled MSD pairs and turning-angle pairs for given grouping."""
            msd_grouped = msd_src.groupby(by_cols, sort=False, observed=True)
            msd_grp = msd_grouped['sq_disp']

            agg_dict = {}
            if self.cat_descr:
                agg_dict['MSD'] = msd_grp.mean()

            if self.cat_descr_err:
                agg_dict['MSD_sd'] = msd_grp.std(ddof=1)

            if self.cat_infer_err:
                agg_dict['MSD_sem'] = msd_grp.agg(self.sem)
                if 'ci' in self.INFER_ERR:
                    ci_series = msd_grp.agg(self.ci)
                    ci_unpacked = pd.DataFrame(
                        ci_series.tolist(),
                        index=ci_series.index,
                        columns=[
                            f'MSD_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}_low',
                            f'MSD_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}_high',
                        ],
                    )
                    for c in ci_unpacked.columns:
                        agg_dict[c] = ci_unpacked[c]

            agg_dict['tracks_contributing'] = msd_grouped['track_uid'].nunique().astype(int)
            agg_dict['position_pairs_contributing'] = msd_grp.size().astype(int)

            result = pd.DataFrame(agg_dict)

            if not turn_src.empty:
                turn_sin = turn_src.assign(
                    _s=np.sin(turn_src.dtheta.values),
                    _c=np.cos(turn_src.dtheta.values),
                )
                turn_circ = turn_sin.groupby(by_cols, sort=False, observed=True).agg(
                    ms=('_s', 'mean'),
                    mc=('_c', 'mean'),
                )
                circ_mean = np.arctan2(turn_circ.ms.values, turn_circ.mc.values)
                circ_R = np.hypot(turn_circ.ms.values, turn_circ.mc.values)
                circ_var = 1.0 - circ_R

                if self.cat_descr or self.cat_descr_err or self.cat_infer_err:
                    result['directional_change_mean'] = pd.Series(np.rad2deg(np.abs(circ_mean)), index=turn_circ.index)

                if self.cat_descr_err:
                    result['directional_change_var'] = pd.Series(circ_var, index=turn_circ.index)

            return result

        def _compute_level(source: pd.DataFrame, grouping_cols: list[str]) -> pd.DataFrame:
            """Compute time-interval stats for a single grouping level."""

            # Vectorized per-track, per-lag computation
            temp = (
                source[grouping_cols + ['track_uid', 'time_point', 'x_coordinate', 'y_coordinate']]
                .copy()
                .sort_values(['track_uid', 'time_point'])
                .reset_index(drop=True)
            )

            # Integer frame index per track (0,1,2,...) based on ordered unique time points.
            # Using dense rank makes lag pairing robust to non-uniform / missing time points:
            # a "lag" is defined in *frames*, not in array-row offsets.
            temp['_frame'] = (
                temp.groupby('track_uid', sort=False)['time_point']
                .rank(method='dense')
                .astype('int64') - 1
            )

            # Number of frames spanned per track (max frame index), for lag validation
            track_span = temp.groupby('track_uid', sort=False)['_frame'].transform('max')
            temp['_n'] = track_span + 1  # frame count spanned

            # Filter out tracks with <2 points early
            track_sizes = temp.groupby('track_uid', sort=False).size()
            valid_uids = track_sizes[track_sizes >= 2].index
            temp = temp[temp['track_uid'].isin(valid_uids)].copy()

            if temp.empty:
                return pd.DataFrame(columns=self.COLUMNS['TIMEINTERVALS'])

            # Consecutive step angles theta[i] = arctan2(dy, dx) between successive frames
            grp_temp = temp.groupby('track_uid', sort=False)
            temp['_dx1']   = grp_temp['x_coordinate'].diff()
            temp['_dy1']   = grp_temp['y_coordinate'].diff()
            temp['_theta'] = np.arctan2(temp['_dy1'].values, temp['_dx1'].values)

            max_lag = int(temp['_frame'].max())
            if max_lag < 1:
                return pd.DataFrame(columns=self.COLUMNS['TIMEINTERVALS'])

            # Build a (track_uid, frame) -> row lookup so a lag pairs points that are
            # exactly `lag` frames apart (correct even with gaps / non-uniform spacing).
            key = list(zip(temp['track_uid'].values, temp['_frame'].values))
            pos_of = {k: i for i, k in enumerate(key)}

            x_arr     = temp['x_coordinate'].values
            y_arr     = temp['y_coordinate'].values
            theta_arr = temp['_theta'].values
            uid_arr   = temp['track_uid'].values
            frame_arr = temp['_frame'].values
            cat_arrs  = {col: temp[col].values for col in grouping_cols}

            msd_records = []
            turn_records = []

            for lag in range(1, max_lag + 1):
                # For every row, does a partner exactly `lag` frames ahead exist in the same track?
                partner_keys = list(zip(uid_arr, frame_arr + lag))
                partner_pos = np.fromiter(
                    (pos_of.get(k, -1) for k in partner_keys),
                    dtype=np.int64, count=len(partner_keys)
                )
                valid_mask = partner_pos >= 0
                if not valid_mask.any():
                    continue

                valid_idx   = np.where(valid_mask)[0]
                partner_idx = partner_pos[valid_idx]

                dx = x_arr[partner_idx] - x_arr[valid_idx]
                dy = y_arr[partner_idx] - y_arr[valid_idx]
                sq_disp = dx * dx + dy * dy

                lag_df = pd.DataFrame({
                    'track_uid': uid_arr[valid_idx],
                    **{col: cat_arrs[col][valid_idx] for col in grouping_cols},
                    'sq_disp':   sq_disp,
                    'frame_lag': lag,
                    'time_lag':  lag * t_step,
                })
                msd_records.append(lag_df)

                # Turning angle: theta at partner frame minus theta at current frame.
                # theta is defined only where a preceding step exists (frame >= 1).
                theta_now = theta_arr[valid_idx]
                theta_par = theta_arr[partner_idx]
                turn_valid = (frame_arr[valid_idx] >= 1) & np.isfinite(theta_now) & np.isfinite(theta_par)

                if turn_valid.any():
                    ti = valid_idx[turn_valid]
                    pi = partner_idx[turn_valid]
                    dtheta = theta_arr[pi] - theta_arr[ti]
                    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
                    turn_records.append(pd.DataFrame({
                        'track_uid': uid_arr[ti],
                        **{col: cat_arrs[col][ti] for col in grouping_cols},
                        'dtheta':    dtheta,
                        'frame_lag': lag,
                        'time_lag':  lag * t_step,
                    }))

            if not msd_records:
                return pd.DataFrame(columns=self.COLUMNS['TIMEINTERVALS'])

            all_msd  = pd.concat(msd_records, ignore_index=True)
            all_turn = (
                pd.concat(turn_records, ignore_index=True)
                if turn_records
                else pd.DataFrame(columns=[*grouping_cols, 'track_uid', 'dtheta', 'frame_lag', 'time_lag'])
            )

            lag_group_cols = grouping_cols + ['frame_lag', 'time_lag']
            lags = _agg_msd_turn(all_msd, all_turn, lag_group_cols)
            lags.reset_index(inplace=True)

            return lags

        # Compute one time-interval block per grouping set, then stack them.
        level_frames = []
        for grouping_cols in grouping_set:

            # Stash color columns if present, to carry them over to the output
            _color_cols = [c for c in df.columns if c.endswith('color')]
            _color_stash = None
            if _color_cols:
                _stash_keys = grouping_cols
                _color_stash = df[_stash_keys + _color_cols].drop_duplicates(subset=_stash_keys)

            level_df = _compute_level(df, grouping_cols)

            if level_df.empty:
                continue

            # Tag the grouping level so stacked rows remain distinguishable
            level_df['grouping_level'] = str(grouping_cols[0])

            # Re-attach color columns if they were present on input
            if _color_stash is not None:
                level_df = level_df.merge(_color_stash, on=_stash_keys, how='left')

            level_frames.append(level_df)

        if not level_frames:
            return pd.DataFrame(columns=self.COLUMNS['TIMEINTERVALS'])

        df = pd.concat(level_frames, ignore_index=True)

        # JSON-safe cleanup for Shiny/front-end serializers (no NaN/Inf in strict JSON)
        df = df.replace([np.inf, -np.inf], np.nan)

        # Re-attach color columns if they were present on input
        if _color_stash is not None:
            df = df.merge(_color_stash, on=_stash_keys, how='left')

        if self.significant_figures:
            df = self.signify(df)
        if self.decimal_places:
            df = self.norm_decimals(df)

        return df



    def _assign_track_uid(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Creates a unique track identifier `track_uid` by combining the default category columns (`self.DEFAULT_CATEGORIES`) present in the DataFrame. """

        if 'track_uid' not in df.columns:
            grouping_columns = [col for col in self.DEFAULT_CATEGORIES if col in df.columns]
            if not all(col in df.columns for col in grouping_columns):
                missing_cols = [col for col in grouping_columns if col not in df.columns]
                raise ColumnsNotFoundError(f"Missing required columns for track_uid creation: {missing_cols}")

            # Create a unique track identifier by assigning a unique integer to each combination of the grouping columns
            df['track_uid'] = df.groupby(grouping_columns, sort=False).ngroup()
        
        df.set_index(['track_uid'], drop=True, append=False, inplace=True, verify_integrity=False)
        return df


    def _get_grouping_level(
        self, 
        df_cols: pd.Index,
        grouping_level: Literal['highest', 'lowest'] | str | int | list | None = 'highest',
        *,
        multiple: bool = False,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> list | list[list]:

        if not isinstance(include, list):
            include = [include] if include is not None else []
        if not isinstance(exclude, list):
            exclude = [exclude] if exclude is not None else []

        grouping_cols = [col for col in self.DEFAULT_CATEGORIES if col in df_cols]

        if isinstance(grouping_level, list):
            grouping_cols = [self._get_grouping_level(df_cols, g_lvl, include=include) for g_lvl in grouping_level]
            if not multiple:
                grouping_cols = max(grouping_cols, key=len)
            
        if is_empty(grouping_cols):
            raise ColumnsNotFoundError(f"No grouping columns found in DataFrame columns: {df_cols}")
        
        if isinstance(grouping_level, int):
            if grouping_level < 0 or grouping_level >= len(grouping_cols):
                raise IndexError(f"Grouping level index {grouping_level} is out of bounds for DataFrame columns: {grouping_cols}")

            grouping_cols = grouping_cols[:grouping_level + 1]
        elif grouping_level == 'highest':
            # Aggregate at the top of the hierarchy (broadest group).
            grouping_cols = [grouping_cols[-1]]
        elif grouping_level == 'lowest':
            # Aggregate at the bottom of the hierarchy (finest group),
            # retaining all parent levels for attribution.
            pass
        elif isinstance(grouping_level, str):
            idx = grouping_cols.index(grouping_level)
            grouping_cols = grouping_cols[:idx + 1]
        else:
            raise InvalidParameterValueError(f"Invalid grouping_level parameter: {grouping_level}. Must be a list of column names, an integer index, 'highest', 'lowest', or None.")
    

        for col in include:
            if col not in grouping_cols:
                grouping_cols.append(col)
            else:
                warnings.warn(message=f"Some columns in 'include' are already present in the default grouping columns: {grouping_cols}. They will be included only once.",
                            category=ConflictWarning,
                            stacklevel=2)

        if not is_empty(exclude):
            grouping_cols = [col for col in grouping_cols if col not in exclude]

            if len(grouping_cols) == 0:
                grouping_cols = ['track_uid']

        return grouping_cols
    

    def format_digits(self, df: pd.DataFrame, *, sig_figs: int = None, decimals: int = None) -> pd.DataFrame:
        """ Formats numeric values in the DataFrame according to specified significant figures and decimal places. """

        if sig_figs:
            df = self.signify(df, sig_figs=sig_figs)

        if decimals:
            df = self.norm_decimals(df, decimals=decimals)

        return df


    def signify(self, df: pd.DataFrame, *, sig_figs: int = None) -> pd.DataFrame:
        """ Round numeric values in a DataFrame to a specified number of significant figures. """

        if df.empty:
            return df.copy()

        if sig_figs is None:
            sig_figs = self.significant_figures

        valuer = Values()

        df_rounded = df.copy()
        for col in df_rounded.select_dtypes(include=[np.number]).columns:
            df_rounded[col] = df_rounded[col].apply(lambda x: valuer.RoundSigFigs(x, sigfigs=sig_figs))

        return df_rounded
    

    def norm_decimals(self, df: pd.DataFrame, decimals: int = None) -> pd.DataFrame:
        """ Normalize decimal places across numeric columns in a DataFrame by rounding to a specified number of decimal places. """

        if df.empty:
            return df.copy()
        
        if decimals is None:
            decimals = self.decimal_places

        # Ensure all values have the same number of decimals: (round, fill)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].apply(lambda x: round(x, decimals) if pd.notnull(x) else x)

        return df


    def _general_agg_stats(self, df: pd.DataFrame, exclude: list[str], *, group_by: list[str] = ['track_uid']) -> pd.DataFrame:
        """ Compute general aggregate statistics (min, max, mean, sd, sem, median) for numeric columns in the DataFrame, grouped by specified columns. Exclude specified columns from aggregation. """

        if exclude is None:
            Reporter(Level.warning, "No columns specified for exclusion in Stats._general_agg_stats(); all numeric columns will be aggregated.", ntcq=self.ntcq)

        exclude = [col for col in exclude if col != 'track_uid']

        # Keep only numeric columns and exclude core columns
        additional = df.copy()

        try:
            additional = additional.drop(columns=exclude, errors='ignore')

            if not additional.shape[1]:
                return pd.DataFrame(index=additional.index)

            additional = additional.select_dtypes(include=[np.number])

            if additional.empty or additional.shape[1] == 0:
                return pd.DataFrame(index=df.index if group_by == ['track_uid'] else df.groupby(level=group_by, sort=False).ngroup().index)

            # Stash leftover columns (exclude 'track_uid' if it's among them)
            other_cols = [c for c in additional.columns.tolist() if c != 'track_uid']

            if not other_cols:
                return pd.DataFrame(index=additional.index)

            # Group by track_uid
            grp = additional.groupby(level=group_by, sort=False)

            # For each bonus column, compute basic statistics, rename columns, and merge back
            for col in other_cols:
                agg = grp[col].agg(['min','max','mean','std','sem','median'])

                agg.columns = [f"{col} min", f"{col} max", f"{col} mean", f"{col} sd", f"{col} sem", f"{col} median"]

                additional = additional.merge(agg, left_index=True, right_index=True)

            # Drop original columns
            additional.drop(columns=other_cols, inplace=True)

            # drop multiplicates if present
            additional = additional.drop_duplicates()
            
            return additional
        
        except Exception as e:
            warnings.warn(message=f"Stats._general_agg_stats() encountered an error: {e}. Returning empty DataFrame. Traceback:\n{traceback.format_exc()}", 
                          category=FailedWarning, 
                          stacklevel=2)
            
            return pd.DataFrame(index=df.index)


    def _describe_infer(self, df: pd.DataFrame, group_cols: list[str], *, stats: dict[str, str] | list[str] = None, **kwargs) -> pd.Series:
        if not stats:
            return df

        # only numeric columns, excluding id/category-like columns
        value_cols = [c for c in df.columns
                      if c not in group_cols
                      and pd.api.types.is_numeric_dtype(df[c])
                      and not any(s in c for s in kwargs.get('exclude', self._EXCLUDE_SUFFIXES))]

        resolving = self.resolve(stats)
        resolving_circular = self.resolve(kwargs.get('circular_stats', {'mean': 'circ_mean', 'var': 'circ_var'}))

        # named aggregation so output columns are already flat
        named_agg = {}
        for col in value_cols:
            
            if any(t in col for t in ['direction', 'direction', 'Directional', 'directional', 'Turn', 'turn']) and not col.endswith('var'):
                for stat_name, func in resolving_circular.items():
                    named_agg[f"per_{group_cols[-1].lower()}_{col}_{stat_name}"] = (col, func)
            else:
                for stat_name, func in resolving.items():
                    if stat_name != 'ci':
                        named_agg[f"per_{group_cols[-1].lower()}_{col}_{stat_name}"] = (col, func)
                    else:
                        named_agg[f"per_{group_cols[-1].lower()}_{col}_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}"] = (col, func)

        grp_stats = (
            df.groupby(group_cols, observed=True, sort=False)
            .agg(**named_agg)
            .reset_index())

        return df.merge(grp_stats, on=group_cols, how='left')
    
    
    def resolve(self, agg_spec: dict[str, str] | list[str]) -> dict[str, str | callable]:
        """ Resolves a list or dictionary of aggregation specs into a mapping of output labels to aggregation functions. """
        
        resolved = {}
        if isinstance(agg_spec, list):
            for func_name in agg_spec:
                if func_name in self._PANDAS_BUILTINS:
                    resolved[func_name] = func_name
                elif func_name in self.CUSTOM_AGG_FUNCTIONS:
                    resolved[func_name] = self.CUSTOM_AGG_FUNCTIONS[func_name]
                else:
                    raise ValueError(
                        f"Unknown aggregation '{func_name}'. "
                        f"Available: {sorted(self._PANDAS_BUILTINS | set(self.CUSTOM_AGG_FUNCTIONS))}"
                    )

        elif isinstance(agg_spec, dict):
            for label, func_name in agg_spec.items():
                if func_name in self._PANDAS_BUILTINS:
                    resolved[label] = func_name
                elif func_name in self.CUSTOM_AGG_FUNCTIONS:
                    resolved[label] = self.CUSTOM_AGG_FUNCTIONS[func_name]
                else:
                    raise ValueError(
                        f"Unknown aggregation '{func_name}'. "
                        f"Available: {sorted(self._PANDAS_BUILTINS | set(self.CUSTOM_AGG_FUNCTIONS))}"
                    )
        return resolved
        

    def _insert_at_position(self, d: dict, key: Any, value: Any = None, *, where: int | str = 0) -> dict:
        """ Insert a (key: value) pair into a dictionary at a specific position.

        Parameters
        ----------
        d : dict
            *The original dictionary.*

        insert : tuple
            *The key-value pair to insert.*

        where : int | str, optional (default=0)
            *The position at which to insert the new key-value pair. If an integer, it is treated as an index. If a string, it is treated as a key name.*
        """

        items = list(d.items())

        if isinstance(where, int):
            index = where

        elif isinstance(where, str):
            keys = [k for k, _ in items]
            if where not in keys:
                raise ValueError(f"Key '{where}' not found in dictionary.")
            index = keys.index(where) + 1
            
        else:
            raise ValueError("Parameter 'where' must be an integer index or a string key.")
        
        items.insert(index, (key, value))

        return dict(items)


    def _wrap_pi(self, a: np.ndarray) -> np.ndarray:
        """ Wrap angles in radians to the range [-π, π]. """
        return (a + np.pi) % (2*np.pi) - np.pi


    def _circ_mean(self, a: np.ndarray) -> float:
        """ Circular mean of angles in radians. """
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        
        return float(np.arctan2(s, c))


    def _circ_var(self, a: np.ndarray) -> float:
        """ Circular variance defined as 1 - R, where R is the mean resultant length of the angles. """
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        
        R = np.hypot(s, c)
        return float(1.0 - R)
    

    def _q25(self, a: np.ndarray) -> float:
        """ Lower bound of the interquartile range = Q1. """
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]  # drop NaN/Inf
        if a.size == 0:
            return np.nan
        return float(np.percentile(a, 25))
    

    def _q75(self, a: np.ndarray) -> float:
        """ Upper bound of the interquartile range = Q3. """
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]  # drop NaN/Inf
        if a.size == 0:
            return np.nan
        return float(np.percentile(a, 75))
    

    def ci(self, a, *, n_resamples: int | None = None, confidence_level: float | None = None, **kwargs) -> tuple[float, float]:
        """ Confidence interval via bootstrap.
        
        Parameters
        ----------
        a : array-like
            *1D array of values to compute the confidence interval for.*
        
        n_resamples : int, (default self.BOOTSTRAP_RESAMPLES = 1000)
            *Number of bootstrap resamples to perform. Can be set through Stats.BOOTSTRAP_RESAMPLES*

        confidence_level : float, (default self.CONFIDENCE_LEVEL = 95)
            *Confidence level for the interval (%). Can be set through Stats.CONFIDENCE_LEVEL*

        statistic : callable, (default `np.mean`)
            *Function for which the confidence interval is computed (e.g. `np.mean`, `np.median`).*
        
        method : str, (default 'BCa')
            *Confidence interval computation method. Default is 'BCa' (bias-corrected and accelerated). 
            If 'BCa' fails, the method falls back to 'percentile'. Used method is stored in and can be acquired through `Stats._ci_method_used`.*
            
        Returns
        -------
        tuple[float, float]
            *A tuple containing the lower and upper bounds of the confidence interval. If computation fails, returns ``(np.nan, np.nan)``.*
        """

        method = kwargs.get('method', 'BCa')
        seed = 42 # Fixed seed for reproducibility
        
        # Ensure input is a numpy array of floats for the bootstrap compatibility 
        a = np.asarray(a, dtype=float)

        # Drop NaN values, for bootstrap cannot handle them
        a = a[~np.isnan(a)]

        if a.size < 2:
            return (np.nan, np.nan)

        # Convert percentage (e.g. 95) to fraction (0.95) for scipy
        cl = self.CONFIDENCE_LEVEL if confidence_level is None else confidence_level
        if cl > 1:
            cl = cl / 100.0

        try:
            result = stats.bootstrap(
                (a,),
                statistic=kwargs.get('statistic', getattr(np, self.CI_STATISTIC)),
                n_resamples=self.BOOTSTRAP_RESAMPLES if n_resamples is None else n_resamples,
                confidence_level=cl,
                method=method,
                random_state=seed
            )
            self._ci_method_used = method
            return (float(result.confidence_interval.low), float(result.confidence_interval.high))
        
        except Exception:
            # Fallback to the percentile method if previous the method fails
            try:
                result = stats.bootstrap(
                    (a,),
                    statistic=kwargs.get('statistic', getattr(np, self.CI_STATISTIC)),
                    n_resamples=self.BOOTSTRAP_RESAMPLES if n_resamples is None else n_resamples,
                    confidence_level=cl,
                    method='percentile',
                    random_state=seed
                )
                self._ci_method_used = 'percentile'
                return (float(result.confidence_interval.low), float(result.confidence_interval.high))
            
            except Exception as e:
                warnings.warn(message=f"Bootstrap confidence interval computation failed for both '{method}' and fallback 'percentile' methods: {e}. Returning (np.nan, np.nan). Traceback:\n{traceback.format_exc()}",
                              category=FailedWarning,
                              stacklevel=2)
                return (np.nan, np.nan)
    

    def sem(self, x: np.ndarray | pd.Series) -> float:
        """ Standard error of the mean. """
        if isinstance(x, np.ndarray):
            # x = x[~np.isnan(x)]
            n = len(x)
            if n < 2:
                return np.nan
            return np.std(x, ddof=1) / np.sqrt(n)
        else:
            n = x.count()
            if n < 2:
                return np.nan
            return x.std(ddof=1) / np.sqrt(n)



    def stat_units(self, col: str = None, *, time_unit: str = None, **kwargs) -> dict[str, str]:
        """ Returns a dictionary mapping metric names to their corresponding units 
        
        Parameters
        ----------
        col : str, optional
            *If provided, returns the unit for the specified column. If not provided, returns a dictionary of all column-unit mappings.*
        time_unit : str, optional
            *If provided, overrides the default time unit for all time-related metrics in the returned mapping.*
        **kwargs : dict
            *Additional keyword arguments*
            - `time_data` (bool): If `True`, strips certain metrics of their time component as the time series chart itself is expected to include a time axis.*
        """

        if time_unit is not None:
            t_unit = time_unit
        else:
            t_unit = self.t_unit

        units = {

            # Spotstats metrics
            'x_coordinate': 'µm',
            'y_coordinate': 'µm',
            'time_point': f'{t_unit}',
            'distance': 'µm',
            'instantaneous_speed': f'µm ⋅ {t_unit}⁻¹',
            'cum_track_length': 'µm',
            'cum_track_displacement': 'µm',
            'cum_straightness_ratio': 'µm',
            'cum_speed_mean': f'µm ⋅ {t_unit}⁻¹',
            'cum_mean_straight_line_speed': f'µm ⋅ {t_unit}⁻¹',
            'cum_forward_progression_linearity': 'µm',
            'direction': 'rad',
            'directional_change': 'rad',
            'cum_sum_directional_change': 'rad',
            'cum_mean_directional_change': 'rad',
            'cum_mean_directional_change_rate': f'rad ⋅ {t_unit}⁻¹',
            'cum_direction_mean': 'rad',

            # Trackstats metrics
            'y_location': 'µm',
            'x_location': 'µm',
            'track_length': 'µm',
            'track_displacement': 'µm',
            'speed_min': f'µm ⋅ {t_unit}⁻¹',
            'speed_max': f'µm ⋅ {t_unit}⁻¹',
            'speed_mean': f'µm ⋅ {t_unit}⁻¹',
            'speed_sd': f'µm ⋅ {t_unit}⁻¹',
            'speed_median': f'µm ⋅ {t_unit}⁻¹',
            'mean_straight_line_speed': f'µm ⋅ {t_unit}⁻¹',
            'max_distance_reached': 'µm',
            'direction_mean': 'rad',
            'mean_directional_change': 'rad',
            'mean_directional_change_rate': f'rad ⋅ {t_unit}⁻¹',

            # Timeintervalstats metrics
            'time_lag': f'{t_unit}',
            'msd': 'µm²',
            'directional_change_mean': 'rad',
        }

        if kwargs.get('time_data', False):
            units.update({
                # Framestats metrics
                'time_point': f'{t_unit}',
                'frame': '',
                'cum_track_length': 'µm',
                'cum_track_displacement': 'µm',
                'cum_speed_mean': 'µm',
                'instantaneous_speed': 'µm',
                'cum_mean_straight_line_speed': 'µm',
                'cum_sum_directional_change': 'rad',
                'cum_mean_directional_change': 'rad',
                'cum_mean_directional_change_rate': f'rad  ⋅ {t_unit}⁻¹',
                'instantaneous_direction_mean': 'rad',
                'cum_direction_mean': 'rad',
            })
        if col is not None:
            return units.get(col, None)

        return units


class Summarize:
    """ Contains static methods utilized in the Peregrin Shiny App """

    @staticmethod
    def dataframe_summary(df: pd.DataFrame) -> dict:
        return {
            "rows": len(df),
            "columns": df.shape[1],
            "missing_cells": int(df.isna().sum().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }

    @staticmethod
    def column_summary(series: pd.Series) -> dict:
        # Robust handling of pandas nullable dtypes (pd.NA) and mixed types
        if pd.api.types.is_numeric_dtype(series):
            s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

            # If there is at least one real numeric value, treat as numeric summary
            if s.notna().any():
                mode = s.mode(dropna=True)
                return {
                    "type": "type_one",
                    "missing": int(series.isna().sum()),
                    "distinct": int(series.nunique(dropna=True)),
                    "min": s.min(skipna=True),
                    "max": s.max(skipna=True),
                    "mean": s.mean(skipna=True),
                    "median": s.median(skipna=True),
                    "mode": float(mode.iloc[0]) if not mode.empty else None,
                    "sd": s.std(ddof=1, skipna=True),
                    "variance": s.var(skipna=True),
                }

        value_counts = series.value_counts(dropna=True, normalize=True).head(3)
        return {
            "type": "type_zero",
            "missing": int(series.isna().sum()),
            "distinct": int(series.nunique(dropna=True)),
            "highest": [(idx, round(val * 100, 1)) for idx, val in value_counts.items()],
        }



stats = Stats()
# get_all = stats.get_all
spots = stats.spots
tracks = stats.tracks
frames = stats.frames
time_intervals = stats.time_intervals