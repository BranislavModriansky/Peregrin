from __future__ import annotations

import traceback
import warnings
import numpy as np
import polars as pl
from scipy import stats
from typing import Any, Callable, Literal, Optional, Dict, List

from ..various import Values, is_empty
from ..settings import params

from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
class MetricRegistry:
    """
    Maps an output column name -> a callable that builds that single column.

    Computers receive a shared context dict and may return either:
      - a `pl.Expr` aggregation expression (collected into ONE group_by().agg()
        call -> single pass over the data), or
      - a post-processing callable `(out_df, ctx) -> out_df` for metrics that
        cannot be expressed as a polars aggregation (e.g. bootstrap CIs).

    A `gate` classifies each column so that, when `subset` is None, only the
    columns enabled by the current statistical flags are computed.

    gate values:
        'always'    -> always produced (e.g. contribution counts)
        'descr'     -> produced only when cat_descr is True
        'descr_err' -> produced only when cat_descr_err is True
        'infer_err' -> produced only when cat_infer_err is True
    """

    def __init__(self) -> None:
        self._computers: Dict[str, Callable] = {}
        self._gate: Dict[str, str] = {}
        self._order: List[str] = []

    def register(self, column: str, *, gate: str = 'always') -> Callable:
        def _wrap(fn: Callable) -> Callable:
            if column not in self._computers:
                self._order.append(column)
            self._computers[column] = fn
            self._gate[column] = gate
            return fn
        return _wrap

    def add(self, column: str, fn: Callable, *, gate: str = 'always') -> None:
        """Imperative registration (non-decorator)."""
        self.register(column, gate=gate)(fn)

    def all_columns(self, *, descr: bool, descr_err: bool, infer_err: bool) -> List[str]:
        allowed = {'always'}
        if descr:
            allowed.add('descr')
        if descr_err:
            allowed.add('descr_err')
        if infer_err:
            allowed.add('infer_err')
        return [c for c in self._order if self._gate[c] in allowed]

    def resolve(
        self,
        subset: Optional[List[str]],
        *,
        descr: bool,
        descr_err: bool,
        infer_err: bool,
    ) -> List[str]:
        """Determine which registered columns to compute."""
        if subset is None:
            return self.all_columns(descr=descr, descr_err=descr_err, infer_err=infer_err)
        requested = set(subset)
        return [c for c in self._order if c in requested and c in self._computers]

    def compute(self, wanted: List[str], ctx: dict) -> pl.DataFrame:
        """Run the requested computers against a shared context.

        `ctx['source']` -> the source pl.DataFrame
        `ctx['by']`     -> grouping column names (list)
        """
        exprs: List[pl.Expr] = []
        posts: List[Callable] = []

        for col in wanted:
            result = self._computers[col](ctx)
            if isinstance(result, pl.Expr):
                exprs.append(result.alias(col))
            elif callable(result):
                posts.append(result)

        # Extra list-aggregations requested by post-processors (e.g. CI sources)
        for name, expr in ctx.get('extra_exprs', {}).items():
            exprs.append(expr.alias(name))

        out = (
            ctx['source']
            .group_by(ctx['by'], maintain_order=True)
            .agg(exprs)
        )

        for post in posts:
            out = post(out, ctx)

        # Drop helper list columns
        helper = [c for c in out.columns if c.startswith('__list_')]
        if helper:
            out = out.drop(helper)

        return out


class Calc:
    """
    A class with methods for computing trajectory statistics at various levels of aggregation:
    spots (per-trajectory-point), tracks (per-whole-trajectory), frames (per-time-point),
    time intervals (per-time-interval).

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
    significant_figures, decimal_places, BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL,
    CI_STATISTIC : see original documentation.
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

    _POLARS_BUILTINS = frozenset({
        'mean', 'median', 'std', 'count', 'sum', 'min', 'max',
        'first', 'last', 'var', 'product', 'len', 'n_unique',
    })

    _EXCLUDE_SUFFIXES = set([
        'track_id', 'track_uid', 'time_point', 'frame', 'time_lag', 'frame_lag', 'sd', 'var', 'sem', 'q25', 'q75'
    ])

    _DESCR_STATS = ['min', 'max', 'mean', 'median', 'q25', 'q75']
    _DESCR_ERR = ['std']
    _INFER_ERR = ['sem']

    COLUMNS = {
        'SPOTS': [
            'track_id', 'track_uid',
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

    UNIT_TO_SECONDS = {
        "ms": 1e-3,
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }

    UNIT_TO_MICRONS = {
        "nm": 1e-3,
        "μm": 1.0,   # U+03BC (matches InputMetadata.UNIT_ALIASES)
        "µm": 1.0,   # U+00B5 (micro sign, kept for safety)
        "mm": 1e3,
        "cm": 1e4,
        "m": 1e6,
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

        self.tier = None

        self.cat_descr = cat_descr
        self.cat_descr_err = cat_descr_err
        self.cat_infer_err = cat_infer_err

        self.DESCR: List[str] = list(self._DESCR_STATS) if cat_descr else []
        self.DESCR_ERR: List[str] = list(self._DESCR_ERR) if cat_descr_err else []
        self.INFER_ERR: List[str] = []

        if cat_infer_err:
            self.INFER_ERR = list(self._INFER_ERR)
            if bootstrap_ci:
                self.INFER_ERR.append('ci')
        else:
            self.INFER_ERR = []

        # Custom aggregation expression builders (column name -> pl.Expr)
        self.CUSTOM_AGG_FUNCTIONS: Dict[str, Callable[[str], pl.Expr]] = {
            'q25': lambda c: pl.col(c).quantile(0.25, interpolation='linear'),
            'q75': lambda c: pl.col(c).quantile(0.75, interpolation='linear'),
            'sem': lambda c: pl.col(c).std(ddof=1) / pl.col(c).count().cast(pl.Float64).sqrt(),
            'circ_mean': lambda c: pl.arctan2(pl.col(c).sin().mean(), pl.col(c).cos().mean()),
            'circ_var': lambda c: 1.0 - (pl.col(c).sin().mean().pow(2) + pl.col(c).cos().mean().pow(2)).sqrt(),
        }

        if self.CI_STATISTIC not in ['mean', 'median']:
            raise Warning(
                f"CI_STATISTIC '{self.CI_STATISTIC}' may not be meaningful; "
                f"consider using 'mean' or 'median'."
            )

        self._frames_registry = self._build_frames_registry()
        self._ti_registry = self._build_time_intervals_registry()
        self._tracks_registry = self._build_tracks_registry()

    # -----------------------------------------------------------------------
    # Registry builders
    # -----------------------------------------------------------------------
    def _build_tracks_registry(self) -> MetricRegistry:
        """One aggregation expression per TRACKS output column.

        ctx['t_step'] -> the resolved time step
        """
        reg = MetricRegistry()

        def _speed(agg: str) -> Callable:
            return lambda ctx: getattr(pl.col('distance'), agg)() / ctx['t_step']

        reg.add('speed_min', _speed('min'))
        reg.add('speed_max', _speed('max'))
        reg.add('speed_mean', _speed('mean'))
        reg.add('speed_sd', lambda ctx: pl.col('distance').std(ddof=1) / ctx['t_step'])
        reg.add('speed_median', _speed('median'))

        reg.add('track_length', lambda ctx: pl.col('distance').sum())

        reg.add('x_location', lambda ctx: pl.col('x_coordinate').mean())
        reg.add('y_location', lambda ctx: pl.col('y_coordinate').mean())

        reg.add('max_distance_reached', lambda ctx: pl.col('cum_track_displacement').max())

        reg.add('track_start_frame', lambda ctx: pl.col('frame').min())
        reg.add('track_end_frame', lambda ctx: pl.col('frame').max())

        reg.add('mean_straight_line_speed', lambda ctx: pl.col('cum_mean_straight_line_speed').last())
        reg.add('forward_progression_linearity', lambda ctx: pl.col('cum_forward_progression_linearity').last())

        reg.add('direction_mean', lambda ctx: pl.col('cum_direction_mean').last())
        reg.add('direction_var', lambda ctx: pl.col('cum_direction_var').last())
        reg.add('mean_directional_change', lambda ctx: pl.col('cum_mean_directional_change').last())
        reg.add('mean_directional_change_rate', lambda ctx: pl.col('cum_mean_directional_change_rate').last())

        reg.add('track_points', lambda ctx: pl.len())

        _disp = (
            (pl.col('x_coordinate').last() - pl.col('x_coordinate').first()).pow(2)
            + (pl.col('y_coordinate').last() - pl.col('y_coordinate').first()).pow(2)
        ).sqrt()

        reg.add('track_displacement', lambda ctx: _disp)
        reg.add('straightness_ratio', lambda ctx: _disp / pl.col('distance').sum())

        return reg

    def _build_frames_registry(self) -> MetricRegistry:
        """One computer per FRAMES output column."""
        reg = MetricRegistry()

        metric_out = {
            'cum_track_length': 'cum_track_length',
            'cum_track_displacement': 'cum_track_displacement',
            'cum_straightness_ratio': 'cum_straightness_ratio',
            'cum_speed_mean': 'cum_speed_mean',
            'distance': 'instantaneous_speed',
            'cum_mean_straight_line_speed': 'cum_mean_straight_line_speed',
            'cum_forward_progression_linearity': 'cum_forward_progression_linearity',
            'cum_sum_directional_change': 'cum_sum_directional_change',
            'cum_mean_directional_change': 'cum_mean_directional_change',
        }

        def _scalar(src: str, agg: str) -> Callable:
            return lambda ctx: getattr(pl.col(src), agg)()

        def _std(src: str) -> Callable:
            return lambda ctx: pl.col(src).std(ddof=1)

        def _sem(src: str) -> Callable:
            return lambda ctx: self.CUSTOM_AGG_FUNCTIONS['sem'](src)

        def _ci(src: str, low_name: str, high_name: str) -> Callable:
            """Bootstrap CI cannot be expressed as a polars aggregation;
            aggregate the raw values as a list and post-process once."""
            def _computer(ctx: dict) -> Callable:
                ctx.setdefault('extra_exprs', {})['__list_sq_disp'] = pl.col('sq_disp')

                def _post(out: pl.DataFrame, _ctx: dict) -> pl.DataFrame:
                    if low_name in out.columns:
                        return out
                    bounds = [
                        self.ci(np.asarray(v, dtype=float))
                        for v in out['__list_sq_disp'].to_list()
                    ]
                    return out.with_columns(
                        pl.Series(low_name, [b[0] for b in bounds], dtype=pl.Float64),
                        pl.Series(high_name, [b[1] for b in bounds], dtype=pl.Float64),
                    )
                return _post
            return _computer

        for src, mout in metric_out.items():
            reg.add(f'{mout}_min', _scalar(src, 'min'), gate='descr')
            reg.add(f'{mout}_max', _scalar(src, 'max'), gate='descr')
            reg.add(f'{mout}_mean', _scalar(src, 'mean'), gate='descr')
            reg.add(f'{mout}_median', _scalar(src, 'median'), gate='descr')
            reg.add(f'{mout}_sd', _std(src), gate='descr_err')
            reg.add(f'{mout}_sem', _sem(src), gate='infer_err')

            if 'ci' in self.INFER_ERR:
                low = f'{mout}_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}_low'
                high = f'{mout}_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}_high'
                reg.add(low, _ci(src, low, high), gate='infer_err')
                reg.add(high, _ci(src, low, high), gate='infer_err')

        # --- Circular statistics (fully expression-based) -------------------
        reg.add('instantaneous_direction_mean',
                lambda ctx: self.CUSTOM_AGG_FUNCTIONS['circ_mean']('direction'), gate='descr')
        reg.add('instantaneous_direction_var',
                lambda ctx: self.CUSTOM_AGG_FUNCTIONS['circ_var']('direction'), gate='descr')
        reg.add('cum_direction_mean',
                lambda ctx: self.CUSTOM_AGG_FUNCTIONS['circ_mean']('cum_direction_mean'), gate='descr')
        reg.add('cum_direction_var',
                lambda ctx: self.CUSTOM_AGG_FUNCTIONS['circ_var']('cum_direction_mean'), gate='descr')
        reg.add('cum_mean_directional_change_mean',
                lambda ctx: pl.col('cum_mean_directional_change').mean(), gate='descr')

        return reg

    def _build_time_intervals_registry(self) -> MetricRegistry:
        """One computer per TIMEINTERVALS output column."""
        reg = MetricRegistry()

        reg.add('MSD', lambda ctx: pl.col('sq_disp').mean(), gate='descr')
        reg.add('MSD_sd', lambda ctx: pl.col('sq_disp').std(ddof=1), gate='descr_err')
        reg.add('MSD_sem', lambda ctx: self.CUSTOM_AGG_FUNCTIONS['sem']('sq_disp'), gate='infer_err')

        def _ci(low_name: str, high_name: str) -> Callable:
            def _computer(ctx: dict) -> Callable:
                ctx.setdefault('extra_exprs', {})['__list_sq_disp'] = pl.col('sq_disp')

                def _post(out: pl.DataFrame, _ctx: dict) -> pl.DataFrame:
                    if low_name in out.columns:
                        return out
                    bounds = [
                        self.ci(np.asarray(v, dtype=float))
                        for v in out['__list_sq_disp'].to_list()
                    ]
                    return out.with_columns(
                        pl.Series(low_name, [b[0] for b in bounds], dtype=pl.Float64),
                        pl.Series(high_name, [b[1] for b in bounds], dtype=pl.Float64),
                    )
                return _post
            return _computer

        if 'ci' in self.INFER_ERR:
            low = f'MSD_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}_low'
            high = f'MSD_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}_high'
            reg.add(low, _ci(low, high), gate='infer_err')
            reg.add(high, _ci(low, high), gate='infer_err')

        reg.add('tracks_contributing', lambda ctx: pl.col('track_uid').n_unique().cast(pl.Int64), gate='always')
        reg.add('position_pairs_contributing', lambda ctx: pl.len().cast(pl.Int64), gate='always')

        # --- Turning-angle circular statistics (computed on turn_src, joined) -
        def _circ(kind: str) -> Callable:
            def _computer(ctx: dict) -> Callable:
                def _post(out: pl.DataFrame, _ctx: dict) -> pl.DataFrame:
                    turn_src: pl.DataFrame = _ctx['turn_src']
                    cache = _ctx['circ_cache']
                    if 'data' not in cache:
                        if is_empty(turn_src):
                            cache['data'] = None
                        else:
                            cache['data'] = (
                                turn_src
                                .group_by(_ctx['by'], maintain_order=True)
                                .agg(
                                    ms=pl.col('dtheta').sin().mean(),
                                    mc=pl.col('dtheta').cos().mean(),
                                )
                                .with_columns(
                                    directional_change_mean=pl.arctan2(pl.col('ms'), pl.col('mc')).abs().degrees(),
                                    directional_change_var=1.0 - (pl.col('ms').pow(2) + pl.col('mc').pow(2)).sqrt(),
                                )
                                .drop(['ms', 'mc'])
                            )
                    data = cache['data']
                    col = f'directional_change_{kind}'
                    if data is None or col in out.columns:
                        return out
                    return out.join(data.select(_ctx['by'] + [col]), on=_ctx['by'], how='left')
                return _post
            return _computer

        reg.add('directional_change_mean', _circ('mean'), gate='descr')
        reg.add('directional_change_var', _circ('var'), gate='descr_err')

        return reg

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------
    def _resolve_t_step(self, df: pl.DataFrame, context: str) -> float:
        """Resolve the time step from data (or self.t_step if set)."""
        if self.t_step is not None:
            return self.t_step

        t_steps = np.diff(np.sort(df['time_point'].unique().to_numpy()))

        if t_steps.size == 0:
            return 1.0
        if np.all(t_steps == t_steps[0]):
            return float(t_steps[0])

        t_step = float(np.median(t_steps))
        warnings.warn(
            message=(f"Time points are not uniformly spaced -> this will most probably lead to "
                     f"incorrect data computation. ({context})\nObserved time steps:\n{t_steps}\nUsing: {t_step}"),
            category=TimePointWarning,
            stacklevel=3,
        )
        return t_step

    @staticmethod
    def _drop_all_null_columns(df: pl.DataFrame) -> pl.DataFrame:
        keep = [c for c in df.columns if df[c].null_count() < df.height]
        return df.select(keep)

    # -----------------------------------------------------------------------
    # SPOTS
    # -----------------------------------------------------------------------
    def spots(
        self,
        df: pl.DataFrame,
        subset: list[str] = None,
        **kwargs
    ) -> pl.DataFrame:
        """Computes per-trajectory-point statistics, both local (previous -> current
        position) and cumulative (start -> current position).

        Fully vectorized via polars window expressions (`.over('track_uid')`);
        see the original documentation for column descriptions.
        """

        if is_empty(df):
            warnings.warn(message="Input DataFrame is empty. No computation performed.",
                          category=DataFrameWarning, stacklevel=2)
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['SPOTS']})

        grouping_cols = [col for col in self.DEFAULT_CATEGORIES if col in df.columns]

        df = self._ensure_polars(df)
        df = self.assign_track_uid(df)

        # Sort so that each track's rows are contiguous and time-ordered.
        df = df.sort(grouping_cols + ['track_uid', 'time_point'])

        t_step = self._resolve_t_step(df, 'spot stats + time stats')

        uid = 'track_uid'

        # frame: dense rank of time_point per track (0-based)
        df = df.with_columns(
            (pl.col('time_point').rank(method='dense').over(uid) - 1).cast(pl.Int64).alias('frame')
        )

        # Validate per TRACK: one time_point -> exactly one frame within a track.
        bad = (
            df.group_by([uid, 'time_point'])
            .agg(pl.col('frame').n_unique().alias('_n'))
            .select(pl.col('_n').max())
            .item()
        )
        if bad and bad > 1:
            raise TimePointError(
                f"Multiple frames assigned to the same track_uid × time_point "
                f"combination. Duplicate time_point values within a track. "
                f"Max frames per time point: {bad}."
            )

        # Step deltas + distance between the previous and the current position
        df = df.with_columns(
            (pl.col('x_coordinate') - pl.col('x_coordinate').shift(1)).over(uid).alias('_dx'),
            (pl.col('y_coordinate') - pl.col('y_coordinate').shift(1)).over(uid).alias('_dy'),
        ).with_columns(
            (pl.col('_dx').pow(2) + pl.col('_dy').pow(2)).sqrt().alias('distance')
        )

        # Cumulative metrics
        df = df.with_columns(
            pl.col('distance').cum_sum().over(uid).alias('cum_track_length'),
            (
                (pl.col('x_coordinate') - pl.col('x_coordinate').first().over(uid)).pow(2)
                + (pl.col('y_coordinate') - pl.col('y_coordinate').first().over(uid)).pow(2)
            ).sqrt().alias('cum_track_displacement'),
            pl.col('time_point').cum_count().over(uid).cast(pl.Float64).alias('_cumcount'),
        ).with_columns(
            # zero displacement (start point) -> null, matching prior behavior
            pl.when(pl.col('cum_track_displacement') == 0)
            .then(None).otherwise(pl.col('cum_track_displacement'))
            .alias('cum_track_displacement'),
        )

        elapsed = (pl.col('time_point') - pl.col('time_point').first().over(uid))
        df = df.with_columns(
            (
                pl.col('cum_track_displacement')
                / pl.when(pl.col('cum_track_length') == 0).then(None).otherwise(pl.col('cum_track_length'))
            ).alias('cum_straightness_ratio'),
            (
                pl.col('cum_track_length')
                / pl.when(elapsed == 0).then(None).otherwise(elapsed)
            ).alias('cum_speed_mean'),
        ).with_columns(
            (pl.col('cum_track_displacement') / (pl.col('_cumcount') * t_step))
            .alias('cum_mean_straight_line_speed'),
        ).with_columns(
            (pl.col('cum_mean_straight_line_speed') / pl.col('cum_speed_mean'))
            .alias('cum_forward_progression_linearity'),
        )

        # Instantaneous direction (rad) and turning angle (deg, wrapped, abs)
        df = df.with_columns(
            pl.arctan2(pl.col('_dy'), pl.col('_dx')).alias('direction')
        ).with_columns(
            (
                ((pl.col('direction') - pl.col('direction').shift(1)).over(uid) + np.pi)
                .mod(2 * np.pi) - np.pi
            ).abs().degrees().alias('directional_change')
        )

        # Cumulative sum / running mean of directional change (nulls skipped)
        df = df.with_columns(
            pl.col('directional_change').cum_sum().over(uid).alias('cum_sum_directional_change'),
            pl.col('directional_change').is_not_null().cum_sum().over(uid)
            .cast(pl.Float64).alias('_valid_count'),
        ).with_columns(
            (
                pl.col('cum_sum_directional_change')
                / pl.when(pl.col('_valid_count') == 0).then(None).otherwise(pl.col('_valid_count'))
            ).alias('cum_mean_directional_change')
        ).with_columns(
            # First two points of each track have no directional change
            pl.when(pl.col('directional_change').is_null())
            .then(None).otherwise(pl.col('cum_mean_directional_change'))
            .alias('cum_mean_directional_change'),
        ).with_columns(
            (pl.col('cum_mean_directional_change') / (pl.col('_cumcount') * t_step))
            .alias('cum_mean_directional_change_rate'),
        )

        # Cumulative circular mean / variance of direction
        df = df.with_columns(
            pl.col('direction').sin().cum_sum().over(uid).alias('_cum_sin'),
            pl.col('direction').cos().cum_sum().over(uid).alias('_cum_cos'),
            (pl.col('_cumcount') - 1).alias('_n_angles'),
        ).with_columns(
            pl.arctan2(pl.col('_cum_sin'), pl.col('_cum_cos')).alias('cum_direction_mean'),
            (
                1.0 - (pl.col('_cum_sin').pow(2) + pl.col('_cum_cos').pow(2)).sqrt()
                / pl.when(pl.col('_n_angles') == 0).then(None).otherwise(pl.col('_n_angles'))
            ).alias('cum_direction_var'),
        ).with_columns(
            pl.when(pl.col('_n_angles') == 0).then(None)
            .when(pl.col('_n_angles') == 1).then(0.0)
            .otherwise(pl.col('cum_direction_var'))
            .alias('cum_direction_var'),
        )

        # Drop helpers + all-null columns
        df = df.drop(['_dx', '_dy', '_cumcount', '_valid_count', '_cum_sin', '_cum_cos', '_n_angles'])
        df = self._drop_all_null_columns(df)

        if subset is not None:
            _always_keep = ['track_id', 'track_uid', 'time_point', 'frame']
            requested = set(subset)
            keep = [c for c in df.columns if c in _always_keep or c in requested or c in grouping_cols]
            df = df.select(keep)

        if self.significant_figures:
            df = self.signify(df)
        if self.decimal_places:
            df = self.norm_decimals(df)

        return df

    # -----------------------------------------------------------------------
    # TRACKS
    # -----------------------------------------------------------------------
    def tracks(
        self,
        df: pl.DataFrame,
        subset: list[str] = None,
        *,
        to_disk: bool = ...,
        **kwargs
    ) -> pl.DataFrame:
        """Computes track-level statistics for each trajectory of the input Spots_df.

        All metrics are built as polars aggregation expressions and computed in a
        single `group_by('track_uid').agg(...)` pass. See the original
        documentation for column descriptions.
        """

        if is_empty(df):
            warnings.warn(message="Input DataFrame is empty. No computation performed.",
                          category=DataFrameWarning, stacklevel=2)
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TRACKS']})

        grouping_cols = [col for col in self.DEFAULT_CATEGORIES if col in df.columns]

        df = self.assign_track_uid(df)
        df = df.sort(['track_uid', 'time_point'])

        t_step = self._resolve_t_step(df, 'track stats')

        # Stash categorical identifiers to merge them back into the result
        stash_cols = [c for c in grouping_cols if c != 'track_uid']
        stash = df.select(['track_uid'] + stash_cols).unique(subset=['track_uid'], keep='first')

        wanted = self._tracks_registry.resolve(subset, descr=True, descr_err=True, infer_err=True)

        ctx = {
            'source': df,
            'by': ['track_uid'],
            't_step': t_step,
        }
        agg = self._tracks_registry.compute(wanted, ctx)

        # Carry over color columns and track_id (first per track)
        carry = [c for c in df.columns if c.endswith('color')]
        if 'track_id' in df.columns:
            carry = ['track_id'] + carry
        if carry:
            firsts = df.group_by('track_uid', maintain_order=True).agg(
                [pl.col(c).first() for c in carry]
            )
            agg = agg.join(firsts, on='track_uid', how='left')

        out = stash.join(agg, on='track_uid', how='right')

        # Drop spot-level columns that leaked through
        drop = [c for c in self.COLUMNS['SPOTS'] if c in out.columns and c not in self.COLUMNS['TRACKS']]
        out = out.drop(drop).unique(maintain_order=True)

        if self.significant_figures:
            out = self.signify(out)
        if self.decimal_places:
            out = self.norm_decimals(out)

        return out

    # -----------------------------------------------------------------------
    # FRAMES
    # -----------------------------------------------------------------------
    def frames(
        self,
        df: pl.DataFrame,
        subset: list[str] = None,
        *,
        grouping_level: Literal['highest', 'lowest'] | str | int | list = 'highest',
        to_disk: bool = ...,
        **kwargs
    ) -> pl.DataFrame:
        """Computes time point statistics for each category (group).

        One `group_by().agg()` pass per grouping level; all descriptive,
        error and circular statistics are polars expressions. See the original
        documentation for details.
        """

        grouping_set = []

        if (isinstance(grouping_level, list)
            and (all(isinstance(g, list) for g in grouping_level)
                 or not any(g in df.columns for g in grouping_level))):
            for g in grouping_level:
                grouping_set.append(self._get_grouping_level(df.columns, g, exclude='track_uid'))
            grouping_cols = max(grouping_set, key=len)
        else:
            grouping_cols = self._get_grouping_level(df.columns, grouping_level, exclude='track_uid')
            grouping_set = [grouping_cols]

        df = self.assign_track_uid(df)

        wanted = self._frames_registry.resolve(
            subset,
            descr=self.cat_descr,
            descr_err=self.cat_descr_err,
            infer_err=self.cat_infer_err,
        )

        level_frames = []
        for grouping_cols in grouping_set:
            group_cols = [grouping_cols[-1]] + ['time_point', 'frame']

            # Stash color columns / parent grouping columns for re-attachment
            _color_cols = [c for c in df.columns if c.endswith('color')]
            _color_stash = (
                df.select(grouping_cols + _color_cols).unique(subset=grouping_cols, keep='first')
                if _color_cols else None
            )

            _parent_cols = grouping_cols[:-1]
            _parent_stash = None
            if _parent_cols:
                _key = grouping_cols[-1]
                _parent_stash = (
                    df.select([_key] + _parent_cols)
                    .unique(subset=[_key], keep='first')
                    .with_columns(pl.col(_key).cast(pl.Utf8))
                )

            ctx = {
                'source': df,
                'by': group_cols,
                'extra_exprs': {},
            }
            level_df = self._frames_registry.compute(wanted, ctx)

            if _parent_stash is not None:
                _key = grouping_cols[-1]
                level_df = level_df.with_columns(pl.col(_key).cast(pl.Utf8))
                level_df = level_df.join(_parent_stash, on=_key, how='left')

            level_df = level_df.with_columns(
                pl.lit(str(grouping_cols[0])).alias('grouping_level')
            )

            if _color_stash is not None:
                level_df = level_df.join(_color_stash, on=grouping_cols, how='left')

            level_frames.append(level_df)

        if not level_frames:
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEINTERVALS']})

        out = pl.concat(level_frames, how='diagonal_relaxed')

        # JSON-safe cleanup (no Inf in strict JSON)
        out = out.with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c, dt in out.schema.items() if dt in (pl.Float32, pl.Float64)
        ])

        if self.significant_figures:
            out = self.signify(out)
        if self.decimal_places:
            out = self.norm_decimals(out)

        return out

    # -----------------------------------------------------------------------
    # TIME INTERVALS
    # -----------------------------------------------------------------------
    def time_intervals(
        self,
        df: pl.DataFrame,
        subset: list[str] = None,
        *,
        grouping_level: Literal['highest', 'lowest'] | str | int | list | None = 'highest',
        to_disk: bool = ...,
        **kwargs
    ) -> pl.DataFrame:
        """Computes per-time-interval statistics (MSD, turning angles, contribution
        counts) per `frame_lag`. See the original documentation for the full
        description; pair-building stays numpy-vectorized, all aggregation runs
        through a single polars `group_by().agg()` per grouping level.
        """

        if is_empty(df):
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEINTERVALS']})

        grouping_set = []

        if (isinstance(grouping_level, list)
            and (all(isinstance(g, list) for g in grouping_level)
                 or not any(g in df.columns for g in grouping_level))):
            for g in grouping_level:
                grouping_set.append(self._get_grouping_level(df.columns, g, exclude='track_uid'))
            grouping_cols = max(grouping_set, key=len)
        else:
            grouping_cols = self._get_grouping_level(df.columns, grouping_level, exclude='track_uid')
            grouping_set = [grouping_cols]

        df = self.assign_track_uid(df)

        # Unique time points
        if df['time_point'].n_unique() < 2:
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEINTERVALS']})

        t_step = self._resolve_t_step(df, 'time interval stats')

        wanted = self._ti_registry.resolve(
            subset,
            descr=self.cat_descr,
            descr_err=self.cat_descr_err,
            infer_err=self.cat_infer_err,
        )

        def _compute_level(source: pl.DataFrame, grouping_cols: list[str]) -> pl.DataFrame:
            """Compute time-interval stats for a single grouping level."""

            temp = (
                source
                .select(grouping_cols + ['track_uid', 'time_point', 'x_coordinate', 'y_coordinate'])
                .sort(['track_uid', 'time_point'])
                .with_columns(
                    (pl.col('time_point').rank('dense').over('track_uid') - 1)
                    .cast(pl.Int64).alias('_frame'),
                    pl.len().over('track_uid').alias('_size'),
                )
                .filter(pl.col('_size') >= 2)
                .with_columns(
                    pl.arctan2(
                        (pl.col('y_coordinate') - pl.col('y_coordinate').shift(1)).over('track_uid'),
                        (pl.col('x_coordinate') - pl.col('x_coordinate').shift(1)).over('track_uid'),
                    ).alias('_theta')
                )
            )

            if is_empty(temp):
                return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEINTERVALS']})

            max_lag = int(temp['_frame'].max())
            if max_lag < 1:
                return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEINTERVALS']})

            uid_arr   = temp['track_uid'].to_numpy()
            frame_arr = temp['_frame'].to_numpy()
            x_arr     = temp['x_coordinate'].to_numpy()
            y_arr     = temp['y_coordinate'].to_numpy()
            theta_arr = temp['_theta'].to_numpy()
            cat_arrs  = {c: temp[c].to_numpy() for c in grouping_cols}

            # (track_uid, frame) -> row lookup so a lag pairs points exactly
            # `lag` frames apart (robust to gaps / non-uniform spacing).
            pos_of = {k: i for i, k in enumerate(zip(uid_arr.tolist(), frame_arr.tolist()))}

            msd_records: List[pl.DataFrame] = []
            turn_records: List[pl.DataFrame] = []

            for lag in range(1, max_lag + 1):
                partner_pos = np.fromiter(
                    (pos_of.get(k, -1) for k in zip(uid_arr.tolist(), (frame_arr + lag).tolist())),
                    dtype=np.int64, count=len(uid_arr)
                )
                valid_mask = partner_pos >= 0
                if not valid_mask.any():
                    continue

                valid_idx   = np.where(valid_mask)[0]
                partner_idx = partner_pos[valid_idx]

                dx = x_arr[partner_idx] - x_arr[valid_idx]
                dy = y_arr[partner_idx] - y_arr[valid_idx]

                msd_records.append(pl.DataFrame({
                    'track_uid': uid_arr[valid_idx],
                    **{c: cat_arrs[c][valid_idx] for c in grouping_cols},
                    'sq_disp':   dx * dx + dy * dy,
                    'frame_lag': np.full(valid_idx.size, lag, dtype=np.int64),
                    'time_lag':  np.full(valid_idx.size, lag * t_step, dtype=np.float64),
                }))

                # Turning angle: theta defined only where a preceding step exists
                theta_now = theta_arr[valid_idx]
                theta_par = theta_arr[partner_idx]
                turn_valid = (frame_arr[valid_idx] >= 1) & np.isfinite(theta_now) & np.isfinite(theta_par)

                if turn_valid.any():
                    ti = valid_idx[turn_valid]
                    pi = partner_idx[turn_valid]
                    dtheta = theta_arr[pi] - theta_arr[ti]
                    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
                    turn_records.append(pl.DataFrame({
                        'track_uid': uid_arr[ti],
                        **{c: cat_arrs[c][ti] for c in grouping_cols},
                        'dtheta':    dtheta,
                        'frame_lag': np.full(ti.size, lag, dtype=np.int64),
                        'time_lag':  np.full(ti.size, lag * t_step, dtype=np.float64),
                    }))

            if not msd_records:
                return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEINTERVALS']})

            all_msd = pl.concat(msd_records, how='vertical_relaxed')
            all_turn = (
                pl.concat(turn_records, how='vertical_relaxed')
                if turn_records
                else pl.DataFrame(schema={**{c: all_msd.schema[c] for c in grouping_cols},
                                          'track_uid': all_msd.schema['track_uid'],
                                          'dtheta': pl.Float64,
                                          'frame_lag': pl.Int64, 'time_lag': pl.Float64})
            )

            lag_group_cols = grouping_cols + ['frame_lag', 'time_lag']
            ctx = {
                'source': all_msd,
                'by': lag_group_cols,
                'turn_src': all_turn,
                'circ_cache': {},
                'extra_exprs': {},
            }
            lags = self._ti_registry.compute(wanted, ctx)

            # Drop columns that produced no data
            return self._drop_all_null_columns(lags) if not is_empty(lags) else lags

        level_frames = []
        for grouping_cols in grouping_set:

            _color_cols = [c for c in df.columns if c.endswith('color')]
            _color_stash = (
                df.select(grouping_cols + _color_cols).unique(subset=grouping_cols, keep='first')
                if _color_cols else None
            )

            level_df = _compute_level(df, grouping_cols)

            if is_empty(level_df):
                continue

            level_df = level_df.with_columns(
                pl.lit(str(grouping_cols[0])).alias('grouping_level')
            )

            if _color_stash is not None:
                level_df = level_df.join(_color_stash, on=grouping_cols, how='left')

            level_frames.append(level_df)

        if not level_frames:
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEINTERVALS']})

        out = pl.concat(level_frames, how='diagonal_relaxed')

        # JSON-safe cleanup
        out = out.with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c, dt in out.schema.items() if dt in (pl.Float32, pl.Float64)
        ])

        if self.significant_figures:
            out = self.signify(out)
        if self.decimal_places:
            out = self.norm_decimals(out)

        return out


    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _ensure_polars(df) -> pl.DataFrame:
        """Accept pandas DataFrames / Input wrappers transparently."""
        if isinstance(df, pl.DataFrame):
            return df
        if hasattr(df, 'df') and isinstance(getattr(df, 'df'), pl.DataFrame):
            return df.df  # loader's Input wrapper
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                return pl.from_pandas(df)
        except ImportError:
            pass
        raise TypeError(f"Expected a polars DataFrame, got {type(df).__name__}.")

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    def assign_track_uid(self, df: pl.DataFrame) -> pl.DataFrame:
        """Creates a unique track identifier `track_uid` by combining the category
        columns present in the DataFrame with `track_id`, so each physical track
        gets its own uid."""

        if 'track_uid' in df.columns:
            return df

        grouping_cols = [c for c in self.DEFAULT_CATEGORIES
                         if c in df.columns and c != 'track_uid']

        if 'track_id' in df.columns:
            grouping_cols = grouping_cols + ['track_id']

        if not grouping_cols:
            raise ColumnsNotFoundError(
                "Cannot create track_uid: no category columns or 'track_id' found."
            )

        keys = (
            df.select(grouping_cols)
            .unique(maintain_order=True)
            .with_row_index('track_uid')
            .with_columns(pl.col('track_uid').cast(pl.Int64))
        )
        return df.join(keys, on=grouping_cols, how='left')


    def _get_grouping_level(
        self,
        df_cols,
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
            grouping_cols = [grouping_cols[-1]]
        elif grouping_level == 'lowest':
            pass
        elif isinstance(grouping_level, str):
            idx = grouping_cols.index(grouping_level)
            grouping_cols = grouping_cols[:idx + 1]
        elif not isinstance(grouping_level, list):
            raise InvalidParameterValueError(f"Invalid grouping_level parameter: {grouping_level}. Must be a list of column names, an integer index, 'highest', 'lowest', or None.")

        for col in include:
            if col not in grouping_cols:
                grouping_cols.append(col)
            else:
                warnings.warn(message=f"Some columns in 'include' are already present in the default grouping columns: {grouping_cols}. They will be included only once.",
                              category=ConflictWarning, stacklevel=2)

        if not is_empty(exclude):
            grouping_cols = [col for col in grouping_cols if col not in exclude]
            if len(grouping_cols) == 0:
                grouping_cols = ['track_uid']

        return grouping_cols


    # -----------------------------------------------------------------------
    # Formatting helpers
    # -----------------------------------------------------------------------
    def format_digits(self, df: pl.DataFrame, *, sig_figs: int = None, decimals: int = None) -> pl.DataFrame:
        """Formats numeric values according to significant figures / decimals."""
        if sig_figs:
            df = self.signify(df, sig_figs=sig_figs)
        if decimals:
            df = self.norm_decimals(df, decimals=decimals)
        return df


    def signify(self, df: pl.DataFrame, *, sig_figs: int = None) -> pl.DataFrame:
        """Round numeric values to a number of significant figures."""
        if is_empty(df):
            return df

        if sig_figs is None:
            sig_figs = self.significant_figures

        valuer = Values()
        num_cols = [c for c, dt in df.schema.items() if dt.is_numeric()]
        return df.with_columns([
            pl.col(c).map_elements(
                lambda x: valuer.RoundSigFigs(x, sigfigs=sig_figs),
                return_dtype=pl.Float64,
            ).alias(c)
            for c in num_cols
        ])


    def norm_decimals(self, df: pl.DataFrame, decimals: int = None) -> pl.DataFrame:
        """Normalize decimal places across numeric columns."""
        if is_empty(df):
            return df

        if decimals is None:
            decimals = self.decimal_places

        float_cols = [c for c, dt in df.schema.items() if dt in (pl.Float32, pl.Float64)]
        return df.with_columns([pl.col(c).round(decimals).alias(c) for c in float_cols])


    def _general_agg_stats(self, df: pl.DataFrame, exclude: list[str], *, group_by: list[str] = ['track_uid']) -> pl.DataFrame:
        """Compute general aggregate statistics (min, max, mean, sd, sem, median)
        for numeric columns, grouped by `group_by`, excluding given columns."""

        exclude = [col for col in (exclude or []) if col != 'track_uid']

        try:
            num_cols = [
                c for c, dt in df.schema.items()
                if dt.is_numeric() and c not in exclude and c not in group_by and c != 'track_uid'
            ]

            if not num_cols:
                return df.select(group_by).unique(maintain_order=True)

            exprs = []
            for col in num_cols:
                exprs += [
                    pl.col(col).min().alias(f"{col} min"),
                    pl.col(col).max().alias(f"{col} max"),
                    pl.col(col).mean().alias(f"{col} mean"),
                    pl.col(col).std(ddof=1).alias(f"{col} sd"),
                    (pl.col(col).std(ddof=1) / pl.col(col).count().cast(pl.Float64).sqrt()).alias(f"{col} sem"),
                    pl.col(col).median().alias(f"{col} median"),
                ]

            return df.group_by(group_by, maintain_order=True).agg(exprs)

        except Exception as e:
            warnings.warn(message=f"Stats._general_agg_stats() encountered an error: {e}. Returning empty DataFrame. Traceback:\n{traceback.format_exc()}",
                          category=FailedWarning, stacklevel=2)
            return pl.DataFrame()


    def _describe_infer(self, df: pl.DataFrame, group_cols: list[str], *, stats: dict[str, str] | list[str] = None, **kwargs) -> pl.DataFrame:
        if not stats:
            return df

        exclude = kwargs.get('exclude', self._EXCLUDE_SUFFIXES)
        value_cols = [
            c for c, dt in df.schema.items()
            if c not in group_cols and dt.is_numeric()
            and not any(s in c for s in exclude)
        ]

        resolving = self.resolve(stats)
        resolving_circular = self.resolve(kwargs.get('circular_stats', {'mean': 'circ_mean', 'var': 'circ_var'}))

        exprs = []
        ci_cols = []
        for col in value_cols:
            circular = (any(t in col for t in ['direction', 'Directional', 'directional', 'Turn', 'turn'])
                        and not col.endswith('var'))
            resolver = resolving_circular if circular else resolving

            for stat_name, builder in resolver.items():
                if stat_name == 'ci':
                    name = f"per_{group_cols[-1].lower()}_{col}_{self.CI_STATISTIC}_ci{self.CONFIDENCE_LEVEL}"
                    exprs.append(pl.col(col).alias(f"__list_{name}"))
                    ci_cols.append((name, col))
                else:
                    exprs.append(builder(col).alias(f"per_{group_cols[-1].lower()}_{col}_{stat_name}"))

        grp_stats = df.group_by(group_cols, maintain_order=True).agg(exprs)

        for name, _ in ci_cols:
            bounds = [self.ci(np.asarray(v, dtype=float)) for v in grp_stats[f"__list_{name}"].to_list()]
            grp_stats = grp_stats.drop(f"__list_{name}").with_columns(
                pl.Series(f"{name}_low", [b[0] for b in bounds], dtype=pl.Float64),
                pl.Series(f"{name}_high", [b[1] for b in bounds], dtype=pl.Float64),
            )

        return df.join(grp_stats, on=group_cols, how='left')


    def resolve(self, agg_spec: dict[str, str] | list[str]) -> dict[str, Callable[[str], pl.Expr]]:
        """Resolves a list/dict of aggregation specs into a mapping of output
        labels to polars aggregation-expression builders."""

        def _builder(func_name: str) -> Callable[[str], pl.Expr]:
            if func_name in self._POLARS_BUILTINS:
                return lambda c, f=func_name: getattr(pl.col(c), f)()
            if func_name in self.CUSTOM_AGG_FUNCTIONS:
                return self.CUSTOM_AGG_FUNCTIONS[func_name]
            if func_name == 'ci':
                return 'ci'  # sentinel handled by callers
            raise ValueError(
                f"Unknown aggregation '{func_name}'. "
                f"Available: {sorted(self._POLARS_BUILTINS | set(self.CUSTOM_AGG_FUNCTIONS) | {'ci'})}"
            )

        resolved = {}
        if isinstance(agg_spec, list):
            for func_name in agg_spec:
                resolved[func_name] = _builder(func_name)
        elif isinstance(agg_spec, dict):
            for label, func_name in agg_spec.items():
                resolved[label] = _builder(func_name)
        return resolved


    def _insert_at_position(self, d: dict, key: Any, value: Any = None, *, where: int | str = 0) -> dict:
        """Insert a (key: value) pair into a dictionary at a specific position."""
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


    # -----------------------------------------------------------------------
    # Scalar / numpy statistics (unchanged semantics)
    # -----------------------------------------------------------------------
    def _wrap_pi(self, a: np.ndarray) -> np.ndarray:
        """Wrap angles in radians to the range [-π, π]."""
        return (a + np.pi) % (2*np.pi) - np.pi

    def _circ_mean(self, a) -> float:
        """Circular mean of angles in radians."""
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        return float(np.arctan2(s, c))

    def _circ_var(self, a) -> float:
        """Circular variance defined as 1 - R."""
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        return float(1.0 - np.hypot(s, c))

    def _q25(self, a) -> float:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, 25)) if a.size else np.nan

    def _q75(self, a) -> float:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, 75)) if a.size else np.nan


    def ci(self, a, *, n_resamples: int | None = None, confidence_level: float | None = None, **kwargs) -> tuple[float, float]:
        """Confidence interval via bootstrap. See original documentation."""

        method = kwargs.get('method', 'BCa')
        seed = 42  # Fixed seed for reproducibility

        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]

        if a.size < 2:
            return (np.nan, np.nan)

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
                              category=FailedWarning, stacklevel=2)
                return (np.nan, np.nan)


    def sem(self, x) -> float:
        """Standard error of the mean."""
        if isinstance(x, pl.Series):
            n = x.len() - x.null_count()
            if n < 2:
                return np.nan
            return x.std(ddof=1) / np.sqrt(n)
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < 2:
            return np.nan
        return np.std(x, ddof=1) / np.sqrt(n)


    def stat_units(self, col: str = None, *, time_unit: str = None, **kwargs) -> dict[str, str]:
        """Returns a dictionary mapping metric names to their corresponding units."""

        t_unit = time_unit if time_unit is not None else self.t_unit

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
    """Contains static methods utilized in the Peregrin Shiny App"""

    @staticmethod
    def dataframe_summary(df: pl.DataFrame) -> dict:
        missing = sum(df[c].null_count() for c in df.columns)
        return {
            "rows": df.height,
            "columns": df.width,
            "missing_cells": int(missing),
            "memory_mb": round(df.estimated_size() / 1e6, 2),
        }

    @staticmethod
    def column_summary(series: pl.Series) -> dict:
        if series.dtype.is_numeric():
            s = series.cast(pl.Float64, strict=False)
            s = s.set(s.is_infinite(), None)

            if s.len() - s.null_count() > 0:
                mode = s.drop_nulls().mode()
                return {
                    "type": "type_one",
                    "missing": int(series.null_count()),
                    "distinct": int(series.n_unique() - (1 if series.null_count() else 0)),
                    "min": s.min(),
                    "max": s.max(),
                    "mean": s.mean(),
                    "median": s.median(),
                    "mode": float(mode[0]) if mode.len() else None,
                    "sd": s.std(ddof=1),
                    "variance": s.var(),
                }

        vc = (
            series.drop_nulls()
            .value_counts(sort=True)
            .head(3)
        )
        total = series.len() - series.null_count()
        highest = [
            (row[0], round(row[1] / total * 100, 1)) for row in vc.iter_rows()
        ] if total else []

        return {
            "type": "type_zero",
            "missing": int(series.null_count()),
            "distinct": int(series.n_unique() - (1 if series.null_count() else 0)),
            "highest": highest,
        }



class Stats(Calc):
    """
    A stateful, callable interface over :class:`Calc`.

    Usage
    -----
    >>> stats = Stats()
    >>> stats = stats(spot_df)                 # computes & stores Spots_df
    >>> tracks_df = stats.compute_tracks()     # per-track statistics
    >>> frames_df = stats.compute_frames()     # per-time-point statistics
    >>> ti_df = stats.compute_time_intervals() # per-time-interval statistics
    >>> stats.plot_tracks()                    # reconstruct trajectories
    >>> stats.plot_msd(band='sem')             # MSD plot
    """

    def __init__(
        self,
        *,
        cat_descr: bool = True,
        cat_descr_err: bool = True,
        cat_infer_err: bool = False,
        bootstrap_ci: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            cat_descr=cat_descr,
            cat_descr_err=cat_descr_err,
            cat_infer_err=cat_infer_err,
            bootstrap_ci=bootstrap_ci,
            **kwargs,
        )

        self.spots_df: Optional[pl.DataFrame] = None
        self.tracks_df: Optional[pl.DataFrame] = None
        self.frames_df: Optional[pl.DataFrame] = None
        self.time_intervals_df: Optional[pl.DataFrame] = None

        self._categories: Optional[dict] = None

    def __repr__(self) -> str:
        n = None if self.spots_df is None else self.spots_df.height
        return f"<Stats object: spots_rows={n}>"

    def __call__(self, df: pl.DataFrame, **kwargs) -> "Stats":
        """Compute per-spot statistics from raw spot data and store them.
        Returns ``self`` so the call can be chained/re-bound."""

        # Accept the Input wrapper from the loader transparently
        if hasattr(df, 'df') and not isinstance(df, pl.DataFrame):
            df = df.df

        self.spots_df = self.spots(df, **kwargs)

        # Invalidate downstream caches on new input.
        self.tracks_df = None
        self.frames_df = None
        self.time_intervals_df = None

        return self

    def _resolve_spots(self, df: Optional[pl.DataFrame]) -> pl.DataFrame:
        source = df if df is not None else self.spots_df
        if source is None:
            raise ValueError("No Spots_df available. Call the Stats instance with a DataFrame first.")
        return source

    def compute_spots(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, **kwargs) -> pl.DataFrame:
        """Compute (and store) per-spot statistics."""
        source = df if df is not None else self.spots_df
        if source is None:
            raise ValueError("No input DataFrame provided for compute_spots().")
        self.spots_df = self.spots(source, subset=subset, **kwargs)
        return self.spots_df

    def compute_tracks(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, **kwargs) -> pl.DataFrame:
        """Compute (and store) per-track statistics from the stored Spots_df."""
        source = self._resolve_spots(df)
        self.tracks_df = self.tracks(source, subset=subset, **kwargs)
        return self.tracks_df

    def compute_frames(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, *, grouping_level: Any = 'highest', **kwargs) -> pl.DataFrame:
        """Compute (and store) per-time-point statistics from the stored Spots_df."""
        source = self._resolve_spots(df)
        self.frames_df = self.frames(source, subset=subset, grouping_level=grouping_level, **kwargs)
        return self.frames_df

    def compute_time_intervals(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, *, grouping_level: Any = 'highest', **kwargs) -> pl.DataFrame:
        """Compute (and store) per-time-interval statistics from the stored Spots_df."""
        source = self._resolve_spots(df)
        self.time_intervals_df = self.time_intervals(source, subset=subset, grouping_level=grouping_level, **kwargs)
        return self.time_intervals_df

    def compute_all(self, df: Optional[pl.DataFrame] = None, **kwargs):
        """Compute and store all four statistics DataFrames."""
        source = self._resolve_spots(df)
        self.spots_df = source
        self.compute_tracks()
        self.compute_frames(**kwargs)
        self.compute_time_intervals(**kwargs)
        return (self.spots_df, self.tracks_df, self.frames_df, self.time_intervals_df)

    # -----------------------------------------------------------------------
    # Plotting wrappers (lazy imports avoid circular deps)
    # -----------------------------------------------------------------------
    def plot_tracks(self, **kwargs):
        """Reconstruct and plot trajectories from the stored Spots_df."""
        from ..plot.tracks.reconstruct import reconstruct
        return reconstruct(self.spots_df, **kwargs)

    def plot_msd(self, band: Optional[str] = None, *, grouping_level: Any = 'highest', **kwargs):
        """Plot MSD from the stored Spots_df."""
        from ..plot.time.lags import msd
        return msd(self.spots_df, band=band, categories=self._categories, grouping_level=grouping_level, **kwargs)

    def plot_turn_angles(self, *, grouping_level: Any = 'highest', **kwargs):
        """Plot the turning-angle heatmap from the stored Spots_df."""
        from ..plot.time.lags import turn_angles
        return turn_angles(self.spots_df, grouping_level=grouping_level, **kwargs)


# input_metadata = InputMetadata()
calc = Calc()
stats = Stats()