import pytest
import pandas as pd
import numpy as np

from ..data import naive_ctr, naive_cxcl12, naive_mu
from ..src.compute.stats import stats


@pytest.fixture(scope="module")
def stats():
    return stats


@pytest.fixture(scope="module")
def all_conditions():
    return [naive_ctr, naive_cxcl12, naive_mu]


# --- Spots ---

class TestSpots:

    def test_spots_returns_dataframe(self, stats):
        result = stats.spots(naive_ctr)
        assert isinstance(result, pd.DataFrame)

    def test_spots_not_empty(self, stats):
        result = stats.spots(naive_ctr)
        assert not result.empty

    def test_spots_required_columns(self, stats):
        result = stats.spots(naive_ctr)
        expected = [
            'condition', 'replicate', 'track_id', 'track_uid',
            'time_point', 'frame', 'x_coordinate', 'y_coordinate',
            'distance', 'cum_track_length', 'cum_track_displacement',
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_spots_no_negative_distances(self, stats):
        result = stats.spots(naive_ctr)
        assert (result['distance'].dropna() >= 0).all()

    def test_spots_cum_track_length_nonnegative(self, stats):
        result = stats.spots(naive_ctr)
        assert (result['cum_track_length'].dropna() >= 0).all()

    def test_spots_all_conditions(self, stats, all_conditions):
        for df in all_conditions:
            result = stats.spots(df)
            assert not result.empty


# --- Tracks ---

class TestTracks:

    def test_tracks_returns_dataframe(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.tracks(spots)
        assert isinstance(result, pd.DataFrame)

    def test_tracks_not_empty(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.tracks(spots)
        assert not result.empty

    def test_tracks_one_row_per_track(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.tracks(spots)
        assert result['track_uid'].nunique() == len(result)

    def test_tracks_required_columns(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.tracks(spots)
        expected = [
            'condition', 'replicate', 'track_id', 'track_uid',
            'track_length', 'track_displacement', 'straightness_ratio',
            'speed_mean',
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_tracks_straightness_ratio_bounded(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.tracks(spots)
        sr = result['straightness_ratio'].dropna()
        assert (sr >= 0).all()
        assert (sr <= 1.01).all()  # small tolerance for floating point


# --- Frames ---

class TestFrames:

    def test_frames_returns_dataframe(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.frames(spots)
        assert isinstance(result, pd.DataFrame)

    def test_frames_not_empty(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.frames(spots)
        assert not result.empty

    def test_frames_required_columns(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.frames(spots)
        for col in ['condition', 'replicate', 'time_point', 'frame']:
            assert col in result.columns, f"Missing column: {col}"


# --- TimeIntervals ---

class TestTimeIntervals:

    def test_time_intervals_returns_dataframe(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.time_intervals(spots)
        assert isinstance(result, pd.DataFrame)

    def test_time_intervals_not_empty(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.time_intervals(spots)
        assert not result.empty

    def test_time_intervals_required_columns(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.time_intervals(spots)
        for col in ['condition', 'replicate', 'frame_lag', 'time_lag']:
            assert col in result.columns, f"Missing column: {col}"

    def test_time_intervals_lag_starts_at_one(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.time_intervals(spots)
        assert result['frame_lag'].min() == 1

    def test_time_intervals_msd_nonnegative(self, stats):
        spots = stats.spots(naive_ctr)
        result = stats.time_intervals(spots)
        if 'per_replicate_msd_mean' in result.columns:
            assert (result['per_replicate_msd_mean'].dropna() >= 0).all()


# --- get_all --- 

class TestGetAll:

    def test_get_all_returns_four_dataframes(self, stats):
        result = stats.get_all(naive_cxcl12)
        assert len(result) == 4
        for df in result:
            assert isinstance(df, pd.DataFrame)

    def test_get_all_none_empty(self, stats):
        for df in stats.get_all(naive_mu):
            assert not df.empty

    @pytest.mark.parametrize("condition", [naive_ctr, naive_cxcl12, naive_mu])
    def test_get_all_all_conditions(self, stats, condition):
        spots, tracks, frames, time_intervals = stats.get_all(condition)
        assert not spots.empty
        assert not tracks.empty
        assert not frames.empty
        assert not time_intervals.empty

