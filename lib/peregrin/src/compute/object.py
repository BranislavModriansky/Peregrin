from __future__ import annotations

from warnings import warn
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from .stats import stats
from .._pckg_exceptions._pckg_warnings import *
from .._pckg_exceptions._pckg_errors import *


@dataclass
class InputMetadata:
    """
    Input metadata container:
    -------------------------

    A dictionary mapping metadata to file names:

        {
            "file1.csv": {
                "spatial_units": str,
                "time_units": str, 
                "time_interval": float,
                "n_frames": int,
            },
            "file2.csv": ...,
        }
    """

    input_metadata: Dict[str, dict] = field(default_factory=dict)

    UNIT_ALIASES = {
        'ms': ['ms', 'millisecond', 'milliseconds'],
        's': ['s', 'sec', 'second', 'seconds'],
        'min': ['m', 'min', 'minute', 'minutes'],
        'h': ['h', 'hr', 'hour', 'hours'],
        'd': ['d', 'day', 'days'],
        'px': ['px', 'pixel', 'pixels'],
        'μm': ['μm', 'um', 'micron', 'microns', 'micrometer', 'micrometers'],
        'mm': ['mm', 'millimeter', 'millimeters'],
        'cm': ['cm', 'centimeter', 'centimeters'],
    }

    def __init__(self, input_metadata: Optional[Dict[str, dict]] = None):
        if input_metadata is not None:
            self.input_metadata = input_metadata

    def update(self, metadata: Dict[str, dict]):
            
        self.input_metadata.update(metadata)

    def get(self, file_name: str) -> Optional[dict]:
        return self.input_metadata.get(file_name)

    def check(self) -> None:
        all_time_units = set()
        all_spatial_units = set()
        all_n_frames = set()
        all_time_intervals = set()

        for _, metadata in self.input_metadata.items(): 
            all_time_units.add(metadata.get("timeunits"))
            all_spatial_units.add(metadata.get("spatialunits"))
            all_n_frames.add(metadata.get("nframes"))
            all_time_intervals.add(metadata.get("timeinterval"))

        if len(all_time_units) > 1:
            warn("Inconsistent time units across input files.", InputWarning, stacklevel=2)
        if len(all_spatial_units) > 1:
            warn("Inconsistent spatial units across input files.", InputWarning, stacklevel=2)
        if len(all_n_frames) > 1:
            warn("Inconsistent number of frames across input files.", InputWarning, stacklevel=2)
        if len(all_time_intervals) > 1:
            raise InputError("Inconsistent time intervals across input files. Please ensure that all input files have the same time interval.")




@dataclass
class SpotsObject:
    """A DataFrame container carrying metadata, with attached compute/plot methods."""

    data: pd.DataFrame
    units: Dict[str, str] = field(default_factory=dict)
    kind: Optional[str] = None  # e.g. 'spots', 'tracks', 'frames', 'timeintervals'

    # --- Metadata helpers ---------------------------------------------------
    def unit(self, column: str) -> Optional[str]:
        return self.units.get(column)

    # --- Chained computations (return new Object instances) -----------------
    def compute_timestats(self, **kwargs) -> "SpotsObject":
        result = stats.time_intervals(self.data, **kwargs)
        return SpotsObject(
            data=result,
            units=self.units,
            kind="timeintervals",
        )

    def compute_tracks(self, **kwargs) -> "SpotsObject":
        result = stats.tracks(self.data, **kwargs)
        return SpotsObject(
            data=result,
            units=self.units,
            meta={"derived_from": self.kind},
            kind="tracks",
        )

    # --- Plotting (delegates, returns whatever the plotter returns) ---------
    def plot_tracks(self, **kwargs):
        from ..plot.tracks.reconstruct import reconstruct
        return reconstruct(self.data, **kwargs)

    # --- Convenience passthroughs ------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Fall through to the underlying DataFrame for unknown attributes.
        return getattr(object.__getattribute__(self, "data"), name)

    def _repr_html_(self):
        return self.data._repr_html_()

@dataclass
class TracksObject(SpotsObject): ...






input_metadata = InputMetadata()
