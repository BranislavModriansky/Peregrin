from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from .stats import stats
from .._pckg_exceptions._pckg_warnings import *


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

        for file_name, metadata in self.input_metadata.items(): 

            all_time_units.add(metadata.get("time_units"))
            all_spatial_units.add(metadata.get("spatial_units"))
            all_n_frames.add(metadata.get("n_frames"))
            all_time_intervals.add(metadata.get("time_interval"))

        print("All time units:", all_time_units)
        print("All spatial units:", all_spatial_units)
        print("All n_frames:", all_n_frames)
        print("All time intervals:", all_time_intervals)
        




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

