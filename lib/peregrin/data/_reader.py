from dataclasses import dataclass
from pathlib import Path
import sys
import pandas as pd

# Works both as package module and as direct script
try:
    from ..src.loader.load import load_data
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from peregrin.src.loader.load import load_data

DATA_DIR = Path(__file__).resolve().parent



colnames = {'id': "TRACK_ID", 't': "POSITION_T", 'x': "POSITION_X", 'y': "POSITION_Y"}



_b_naive = load_data(
    files=[
        [
            str(DATA_DIR / "naive_ctr_BC39.csv"),
            str(DATA_DIR / "naive_ctr_BC42.csv"),
            str(DATA_DIR / "naive_ctr_BC43.csv")
        ],
        [
            str(DATA_DIR / "naive_cxcl12_BC39.csv"),
            str(DATA_DIR / "naive_cxcl12_BC42.csv"),
            str(DATA_DIR / "naive_cxcl12_BC43.csv")
        ],
        [
            str(DATA_DIR / "naive_mu_BC39.csv"),
            str(DATA_DIR / "naive_mu_BC42.csv"),
            str(DATA_DIR / "naive_mu_BC43.csv")
        ]
    ],
    colnames=colnames,
    cond_lbls=["ctr", "cxcl12", "mu"],
    rep_lbls=[['1', '2', '3'], ['1', '2', '3'], ['1', '2', '3']]
)

_b_memory = load_data(
    files=[
        [
            str(DATA_DIR / "memory_ctr_BC39.csv"),
            str(DATA_DIR / "memory_ctr_BC42.csv"),
            str(DATA_DIR / "memory_ctr_BC43.csv")
        ],
        [
            str(DATA_DIR / "memory_cxcl12_BC39.csv"),
            str(DATA_DIR / "memory_cxcl12_BC42.csv"),
            str(DATA_DIR / "memory_cxcl12_BC43.csv")
        ],
        [
            str(DATA_DIR / "memory_mu_BC39.csv"),
            str(DATA_DIR / "memory_mu_BC42.csv"),
            str(DATA_DIR / "memory_mu_BC43.csv")
        ],
    ],
    colnames=colnames,
    cond_lbls=["ctr", "cxcl12", "mu"],
    rep_lbls=[['1', '2', '3'], ['1', '2', '3'], ['1', '2', '3']]
)

class PeregrinDataFrame(pd.DataFrame):
    _metadata = ["ctr", "cxcl12", "mu"]

    @property
    def _constructor(self):
        return PeregrinDataFrame

    def build_subsets(self):
        self.ctr    = PeregrinDataFrame(self.loc[self["condition"] == "ctr"].copy())
        self.cxcl12 = PeregrinDataFrame(self.loc[self["condition"] == "cxcl12"].copy())
        self.mu     = PeregrinDataFrame(self.loc[self["condition"] == "mu"].copy())
        return self


b_naive = PeregrinDataFrame(_b_naive).build_subsets()
b_memory = PeregrinDataFrame(_b_memory).build_subsets()



