from __future__ import annotations
import traceback

import math
from os import path
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from typing import Any, List, Tuple

import warnings
from ..settings import params
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *



class Categorizer:

    def __init__(self): ...


    def categorize(
        self,
        data: pd.DataFrame,
        sets: dict[str, Any] | None = None,
        *,
        aggby: list | None = None,
        aggdict: dict | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Categorize and aggregate data.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame to be categorized and aggregated

        sets : dict[str, Any], optional
            A dictionary containing the sets (categories) `{'column name': list}` to be included in the categorized DataFrame.

        aggby : list, optional
            A list of columns to group by for aggregation. Default is an empty list.
        
        aggdict : dict, optional
            A dictionary specifying the aggregation functions to apply to each column. Default is an empty dictionary.

        ignore_categories : bool, optional
            If True, the `conditions` and `replicates` parameters are ignored and all data are treated as a single group.
            If not specified, the default value is taken from the package settings. 
            To change the default configuration and behavior throughout all computations, use `peregrin.settings(ignore_categories=...)`
        
        """

        self.data = data
        self.sets = sets if sets is not None else {}
        self.aggby = aggby if aggby is not None else []
        self.aggdict = aggdict if aggdict is not None else {}

        if not params.ignore_categories or not kwargs.get('ignore_categories', False):
            self._checkcats()
            self._filter()

        if self.aggdict and self.aggby:
            self._aggregate()

        return self.data


    def _checkcats(self) -> bool:
        """ Check for errors in the provided categories and replicates. """

        for cat in self.sets.keys():
            if cat not in self.data.columns:
                raise CategorizerError(f"Column '{cat}' not found in DataFrame.")
            
            for val in self.sets[cat]:
                if val not in self.data[cat].unique():
                    raise CategorizerError(f"Value '{val}' not found in column '{cat}'.")
        

    def _filter(self) -> pd.DataFrame:
        """ Filter DataFrame categories. """

        for cat in self.sets.keys():
            self.data = self.data[self.data[cat].isin(self.sets[cat])]

    def _aggregate(self) -> pd.DataFrame:
        """ Aggregate the filtered DataFrame. """

        self.data = self.data.groupby(self.aggby).agg(self.aggdict).reset_index()



categorize = Categorizer().categorize