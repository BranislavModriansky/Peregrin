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
        conditions: list | None = None,
        replicates: list | None = None,
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

        conditions : list, optional
            A list of conditions to be included in the categorized DataFrame.
            If `ignore_categories` is set to True, this parameter will be ignored.

        replicates : list, optional
            A list of replicates to be included in the categorized DataFrame.
            If `ignore_categories` is set to True, this parameter will be ignored.

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
        self.conditions = conditions if conditions is not None else []
        self.replicates = replicates if replicates is not None else []
        self.aggby = aggby if aggby is not None else []
        self.aggdict = aggdict if aggdict is not None else {}

        if not params.ignore_categories:
            self._checkcats()
            self._filter()

        if self.aggdict and self.aggby:
            self._aggregate()

        return self.data


    def _checkcats(self) -> bool:
        """ Check for errors in the provided categories and replicates. """

        if self.conditions == []:
            warnings.warn(message="Conditions not specified. <- Returning all conditions.", 
                          category=CategorizerWarning,
                          stacklevel=2)
            self.conditions = self.data['condition'].unique().tolist()

        if self.replicates == []:
            warnings.warn(message="Replicates not specified. <- Returning all replicates.", 
                          category=CategorizerWarning,
                          stacklevel=2)
            self.replicates = self.data['replicate'].unique().tolist()
        
        conds_not_found = [cond for cond in self.conditions if cond not in self.data['condition'].values]
        reps_not_found = [rep for rep in self.replicates if rep not in self.data['replicate'].values]
        if conds_not_found:
            warnings.warn(message=f"Couldn't find conditions: {', '.join(conds_not_found)}. <- Returning empty DataFrame.", 
                          category=CategorizerWarning,
                          stacklevel=2)
            return pd.DataFrame()
        if reps_not_found:
            warnings.warn(message=f"Couldn't find replicates: {', '.join(reps_not_found)}. <- Returning empty DataFrame.", 
                          category=CategorizerWarning,
                          stacklevel=2)
            return pd.DataFrame()
        

    def _filter(self) -> pd.DataFrame:
        """ Filter DataFrame categories. """

        if self.replicates:
            self.data = self.data[
                (self.data['condition'].isin(self.conditions)) &
                (self.data['replicate'].isin(self.replicates))
            ]
        else:
            self.data = self.data[self.data['condition'].isin(self.conditions)]

    def _aggregate(self) -> pd.DataFrame:
        """ Aggregate the filtered DataFrame. """

        self.data = self.data.groupby(self.aggby).agg(self.aggdict).reset_index()



categorize = Categorizer().categorize