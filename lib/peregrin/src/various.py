import math
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import *
from itertools import zip_longest
from functools import wraps

from ._pckg_exceptions._pckg_warnings import *
from ._pckg_exceptions._pckg_errors import *


class CheckData:

    def __init__(self):
        pass

    def is_empty(self, data: pd.DataFrame | pd.Series, *, details: bool = False) -> bool:
        """
        Checks if a pd.DataFrame or pd.Series is empty.
        """

        isempty = False
        
        if data is None or data.empty:
            isempty = True

        if details:
            self._get_details(isempty, data)
        
        return isempty


    def _get_details(self, isempty: bool, data: pd.DataFrame | pd.Series) -> None:
        """
        Print details of the DataFrame or Series.
        """

        if isempty:
            return

        if isinstance(data, pd.DataFrame):
            table = self._get_df_details(data)
        else:
            table = self._get_sr_details(data)

        self._print_table(table)


    def _print_table(self, table: dict) -> None:

        headers = list(table.keys())
        values = list(table.values())

        # Compute column widths
        col_widths = [
            max(len(headers[i]), max(len(v) for v in values[i]))
            for i in range(len(headers))
        ]

        # Headers
        header_line = "  ".join(
            headers[i].ljust(col_widths[i])
            for i in range(len(headers))
        )

        # Separators
        separator_line = "  ".join(
            "-" * col_widths[i]
            for i in range(len(col_widths))
        )

        print("")
        print(header_line)
        print(separator_line)

        # Values (shorter columns filled with empty strings)
        for row in zip_longest(*values, fillvalue=""):
            print(
                "  ".join(
                    row[i].rjust(col_widths[i])
                    for i in range(len(row))
                )
            )

        print("")

    def _get_df_details(self, df: pd.DataFrame) -> dict:
        """
        Get a summary of the DataFrame's properties.
        """

        df_shape = df.shape
        
        try:
            index_label = df.index.names if df.index.names is not None else "<unnamed>"
            index_type = df.index.dtypes
        except Exception:
            index_label = df.index.name if df.index.name is not None else "<unnamed>"
            index_type = df.index.dtype
            

        return {
            "MemoryMB": [f"{round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)}"],
            "Rows": [f"{df_shape[0]}"],
            "Columns": [f"{df_shape[1]}"],
            "ColumnLabels": list(df.columns),
            "IndexLabel": [f"{index_label}"],
            "IndexType": [f"{index_type}"],
            "MissingValues%": [f"{(df.isna().sum().sum() / (df_shape[0] * df_shape[1]) * 100):.2f}"],
            "RowDuplicates": [f"{df.duplicated().sum()}"],
            "ColumnDuplicates": [f"{df.columns.duplicated().sum()}"],
        }
    
    def _get_sr_details(self, series: pd.Series) -> dict:
        """
        Get a summary of the Series' properties.
        """

        return {
            "MemoryMB": [f"{round(series.memory_usage(deep=True) / (1024 ** 2), 2)}"],
            "Label": [series.name],
            "Length": [f"{len(series)}"],
            "MissingValues%": [f"{(series.isna().sum() / len(series) * 100):.2f}"],
            "Duplicates": [f"{series.duplicated().sum()}"],
        }
    
is_empty = CheckData().is_empty


def clock(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        finish = time.time()
        print("")
        print(f"Clocked: '{f.__name__}' <- {finish - start:.4f} s")
        print("")

        return result
    
    return wrap



class Values:

    @staticmethod
    def Clamp01(value: float, **kwargs) -> float:
        """
        Clamp a value between 0 and 1.
        """
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        if not (0.0 <= value <= 1.0):    

            if value < 0.0:
                clamped = 0
            else:
                clamped = 1

            if noticequeue:
                noticequeue.Report(Level.warning, f"{value} out of 0-1 range. Clamping to {clamped}.")

            return clamped
        
        return value
    
    @staticmethod
    def RoundSigFigs(x, sigfigs: int = 5, **kwargs) -> float:
        """
        Round a number to a given number of significant figures.

        Parameters
        ----------
        x : any
            The value to round.
        sigfigs : int
            Number of significant figures (default = 5).

        Returns
        -------
        int, float, or None
            Rounded value, or None if input is None.
        """

        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        if x is None:
            return None

        try:
            x = float(x)

        except (TypeError, ValueError) as e:
            if noticequeue: noticequeue.Report(Level.Error, f"Cannot convert {type(x)}: {x} to float.", str(e))
            return None
        
        except Exception as e:
            if noticequeue: noticequeue.Report(Level.Error, f"Error converting {type(x)}: {x} to float.", str(e))
            return None

        if math.isnan(x) or math.isinf(x):
            return x

        if x == 0.0:
            return 0.0

        return round(x, sigfigs - int(math.floor(math.log10(abs(x)))) - 1)
    

    @staticmethod
    def cmap_lut(data: pd.Series, *args, min: float = None, max: float = None, **kwargs) -> Tuple[Any, Any]:

        try:
            if not isinstance(min, (int, float)):
                min = float(data.min())
            if not isinstance(max, (int, float)):
                max = float(data.max())

            if not (np.isfinite(max) or np.isfinite(min)):
                warnings.warn(message=f"Invalid LUT range. Max and min values are not finite. Using default range (0.0, 100.0).", 
                              category=LUTWarning, 
                              stacklevel=2)

                if not np.isfinite(min):
                    min = 0.0
                if not np.isfinite(max):
                    max = 100.0
                    
            if max <= min:
                warnings.warn(message=f"Invalid LUT range. Max value must be greater than min value. Using default range (0.0, 100.0).", 
                              category=LUTWarning, 
                              stacklevel=2)
                
                min = 0.0
                max = 100.0
            
            norm = plt.Normalize(min, max)
            vals = data.to_numpy()

            return norm, vals
        
        except Exception as e:
            raise LUTError(f"Error while computing LUT map: {str(e)}")
        


class Kwargs:

    @staticmethod
    def get_kwarg(key: str, aliases: dict) -> str:
        """
        Get the canonical key for a given keyword argument.

        Parameters
        ----------
        key : str
            The keyword argument to check.
        aliases : dict
            A dictionary where keys are canonical names and values are lists of aliases.

        Returns
        -------
        str
            The canonical key if found, otherwise the original key.
        """
        for canonical, alias_list in aliases.items():
            if key in alias_list:
                return canonical
        return key

    @staticmethod
    def get_aliases(kwargs, aliases):
        """
        Resolves aliased kwargs to their canonical parameter names.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments as passed by the user, e.g. {'colour': 'black', 'line_width': 1}
        aliases : dict
            Maps canonical name -> list of accepted alias names (including the
            canonical name itself), e.g. {'color': ['color', 'colour', 'c'], ...}

        Returns
        -------
        dict
            kwargs with all recognized aliases rewritten to their canonical key.
            Keys not found in `aliases` are passed through unchanged, with a warning.
        """
        # Build a reverse lookup: alias -> canonical name
        alias_to_canonical = {
            alias: canonical
            for canonical, alias_list in aliases.items()
            for alias in alias_list
        }

        resolved = {}

        for key, value in kwargs.items():
            canonical = alias_to_canonical.get(key, key)
            resolved[canonical] = value

        return resolved


get_aliases = Kwargs.get_aliases
is_empty = CheckData().is_empty