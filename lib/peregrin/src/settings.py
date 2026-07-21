from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class Parameters:
    """
    Package settings.

    Attributes
    ----------
    ignore_categories : bool, default False
        If True, categories will be ignored in the statistical analysis. This setting a part of the package configuration and 
        can be set by the user.
    """

    ignore_categories: bool = False

    # colors: dict[int, str] = {
    #     0: '#477fc2',
    #     1: '#39383f',
    #     2: '#a1b6ca',
    #     3: '#c9cd6a',
    #     4: '#3b5575'
    # }

    def settings(self, **kwargs: Any) -> None:
        """ Update package settings. """

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid setting: {key}. Valid settings are: {', '.join(self.__dataclass_fields__.keys())}")
            
    def __post_init__(self):
        if not isinstance(self.ignore_categories, bool):
            raise TypeError("ignore_categories must be a boolean value.")
        
        

params = Parameters()
settings = params.settings