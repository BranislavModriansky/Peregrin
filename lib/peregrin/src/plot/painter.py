from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional, Tuple, Literal
from pandas.api.types import is_categorical_dtype, is_numeric_dtype

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

import warnings
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *


@dataclass
class Dyes:
    """
    Class holding color options.
    """

    _base_quantitative_cmaps = [
        'gist_grey',
        'gist_yarg',
        'viridis',
        'cividis',
        'plasma',
        'inferno',
        'magma',
        'gist_heat',
        'hot',
        'afmhot',
        'copper',
        'Wistia',
        'pink',
        'bone',
        'spring',
        'summer',
        'autumn',
        'winter',
        'cool',
        'ocean',
        'gist_earth',
        'terrain',
        'cubehelix',
        'CMRmap',
        'gnuplot2',
        'gnuplot',
        'gist_stern',
        'nipy_spectral',
        'gist_ncar',
        'brg',
        'jet',
        'turbo',
        'rainbow',
        'gist_rainbow',
        'twilight',
        'twilight_shifted',
        'hsv',
        'Purples',
        'Blues',
        'Greens',
        'Oranges',
        'Reds',
        'YlOrBr',
        'YlOrRd',
        'OrRd',
        'PuRd',
        'RdPu',
        'BuPu',
        'GnBu',
        'PuBu',
        'YlGnBu',
        'PuBuGn',
        'BuGn',
        'YlGn',
        'PiYG',
        'PRGn',
        'BrBG',
        'PuOr',
        'RdGy',
        'RdBu',
        'RdYlBu',
        'RdYlGn',
        'Spectral',
        'coolwarm',
        'bwr',
        'seismic',
        'berlin',
        'managua',
        'vanimo',
    ]

    quantitative_cmaps = []

    for _cmap in _base_quantitative_cmaps:
        quantitative_cmaps.append(_cmap)

        if f"{_cmap}_r" in mpl.colormaps:
            quantitative_cmaps.append(f"{_cmap}_r")

    del _base_quantitative_cmaps


    qualitative_palettes_matplotlib = [
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "Accent",
        "Dark2",
        "Pastel1",
        "Pastel2"
    ]

    qualitative_palettes_seaborn = [
        "deep", 
        "muted", 
        "bright", 
        "pastel", 
        "dark", 
        "colorblind", 
        "husl",
        "hsl"
    ]

    qualitative_palettes = qualitative_palettes_matplotlib + qualitative_palettes_seaborn

    all_cmaps = quantitative_cmaps + qualitative_palettes


class ColorGenerator:

    _HEX = np.array([f"{i:02x}" for i in range(256)], dtype="<U2")

    def __init__(self): ...

    def random_color(
        self,
        n: Optional[int] = 1,
        *,
        code: str = "hex",
        a: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        
        if not isinstance(n, int) or n < 1:
            raise ColorGeneratorError("n must be a positive integer.")
        
        rng = np.random.default_rng(kwargs.get("seed", 42))

        rgb = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
        return self._color_value(rgb, code=code, a=a)

    def random_grey(
        self,
        n: Optional[int] = 1,
        *,
        code: str = "hex",
        a: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        
        if not isinstance(n, int) or n < 1:
            raise ColorGeneratorError("n must be a positive integer.")
        
        rng = np.random.default_rng(kwargs.get("seed", 42))

        grey = rng.integers(0, 240, size=(n, 1), dtype=np.uint8)
        rgb = np.repeat(grey, 3, axis=1)
        return self._color_value(rgb, code=code, a=a)

    def _color_value(
        self,
        rgb: np.ndarray,
        *,
        code: str = "hex",
        a: float = 1.0,
    ) -> np.ndarray:
        
        rgb = np.asarray(rgb, dtype=np.uint8)

        if rgb.ndim == 1:
            rgb = rgb.reshape(1, -1)

        alpha = float(np.clip(a, 0.0, 1.0))

        match code:
            case "hex":
                alpha_hex = np.full((rgb.shape[0], 1), round(alpha * 255), dtype=np.uint8)
                rgba = np.hstack((rgb, alpha_hex))
                parts = self._HEX[rgba]

                out = np.char.add("#", parts[:, 0])
                out = np.char.add(out, parts[:, 1])
                out = np.char.add(out, parts[:, 2])
                out = np.char.add(out, parts[:, 3])
                return out

            case "rgb":
                return np.array(
                    [f"rgb({r}, {g}, {b})" for r, g, b in rgb],
                    dtype=object,
                )

            case "rgba":
                return np.array(
                    [f"rgba({r}, {g}, {b}, {alpha})" for r, g, b in rgb],
                    dtype=object,
                )

            case _:
                raise ValueError(
                    "Unsupported color code. Use one of: 'hex', 'rgb', 'rgba'."
                )
            
    @staticmethod
    def is_color_code(value, raise_on_out_of_range=True):
        """
        Return True if value is a valid color code.

        Supports:
        - Matplotlib color codes, including #RGB, #RGBA, #RRGGBB, #RRGGBBAA
        - CSS-like rgb(...) and rgba(...)
        """
        if not isinstance(value, str):
            return False

        s = value.strip()

        # Matplotlib already handles hex colors including #RRGGBBAA.
        if mcolors.is_color_like(s):
            return True

        rgb_pattern = re.compile(
            r"^rgb\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*\)$",
            re.IGNORECASE,
        )
        rgba_pattern = re.compile(
            r"^rgba\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*\)$",
            re.IGNORECASE,
        )

        def _channel_in_range(channel: str) -> bool:
            if channel.endswith("%"):
                n = float(channel[:-1])
                return 0.0 <= n <= 100.0
            n = float(channel)
            return 0.0 <= n <= 255.0

        def _alpha_in_range(alpha: str) -> bool:
            if alpha.endswith("%"):
                n = float(alpha[:-1])
                return 0.0 <= n <= 100.0
            n = float(alpha)
            return 0.0 <= n <= 1.0

        match = rgb_pattern.match(s)
        if match:
            if all(_channel_in_range(channel) for channel in match.groups()):
                return True
            if raise_on_out_of_range:
                raise InvalidColorRangeError(
                    f"'{s}' has the shape of an rgb() color but values are out of range "
                    f"(expected 0-255 or 0%-100% per channel)."
                )
            return False

        match = rgba_pattern.match(s)
        if match:
            *channels, alpha = match.groups()
            if all(_channel_in_range(channel) for channel in channels) and _alpha_in_range(alpha):
                return True
            if raise_on_out_of_range:
                raise InvalidColorRangeError(
                    f"'{s}' has the shape of an rgba() color but values are out of range "
                    f"(channels 0-255 or 0%-100%, alpha 0-1 or 0%-100%)."
                )
            return False

        return False



class Cmaps:

    def __init__(self): ...

    @staticmethod
    def retrieve_palette(categories: list, palette: Optional[str | list] = "tab10") -> dict:
        if isinstance(palette, list):
            if len(palette) < len(categories):
                raise PaletteBuilderError(f"More categories ({len(categories)}) than colors ({len(palette)}). Please provide a palette with at least as many colors as there are categories.")
            return {cat: color for cat, color in zip(categories, palette)}

        else:
            try:
                palette = plt.get_cmap(palette)
            except ValueError:
                palette = sns.color_palette(palette)
            except Exception as e:
                warnings.warn(message=f"An error occurred while retrieving the palette '{palette}': {str(e)}. <- Defaulting to 'tab10' colormap. Supported palettes include: {', '.join(sorted(mpl.colormaps.keys()))} for matplotlib; {', '.join(sorted(sns.palettes.SEABORN_PALETTES.keys()))} for seaborn or a list of colors.",
                            category=PaletteBuilderWarning,
                            stacklevel=2)
                palette = plt.get_cmap('tab10')
            cat_count = len(categories)
            return {cat: mcolors.to_hex(palette(i / cat_count)) for i, cat in enumerate(categories)}


    @staticmethod
    def retrieve_cmap(qnt_cmap: str | mcolors.Colormap) -> mcolors.Colormap:
        """
        Retrieve a quantitative colormap.
        """
        if isinstance(qnt_cmap, mcolors.Colormap):
            return qnt_cmap

        try:
            return mpl.colormaps[qnt_cmap]

        except Exception as e:
            warnings.warn(message=f"An error occurred while retrieving the colormap for '{qnt_cmap}': {str(e)}. Available colormaps are: {', '.join(Dyes.quantitative_cmaps)}. Defaulting to 'jet' colormap.",
                            category=PainterWarning,
                            stacklevel=2)
            
            return mpl.colormaps['jet']
        
    @staticmethod
    def scale_cmap(data: pd.Series, *, min: float = None, max: float = None) -> Tuple[Any, Any]:
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
                warnings.warn(message=f"Invalid LUT range. Max value must be greater than min value. Swapping values.", 
                            category=LUTWarning, 
                            stacklevel=2)
                
                min, max = max, min
            
            norm = plt.Normalize(min, max)
            vals = data.to_numpy()

            return norm, vals
        
        except Exception as e:
            raise LUTError(f"Error while computing LUT for {data.name if hasattr(data, 'name') else 'unknown data'}: {str(e)}")

        
    @staticmethod
    def showcase_colormaps(*, which: str = "quantitative", **kwargs) -> plt.Figure:
        """
        ### *Showcase colormaps.*
        
        Parameters
        ----------
        which : str, optional
            Specify the type of colormaps to showcase. Options are 'quantitative' or 'qualitative'. Default is 'quantitative'.

        Returns
        -------
        plt.Figure
            A figure showcasing the gradients of the specified type of colormaps.

        Kwargs
        ------
        **cmaps** : *list[str], default is None -> a custom list of colormap names to be showcased.*
        **text_color** : *str, default is 'black'.*
        **strip_background** : *bool, default is False.
        """

        text_color = kwargs.get('text_color', 'black')
        strip_background = kwargs.get('strip_background', False)

        if kwargs.get('cmaps') is None:
            match which:
                case "quantitative":
                    cmaps = Dyes.quantitative_cmaps
                case "qualitative":
                    cmaps = Dyes.qualitative_palettes
                case _:
                    raise ValueError(f"Unknown colormap type '{which}'. Supported types are 'quantitative' and 'qualitative'.")
        else:
            cmaps = kwargs.get('cmaps')

        n = len(cmaps)
        if n == 0:
            raise ValueError("No colormaps provided for showcasing. Please provide a list of colormap names or use the default 'quantitative' or 'qualitative' options.")

        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # Calculate figure height based on the number of colormaps to display
        height = 0.35 + 0.15 + (n + (n - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=n + 1, figsize=(6.4, height))

        # Adjust subplot parameters to create space for labels
        fig.subplots_adjust(
            top=1 - 0.35 / height, 
            bottom=0.15 / height,
            left=0.2, right=0.99
        )
        
        # Display the gradient for each colormap with its name as a label
        for ax, name in zip(axs, cmaps):
            ax.imshow(gradient, aspect='auto', cmap=Cmaps.retrieve_cmap(name))
            ax.text(
                -0.02, 0.5, name[:-4] if name.endswith(' LUT') else name, 
                va='center', ha='right', 
                fontsize=10, color=text_color,
                fontfamily='monospace',
                transform=ax.transAxes,
            )

        # Turn off all axes and spines for a clean look
        for ax in axs:
            ax.set_axis_off()
            
        if strip_background:
            fig.set_facecolor('none')
                
        return plt.gcf()
    

class Painter:

    def __init__(self): ...

    def paint(
        self,
        data: pd.DataFrame,
        *,
        color: Literal['random', 'random greys'] | str  = 'black',
        color_by: Optional[str | tuple[str, Literal['categorical', 'numeric']]] = None,
        **kwargs
    ) -> None:

        self.data = data
        self.color = color
        self.color_by = color_by
        self.kwargs = kwargs

        self.colors = None

        if self.color_by is not None:
            self._color_by()
        elif self.color in list(mcolors.CSS4_COLORS.keys()) or is_color_code(self.color):
            pass
        elif self.color is not None:
            self._color()

        if self.colors is not None:
            return self.colors
        else:
            return self.color


    def _color_by(self) -> np.ndarray:
        if self.color is not None:
            warnings.warn(
                "Both 'color' and 'color_by' parameters are provided -> Parameter "
                "'color' will be ignored -> Using 'color_by' for color assignment.",
                category=ConflictingParametersWarning, stacklevel=2)

        datatype = None
        if isinstance(self.color_by, tuple) and len(self.color_by) == 2:
            self.color_by, datatype = self.color_by
            if datatype not in ('categorical', 'numeric'):
                raise ValueError(
                    f"Invalid datatype parameter '{datatype}' for color_by. "
                    "Must be one of ['categorical', 'numeric'].")

        if datatype == 'categorical' or is_categorical_dtype(self.data[self.color_by]):
            self.colors = self._categorical_colors()
        elif datatype == 'numeric' or is_numeric_dtype(self.data[self.color_by]):
            self.colors = self._numeric_colors()
        else:
            raise InvalidParameterValueError(
                f"Invalid color_by value: '{self.color_by}'. Must be a column name in "
                "spot_data with categorical or numeric data or a tuple where the "
                "data type is specified (column_name, 'categorical'|'numeric').")

        self.color = None


    def _color(self) -> None:
        n = self.kwargs.get('n', self.data.index.nunique())
        if self.color == 'random':
            self.colors = random_color(n)
            self.color = None
        elif self.color == 'random greys':
            self.colors = random_grey(n)
            self.color = None
        else:
            raise InvalidParameterValueError(
                f"Invalid color parameter: {self.color}. Must be either a valid color name, hex "
                "code, or one of ['random', 'random greys'].")


    def _categorical_colors(self) -> np.ndarray:
        palette = self.kwargs.get('palette', 'tab10')
        if isinstance(palette, str) and palette not in Dyes.qualitative_palettes:
            warnings.warn(
                f"Palette '{palette}' is not a recognized qualitative palette. "
                "Defaulting to 'tab10'. Supported palettes include: "
                f"{', '.join(Dyes.qualitative_palettes)}.",
                category=PainterWarning, stacklevel=2)
            palette = 'tab10'
        elif isinstance(palette, list):
            categories = self.data[self.color_by].dropna().unique().tolist()
            palette = retrieve_palette(categories, palette)
        elif not isinstance(palette, dict):
            raise PlottingError(
                f"Invalid palette type: {type(palette)}. Must be str, list, or dict.")
        return self.data[self.color_by].map(palette).fillna("#000000FF").to_numpy()

    def _numeric_colors(self) -> np.ndarray:
        cmap_name = self.kwargs.get('cmap', 'viridis')
        if cmap_name not in Dyes.quantitative_cmaps:
            warnings.warn(
                f"Colormap '{cmap_name}' is not a recognized quantitative colormap. "
                "Defaulting to 'viridis'. Supported colormaps include: "
                f"{', '.join(Dyes.quantitative_cmaps)}.",
                category=PainterWarning, stacklevel=2)
            cmap = retrieve_cmap('viridis')
        else:
            cmap = retrieve_cmap(cmap_name)

        norm, vals = Cmaps.scale_cmap(
            self.data[self.color_by],
            min=self.kwargs.get('lut_vmin', self.data[self.color_by].min()),
            max=self.kwargs.get('lut_vmax', self.data[self.color_by].max()),
        )
        try:
            # RGBA (N,4) float array — far cheaper downstream than an object column.
            return cmap(norm(np.asarray(vals, dtype=float)))
        except Exception as e:
            raise PlottingError(
                f"Error applying quantitative colormap: '{cmap}' to data: {e}")



dyes = Dyes()

retrieve_palette = Cmaps.retrieve_palette
retrieve_cmap = Cmaps.retrieve_cmap
scale_cmap = Cmaps.scale_cmap

color_generator = ColorGenerator()
random_color = color_generator.random_color
random_grey = color_generator.random_grey
is_color_code = color_generator.is_color_code

paint = Painter().paint