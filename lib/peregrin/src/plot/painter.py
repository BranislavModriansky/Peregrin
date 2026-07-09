from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional, Tuple

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

import warnings
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *



class Painter:

    def __init__(self): ...


    @dataclass
    class Dyes:
        """
        Class holding color options for the UI.
        """

        BaseCModes = [
            "single color",
            "random colors",
            "random greys",
            "differentiate conditions",
            "differentiate replicates"
        ]

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




        PaletteQualitativeMatplotlib = [
            "Set1",
            "Set2",
            "Set3",
            "tab10",
            "Accent",
            "Dark2",
            "Pastel1",
            "Pastel2"
        ]

        PaletteQualitativeSeaborn = [
            "deep", 
            "muted", 
            "bright", 
            "pastel", 
            "dark", 
            "colorblind", 
            "husl",
            "hsl"
        ]

        colors = {
            "#000000": "black",
            "#0f0f0f": "onyx",
            "#070d0d": "deep ocean",
            "#1a1a1a": "space",
            "#343837": "charcoal",
            "#3c4142": "charcoal grey",
            "#363737": "dark grey",
            "#4a4e4d": "dim grey",
            "#536267": "gunmetal",
            "#59656d": "slate grey",
            "#516572": "slate",
            "#929591": "grey",
            "#738595": "steel",
            "#7d7f7c": "medium grey",
            "#95a3a6": "cool grey",
            "#c5c9c7": "silver",
            "#d8dcd6": "light grey",
            "#ffffe4": "off white",
            "#ffffff": "white",
            "#4a0100": "mahogany",
            "#770001": "blood",
            "#840000": "dark red",
            "#650021": "maroon",
            "#610023": "burgundy",
            "#680018": "claret",
            "#80013f": "wine",
            "#7b0323": "wine red",
            "#8c0034": "red wine",
            "#7b002c": "bordeaux",
            "#980002": "blood red",
            "#9a0200": "deep red",
            "#a90308": "darkish red",
            "#8c000f": "crimson",
            "#af2f0d": "rusty red",
            "#8f1402": "brick red",
            "#9f2305": "burnt red",
            "#8b2e16": "red brown",
            "#a83c09": "rust",
            "#a03623": "brick",
            "#fe0002": "fire engine red",
            "#e50000": "red",
            "#ff000d": "bright red",
            "#f7022a": "cherry red",
            "#cf0234": "cherry",
            "#be0119": "scarlet",
            "#f4320c": "vermillion",
            "#ec2d01": "tomato red",
            "#ef4026": "tomato",
            "#fa4224": "orangey red",
            "#fd3c06": "red orange",
            "#ff073a": "neon red",
            "#fb2943": "strawberry",
            "#ff474c": "light red",
            "#d9544d": "pale red",
            "#fe2f4a": "lightish red",
            "#f10c45": "pinkish red",
            "#be013c": "rose red",
            "#850e04": "indian red",
            "#9d0216": "carmine",
            "#3d1c02": "chocolate",
            "#411900": "chocolate brown",
            "#341c02": "dark brown",
            "#1d0200": "very dark brown",
            "#985e2b": "sepia",
            "#a6814c": "coffee",
            "#653700": "brown",
            "#583101": "saddle brown",
            "#a13905": "russet",
            "#a9561e": "sienna",
            "#b04e0f": "burnt sienna",
            "#a0450e": "burnt umber",
            "#b26400": "umber",
            "#c45508": "rust orange",
            "#c04e01": "burnt orange",
            "#e17701": "pumpkin",
            "#fb7d07": "pumpkin orange",
            "#be6400": "orange brown",
            "#b96902": "brown orange",
            "#bf9005": "ochre",
            "#c65102": "dark orange",
            "#f97306": "orange",
            "#ff5b00": "bright orange",
            "#fe4b03": "blood orange",
            "#fe420f": "orangered",
            "#f8481c": "reddish orange",
            "#ff9408": "tangerine",
            "#f9bc08": "mandarin",
            "#fdaa48": "light orange",
            "#ffb07c": "peach",
            "#ffb16d": "apricot",
            "#ff7855": "melon",
            "#ff796c": "salmon",
            "#fe7b7c": "salmon pink",
            "#fc5a50": "coral",
            "#ff6163": "coral pink",
            "#ff964f": "pastel orange",
            "#f0944d": "faded orange",
            "#c9643b": "terra cotta",
            "#ca6641": "terracotta",
            "#b66a50": "clay",
            "#b2713d": "clay brown",
            "#b66325": "copper",
            "#a87900": "bronze",
            "#af6f09": "caramel",
            "#fdb147": "butterscotch",
            "#d1b26f": "tan",
            "#e2ca76": "sand",
            "#c9ae74": "sandstone",
            "#e6daa6": "beige",
            "#fef69e": "buff",
            "#fbdd7e": "wheat",
            "#aaa662": "khaki",
            "#c69f59": "camel",
            "#cfaf7b": "fawn",
            "#b9a281": "taupe",
            "#ac9362": "dark beige",
            "#8a6e45": "dirt",
            "#836539": "dirt brown",
            "#735c12": "mud",
            "#60460f": "mud brown",
            "#f9bc08": "golden rod",
            "#fac205": "goldenrod",
            "#dbb40c": "gold",
            "#b59410": "dark gold",
            "#ceb301": "mustard",
            "#d2bd0a": "mustard yellow",
            "#d5b60a": "dark yellow",
            "#b79400": "yellow brown",
            "#cb9d06": "yellow ochre",
            "#c2b709": "olive yellow",
            "#b5a642": "brass",
            "#ffff14": "yellow",
            "#fdff63": "canary",
            "#fffe40": "canary yellow",
            "#fdff52": "lemon",
            "#fdff38": "lemon yellow",
            "#fffd01": "bright yellow",
            "#ffc512": "sunflower",
            "#ffda03": "sunflower yellow",
            "#ffdf22": "sun yellow",
            "#fff917": "sunny yellow",
            "#fffd37": "sunshine yellow",
            "#ffff7e": "banana",
            "#fafe4b": "banana yellow",
            "#ffff81": "butter",
            "#fffd74": "butter yellow",
            "#ffffc2": "cream",
            "#ffff84": "pale yellow",
            "#fffe7a": "light yellow",
            "#fffe71": "pastel yellow",
            "#ffffcb": "ivory",
            "#ffffd4": "eggshell",
            "#6e750e": "olive",
            "#373e02": "dark olive",
            "#6f7632": "olive drab",
            "#677a04": "olive green",
            "#4b5d16": "army green",
            "#667c3e": "military green",
            "#698339": "swamp",
            "#748500": "swamp green",
            "#7f8f4e": "camo",
            "#526525": "camo green",
            "#4b6113": "camouflage green",
            "#828344": "drab",
            "#749551": "drab green",
            "#ada587": "stone",
            "#769958": "moss",
            "#658b38": "moss green",
            "#3c4d03": "dark olive green",
            "#0b5509": "forest",
            "#06470c": "forest green",
            "#002d04": "dark forest green",
            "#033500": "dark green",
            "#0b4008": "hunter green",
            "#044a05": "bottle green",
            "#05472a": "evergreen",
            "#2b5d34": "pine",
            "#0a481e": "pine green",
            "#014600": "racing green",
            "#05480d": "british racing green",
            "#048243": "jungle green",
            "#01a049": "emerald",
            "#028f1e": "emerald green",
            "#01b44c": "shamrock",
            "#02c14d": "shamrock green",
            "#15b01a": "green",
            "#089404": "true green",
            "#5cac2d": "grass",
            "#3f9b0b": "grass green",
            "#4da409": "lawn green",
            "#388004": "dark grass green",
            "#63a950": "fern",
            "#548d44": "fern green",
            "#71aa34": "leaf",
            "#5ca904": "leaf green",
            "#51b73b": "leafy green",
            "#2a7e19": "tree green",
            "#01ff07": "bright green",
            "#2fef10": "vivid green",
            "#21fc0d": "electric green",
            "#0cff0c": "neon green",
            "#08ff08": "fluorescent green",
            "#25ff29": "hot green",
            "#aaff32": "lime",
            "#89fe05": "lime green",
            "#87fd05": "bright lime",
            "#65fe08": "bright lime green",
            "#8ffe09": "acid green",
            "#c1f80a": "chartreuse",
            "#c0fb2d": "yellow green",
            "#bbf90f": "yellowgreen",
            "#c9ff27": "green yellow",
            "#6ecb3c": "apple",
            "#76cd26": "apple green",
            "#5edc1f": "green apple",
            "#02ab2e": "kelly green",
            "#019529": "irish green",
            "#58bc08": "frog green",
            "#84b701": "dark lime",
            "#7ebd01": "dark lime green",
            "#a4bf20": "pea",
            "#8eab12": "pea green",
            "#929901": "pea soup",
            "#94a617": "pea soup green",
            "#90b134": "avocado",
            "#87a922": "avocado green",
            "#77ab56": "asparagus",
            "#87ae73": "sage",
            "#88b378": "sage green",
            "#789b73": "grey green",
            "#96f97b": "light green",
            "#c7fdb5": "pale green",
            "#b0ff9d": "pastel green",
            "#9ffeb0": "mint",
            "#8fff9f": "mint green",
            "#b6ffbb": "light mint",
            "#a6fbb2": "light mint green",
            "#1ef876": "spearmint",
            "#80f9ad": "seafoam",
            "#7af9ab": "seafoam green",
            "#a0febf": "light seafoam",
            "#a7ffb5": "light seafoam green",
            "#90fda9": "foam green",
            "#a9f971": "spring green",
            "#c1fd95": "celery",
            "#befdb7": "celadon",
            "#d1ffbd": "very light green",
            "#cffdbc": "very pale green",
            "#029386": "teal",
            "#014d4e": "dark teal",
            "#00555a": "deep teal",
            "#25a36f": "teal green",
            "#01889f": "teal blue",
            "#06b48b": "green blue",
            "#137e6d": "blue green",
            "#005249": "dark blue green",
            "#53fca1": "sea green",
            "#11875d": "dark sea green",
            "#98f6b0": "light sea green",
            "#3d9973": "ocean green",
            "#1e9167": "viridian",
            "#1fa774": "jade",
            "#2baf6a": "jade green",
            "#06c2ac": "turquoise",
            "#045c5a": "dark turquoise",
            "#0ffef9": "bright turquoise",
            "#04f489": "turquoise green",
            "#06b1c4": "turquoise blue",
            "#13eac9": "aqua",
            "#05696b": "dark aqua",
            "#08787f": "deep aqua",
            "#0bf9ea": "bright aqua",
            "#8cffdb": "light aqua",
            "#04d8b2": "aquamarine",
            "#7bfdc7": "light aquamarine",
            "#017371": "dark aquamarine",
            "#00ffff": "cyan",
            "#41fdfe": "bright cyan",
            "#acfffc": "light cyan",
            "#b7fffa": "pale cyan",
            "#6dedfd": "robin's egg",
            "#98eff9": "robin's egg blue",
            "#8af1fe": "robin egg blue",
            "#7bf2da": "tiffany blue",
            "#d6fffa": "ice",
            "#d7fffe": "ice blue",
            "#01153e": "navy",
            "#001146": "navy blue",
            "#000435": "dark navy",
            "#00022e": "dark navy blue",
            "#000133": "very dark blue",
            "#03012d": "midnight",
            "#020035": "midnight blue",
            "#00035b": "dark blue",
            "#040273": "deep blue",
            "#0504aa": "royal blue",
            "#02066f": "dark royal blue",
            "#0c1793": "royal",
            "#1b2431": "dark steel blue",
            "#1e488f": "cobalt",
            "#030aa7": "cobalt blue",
            "#004577": "prussian blue",
            "#042e60": "marine",
            "#01386a": "marine blue",
            "#017b92": "ocean",
            "#03719c": "ocean blue",
            "#015482": "deep sea blue",
            "#2138ab": "sapphire",
            "#0343df": "blue",
            "#0203e2": "pure blue",
            "#010fcc": "true blue",
            "#152eff": "vivid blue",
            "#0165fc": "bright blue",
            "#0652ff": "electric blue",
            "#04d9ff": "neon blue",
            "#0c06f7": "strong blue",
            "#0804f9": "primary blue",
            "#021bf9": "rich blue",
            "#247afd": "clear blue",
            "#0339f8": "vibrant blue",
            "#069af3": "azure",
            "#0485d1": "cerulean",
            "#056eee": "cerulean blue",
            "#75bbfd": "sky blue",
            "#02ccfe": "bright sky blue",
            "#0d75f8": "deep sky blue",
            "#448ee4": "dark sky blue",
            "#c6fcff": "light sky blue",
            "#bdf6fe": "pale sky blue",
            "#95d0fc": "light blue",
            "#a2cffe": "baby blue",
            "#b1d1fc": "powder blue",
            "#d0fefe": "pale blue",
            "#d5ffff": "very light blue",
            "#8e82fe": "periwinkle",
            "#8f99fb": "periwinkle blue",
            "#c1c6fc": "light periwinkle",
            "#6a79f7": "cornflower",
            "#5170d7": "cornflower blue",
            "#3b638c": "denim",
            "#3b5b92": "denim blue",
            "#5a7d9a": "steel blue",
            "#4e7496": "cadet blue",
            "#607c8e": "blue grey",
            "#6b8ba4": "grey blue",
            "#5b7c99": "slate blue",
            "#214761": "dark slate blue",
            "#4e518b": "twilight",
            "#0a437a": "twilight blue",
            "#464196": "blueberry",
            "#380282": "indigo",
            "#1f0954": "dark indigo",
            "#3a18b1": "indigo blue",
            "#7e1e9c": "purple",
            "#35063e": "dark purple",
            "#36013f": "deep purple",
            "#280137": "midnight purple",
            "#4b006e": "royal purple",
            "#580f41": "plum",
            "#4e0550": "plum purple",
            "#3f012c": "dark plum",
            "#380835": "eggplant",
            "#3d0734": "aubergine",
            "#6c3461": "grape",
            "#5d1451": "grape purple",
            "#9a0eea": "violet",
            "#34013f": "dark violet",
            "#490648": "deep violet",
            "#5d06e9": "blue violet",
            "#510ac9": "violet blue",
            "#632de9": "purple blue",
            "#5729ce": "blue purple",
            "#5539cc": "blurple",
            "#6832e3": "burple",
            "#aa23ff": "electric purple",
            "#bc13fe": "neon purple",
            "#ad0afd": "bright violet",
            "#9900fa": "vivid purple",
            "#be03fd": "bright purple",
            "#cb00f5": "hot purple",
            "#c20078": "magenta",
            "#960056": "dark magenta",
            "#a0025c": "deep magenta",
            "#ff08e8": "bright magenta",
            "#f504c9": "hot magenta",
            "#c875c4": "orchid",
            "#9e43a2": "medium purple",
            "#9b5fc0": "amethyst",
            "#a87dc2": "wisteria",
            "#cea2fd": "lilac",
            "#e4cbff": "pale lilac",
            "#edc8ff": "light lilac",
            "#c95efb": "bright lilac",
            "#c79fef": "lavender",
            "#eecffe": "pale lavender",
            "#dfc5fe": "light lavender",
            "#8b88f8": "lavender blue",
            "#bf77f6": "light purple",
            "#b790d4": "pale purple",
            "#ae7181": "mauve",
            "#fed0fc": "pale mauve",
            "#c292a1": "light mauve",
            "#874c62": "dark mauve",
            "#a57e52": "puce",
            "#a484ac": "heather",
            "#d8bfd8": "thistle",
            "#ff81c0": "pink",
            "#ffd1df": "light pink",
            "#ffcfdc": "pale pink",
            "#ffb7ce": "baby pink",
            "#ffbacd": "pastel pink",
            "#fdb0c0": "soft pink",
            "#cf6275": "rose",
            "#f7879a": "rose pink",
            "#fdc1c5": "pale rose",
            "#ffc5cb": "light rose",
            "#c0737a": "dusty rose",
            "#c87f89": "old rose",
            "#c77986": "old pink",
            "#cb416b": "dark pink",
            "#cb0162": "deep pink",
            "#ff028d": "hot pink",
            "#fe019a": "neon pink",
            "#fe02a2": "shocking pink",
            "#fe01b1": "bright pink",
            "#ff0490": "electric pink",
            "#ed0dd9": "fuchsia",
            "#9d0759": "dark fuchsia",
            "#de0c62": "cerise",
            "#b00149": "raspberry",
            "#990f4b": "berry",
            "#920a4e": "mulberry",
            "#f29e8e": "blush",
            "#fe828c": "blush pink",
            "#fd798f": "carnation",
            "#ff7fa7": "carnation pink",
            "#ff6cb5": "bubblegum",
            "#fe83cc": "bubblegum pink",
            "#fe46a5": "barbie pink",
            "#ff63e9": "candy pink",
            "#e78ea5": "pig pink",
            "#f255db": "flamingo",
            "#fb5ffc": "violet pink",
            "#a50055": "violet red",
            "#9e0168": "red violet",
            "#e03fd8": "purple pink",
            "#db4bda": "pink purple"
        }
        
        Background = [
            "white",
            "light",
            "mid",
            "dark",
            "black"
        ]

        LineStyle = [
            "solid",
            "dashed",
            "dotted",
            "dashdot",
        ]

        # def __post_init__(self):

        #     for _cmap in self._base_quantitative_cmaps:
        #         self.quantitative_cmaps.append(_cmap)

        #         if f"{_cmap}_r" in mpl.colormaps:
        #             self.quantitative_cmaps.append(f"{_cmap}_r")

        #     del _cmap, self._base_quantitative_cmaps



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



    class QualPaletteGenerator:

        def __init__(self): ...

        
        def retrieve_palette(
            self,
            categories: list,
            palette: Optional[str | list] = "tab10",
        ) -> dict:
            
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
                    cmap = plt.get_cmap('tab10')
                        
                cat_count = len(categories)
                return {cat: mcolors.to_hex(cmap(i / cat_count)) for i, cat in enumerate(categories)}


    class Cmap:

        def __init__(self): ...

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
                warnings.warn(message=f"An error occurred while retrieving the colormap for '{qnt_cmap}': {str(e)}. Available colormaps are: {', '.join(Painter.Dyes.quantitative_cmaps)}. Defaulting to 'jet' colormap.",
                              category=PainterWarning,
                              stacklevel=2)
                
                return mpl.colormaps['jet']
            
        @staticmethod
        def cmap_lut(data: pd.Series, *, min: float = None, max: float = None) -> Tuple[Any, Any]:
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
            



    def ShowcaseGradients(self, *, cmaps: list[str] = Dyes.quantitative_cmaps, **kwargs) -> plt.Figure:
        """
        ### *Showcase qualitative colormaps.*
        
        Parameters
        ----------
        cmaps : list[str], optional
            List of colormaps to showcase.

        Returns
        -------
        plt.Figure
            A figure showcasing the gradients of qualitative colormaps.
        """

        text_color = kwargs.get('text_color', 'black')
        strip_background = kwargs.get('strip_background', False)

        n = len(cmaps)

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
            ax.imshow(gradient, aspect='auto', cmap=self.GetCmap(name))
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



painter = Painter()
retrieve_palette = painter.QualPaletteGenerator().retrieve_palette
retrieve_cmap = painter.Cmap().retrieve_cmap
cmap_lut = painter.Cmap().cmap_lut
random_color = painter.ColorGenerator().random_color
random_grey = painter.ColorGenerator().random_grey
is_color_code = painter.ColorGenerator().is_color_code
dyes = painter.Dyes()