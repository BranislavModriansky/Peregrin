from __future__ import annotations
import re

import xml.etree.ElementTree as ET
import polars as pl
import numpy as np
import os.path as op
from typing import Dict, List, Optional, Tuple
from os import PathLike

from warnings import warn
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *


from ..various import get_aliases
from ..compute.stats import calc
import io
from urllib.request import urlopen


class Input:
    """A thin wrapper around a polars DataFrame that carries a `.metadata` (InputMetadata) attribute."""

    def __init__(self, df: pl.DataFrame, metadata: "InputMetadata" = None):
        self._df = df
        self.metadata = metadata

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    def __getattr__(self, name):
        # delegate everything else to the underlying polars DataFrame
        return getattr(self._df, name)

    def __getitem__(self, key):
        return self._df[key]

    def __len__(self):
        return len(self._df)

    def __repr__(self):
        return repr(self._df)


class InputMetadata:
    """
    Input metadata container:
    -------------------------

    Keeps a dictionary (1) of the metadata unified across all files.

        {
            "spatial_units": str,
            "time_units": str, 
            "time_interval": float,
            "n_frames": int,
        }

    and a dictionary (2) mapping metadata to files:

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

    input_metadata_unified: Dict[str, str] = {}
    input_metadata_separate: Dict[str, dict] = {}

    UNIT_ALIASES = {
        'nm': ['nm', 'nanometer', 'nanometers'],
        'μm': ['μm', 'um', 'micron', 'microns', 'micrometer', 'micrometers'],
        's': ['s', 'sec', 'second', 'seconds'],
        'min': ['m', 'min', 'minute', 'minutes'],
        'ms': ['ms', 'millisecond', 'milliseconds'],
        'h': ['h', 'hr', 'hour', 'hours'],
        'd': ['d', 'day', 'days'],
        'px': ['px', 'pixel', 'pixels'],
        'mm': ['mm', 'millimeter', 'millimeters'],
        'cm': ['cm', 'centimeter', 'centimeters'],
        'm': ['m', 'meter', 'meters'],
    }

    def __init__(self):
        self.input_metadata_unified = {
            'spatialunits': '',
            'timeunits': '',
            'timeinterval': '',
            'nframes': '',
            'columns': '',
        }
        self.input_metadata_separate = {}

    def __getitem__(self, key):
        return self.input_metadata_separate[key]

    def __setitem__(self, key, value):
        self.input_metadata_separate[key] = value

    def update(self, metadata: Dict[str, dict]):
        for key, item in metadata.items():
            for sub_key, sub_item in item.items():
                if not isinstance(sub_item, str):
                    continue  # e.g. 'columns' list, numeric timeinterval/nframes
                for unit, aliases in self.UNIT_ALIASES.items():
                    if sub_item in aliases:
                        metadata[key][sub_key] = unit
                        break

        self.input_metadata_separate.update(metadata)
        self._check()

    def get(self, metadata_key: str = None) -> Dict[str, str] | str:
        if metadata_key is not None:
            return self.input_metadata_unified.get(metadata_key)
        return self.input_metadata_unified

    def get_each(self, file_name: Optional[str] = None, metadata_keys: Optional[List[str]] = None) -> Optional[dict]:
        if file_name is None:
            if metadata_keys is not None:
                return {file: {
                    key: metadata.get(key) for key in metadata_keys
                } for file, metadata in self.input_metadata_separate.items()}
            return self.input_metadata_separate
        else:
            if metadata_keys is not None:
                return {key: self.input_metadata_separate[file_name].get(key) for key in metadata_keys}
            return self.input_metadata_separate[file_name]

    def write(self, *, spatialunits = None, timeunits = None, timeinterval = None, nframes = None, columns = None):
        self.input_metadata_unified["spatialunits"] = self._get_alias(spatialunits) if spatialunits is not None else self.input_metadata_unified["spatialunits"]
        self.input_metadata_unified["timeunits"] = self._get_alias(timeunits) if timeunits is not None else self.input_metadata_unified["timeunits"]
        self.input_metadata_unified["timeinterval"] = timeinterval if timeinterval is not None else self.input_metadata_unified["timeinterval"]
        self.input_metadata_unified["nframes"] = nframes if nframes is not None else self.input_metadata_unified["nframes"]
        self.input_metadata_unified["columns"] = columns if columns is not None else self.input_metadata_unified["columns"]

    def _get_alias(self, unit: str) -> str:
        for alias, aliases in self.UNIT_ALIASES.items():
            if unit in aliases:
                return alias
        raise ValueError(f"Unit '{unit}' is not recognized. Please use one of the following units: {list(self.UNIT_ALIASES.keys())}")

    def _check(self) -> None:
        all_time_units = set()
        all_spatial_units = set()
        all_n_frames = set()
        all_time_intervals = set()
        columns_set = set()

        for _, metadata in self.input_metadata_separate.items(): 
            all_time_units.add(metadata.get("timeunits"))
            all_spatial_units.add(metadata.get("spatialunits"))
            all_n_frames.add(metadata.get("nframes"))
            all_time_intervals.add(metadata.get("timeinterval"))
            columns_set.add(tuple(metadata.get("columns")) if metadata.get("columns") is not None else None)

        first_metadata = next(iter(self.input_metadata_separate.values()))
        self.input_metadata_unified["timeunits"] = first_metadata.get("timeunits")
        self.input_metadata_unified["spatialunits"] = first_metadata.get("spatialunits")
        self.input_metadata_unified["nframes"] = first_metadata.get("nframes")
        self.input_metadata_unified["timeinterval"] = first_metadata.get("timeinterval")
        self.input_metadata_unified["columns"] = first_metadata.get("columns")


        if self.input_metadata_unified.get("timeunits") is None or self.input_metadata_unified.get("timeunits") == '':
            warn("No time units found in input files.\n Please specify the time units using <load_data result>.metadata.write(time_unit=\"<unit>\")", InputWarning, stacklevel=2)
        if self.input_metadata_unified.get("spatialunits") is None or self.input_metadata_unified.get("spatialunits") == '':
            warn("No spatial units found in input files.\n Please specify the spatial units using <load_data result>.metadata.write(spatial_unit=\"<unit>\")", InputWarning, stacklevel=2)
        
        if len(all_time_units) > 1:
            self.input_metadata_unified["timeunits"] = ''
            warn(f"Inconsistent time units across input files -> found {all_time_units}.", InputWarning, stacklevel=2)
        if len(all_spatial_units) > 1:
            self.input_metadata_unified["spatialunits"] = ''
            warn(f"Inconsistent spatial units across input files -> found {all_spatial_units}.", InputWarning, stacklevel=2)
        if len(all_n_frames) > 1:
            self.input_metadata_unified["nframes"] = ''
            warn(f"Inconsistent number of frames across input files -> found {all_n_frames}.", InputWarning, stacklevel=2)
        if len(all_time_intervals) > 1:
            raise InputError(f"Inconsistent time intervals across input files -> found {all_time_intervals}. Please ensure that all input files have the same time interval.")

        if len(columns_set) > 1:
            self.input_metadata_unified["columns"] = ''
            warn(f"Inconsistent columns across input files -> found {len(columns_set)} distinct schemas.", InputWarning, stacklevel=2)


class DataLoader:

    CATEGORIES = {
        'set': 5,
        'subset': 4,
        'group': 3,
        'subgroup': 2,
        'subsubgroup': 1
    }

    def __init__(self):
        pass

    def load_data(
        self, 
        files: PathLike[str] | list | dict = None,
        colnames: dict = {
            'id': 'TRACK_ID', 
            't': 'POSITION_T', 
            'x': 'POSITION_X', 
            'y': 'POSITION_Y'
        }, 
        *,
        retain_cols: Optional[list[str]] = None,
        **kwargs
    ) -> pl.DataFrame:

        """
        Load tracking data from any number of files into a single DataFrame, 
        while assigning condition and replicate labels. (See original docstring
        for the full description; behaviour is unchanged but the returned
        object is now polars-based.)
        """

        self.retain = retain_cols
        self.kwargs = kwargs

        # Per-load metadata container
        self.metadata = InputMetadata()

        self.id_col = colnames['id']
        self.t_col  = colnames['t']
        self.x_col  = colnames['x']
        self.y_col  = colnames['y']

        # Time / transform options (wired as kwargs)
        self.mirror_y = kwargs.get('mirror_y', False)
        self.mirror_x = kwargs.get('mirror_x', False)
        self.merge = kwargs.get('merge', 'all')

        # Wrap single file into a list for uniform handling
        if isinstance(files, str):
            files = [files]

        if isinstance(files, list):
            depth = self._max_list_depth(files)
            leaves = self._iter_list_tree(files, depth)
        elif isinstance(files, dict):
            depth = self._max_dict_depth(files)
            leaves = self._iter_dict_tree(files, depth)
        else:
            raise TypeError("`files` must be a str, list or dict.")

        # Category columns used (bottom-up), 'subsubgroup' is always the file level
        category_order = ['set', 'subset', 'group', 'subgroup', 'subsubgroup']
        used_categories = category_order[:depth]

        records = []
        for labels, filepath in leaves:
            df, metadata = self._read_file(filepath)
            if not self.kwargs.get('ignore_metadata', False):
                self.metadata.update(self._filter_metadata(metadata))
            df = self._extract(df)

            df = df.with_columns([
                pl.lit(label).alias(cat)
                for cat, label in zip(used_categories, labels)
            ])

            records.append(df)

        result = self._merge(records, used_categories)

        return self._attach_metadata(result)


    def _attach_metadata(self, result):
        """Attach the per-load Metadata object to the merged result."""
        if isinstance(result, pl.DataFrame):
            result = Input(result, self.metadata)
        elif isinstance(result, dict):
            self._attach_metadata_dict(result)
        return result

    def _attach_metadata_dict(self, node):
        for key, val in node.items():
            if isinstance(val, pl.DataFrame):
                node[key] = Input(val, self.metadata)
            elif isinstance(val, dict):
                self._attach_metadata_dict(val)


    def _merge(self, records: list, used_categories: list):
        """
        Merge extracted per-file DataFrames according to `self.merge`.

        'all'        -> single concatenated DataFrame
        'sets'       -> dict keyed by set
        'subsets'    -> dict keyed by set/subset
        'groups'     -> dict keyed down to group
        'subgroups'  -> dict keyed down to subgroup
        None         -> dict keyed down to subsubgroup (unconcatenated leaves)
        """
        if not records:
            return pl.DataFrame()

        if self.merge == 'all':
            return pl.concat(records, how='diagonal_relaxed')

        merge_levels = {
            'sets': 1,
            'subsets': 2,
            'groups': 3,
            'subgroups': 4,
            None: 5,
        }
        target = merge_levels.get(self.merge)
        if target is None:
            raise ValueError(
                f"Invalid merge option: {self.merge!r}. "
                "Choose from 'all', 'sets', 'subsets', 'groups', 'subgroups', None."
            )

        full_order = ['set', 'subset', 'group', 'subgroup', 'subsubgroup']
        key_cats = [c for c in full_order[:target] if c in used_categories]

        result = {}
        for df in records:
            keys = [str(df[c][0]) for c in key_cats]

            node = result
            for k in keys[:-1]:
                node = node.setdefault(k, {})

            last = keys[-1]
            if self.merge is None:
                node[last] = df
            else:
                node.setdefault(last, []).append(df)

        if self.merge is not None:
            self._concat_leaves(result)

        return result

    def _concat_leaves(self, node):
        """Recursively concatenate lists of DataFrames stored in a nested dict."""
        for key, val in node.items():
            if isinstance(val, list):
                node[key] = pl.concat(val, how='diagonal_relaxed')
            elif isinstance(val, dict):
                self._concat_leaves(val)


    def _iter_list_tree(self, tree: list, depth: int, labels: tuple = ()):
        """
        Yield (labels, filepath) leaves from a nested list tree.
        Non-file levels are labelled with their 1-based index (as str);
        the file level is labelled with the file's basename.
        """
        if depth == 1:
            for path in tree:
                yield labels + (self._subsubgroup_label(path),), path
        else:
            for i, sub in enumerate(tree):
                yield from self._iter_list_tree(sub, depth - 1, labels + (str(i + 1),))

    def _iter_dict_tree(self, tree, depth: int, labels: tuple = ()):
        """
        Yield (labels, filepath) leaves from a nested dict tree.
        Non-file levels are labelled with their dict key;
        the file level is labelled with the file's basename.
        """
        if depth == 1:
            if isinstance(tree, dict):
                for _, path in tree.items():
                    yield labels + (self._subsubgroup_label(path),), path
            else:  # list of file paths
                for path in tree:
                    yield labels + (self._subsubgroup_label(path),), path
        else:
            for key, sub in tree.items():
                yield from self._iter_dict_tree(sub, depth - 1, labels + (str(key),))
            

    def _max_list_depth(self, obj):
        if not isinstance(obj, list) or not obj:
            return 0
        return 1 + self._max_list_depth(obj[0])

    def _max_dict_depth(self, d):
        if not isinstance(d, dict) or not d:
            return 0
        return 1 + max(self._max_dict_depth(v) for v in d.values())


    def _subsubgroup_label(self, path: str) -> str:
        """
        Derive the subsubgroup (file-level) label from a file path.
        """
        name = op.basename(path)
        stem, _ = op.splitext(name)

        if not self.kwargs.get('split_filename', False):
            return name

        sep = self.kwargs.get('split_filename_delimiter', '-')
        how = self.kwargs.get('split_filename_position', 'last')

        if isinstance(how, int):
            parts = stem.split(sep)
            if -len(parts) <= how < len(parts):
                return parts[how]
            return stem  # index out of range -> fall back to full stem

        if how == 'first':
            return stem.split(sep, 1)[0]

        if how == 'last':
            return stem.rsplit(sep, 1)[-1] if sep in stem else stem

        return stem


    def _extract(self, df: pl.DataFrame) -> pl.DataFrame:

        essential = [self.id_col, self.t_col, self.x_col, self.y_col]

        if not all(col in df.columns for col in essential):
            missing = [col for col in essential if col not in df.columns]
            raise ColumnsNotFoundError(f"missing columns: {missing}")

        if self.retain is None:
            df = df.select([
                pl.col(c).cast(pl.Float64, strict=False) for c in essential
            ])
        else:
            df = df.select(essential + list(self.retain)).with_columns([
                pl.col(c).cast(pl.Float64, strict=False) for c in essential
            ])
            df = self._py_numeric_df(df)

        df = df.drop_nulls(subset=essential)
        
        if self.mirror_y:
            y_mid = (df[self.y_col].min() + df[self.y_col].max()) / 2
            df = df.with_columns((2 * y_mid - pl.col(self.y_col)).alias(self.y_col))
        if self.mirror_x:
            x_mid = (df[self.x_col].min() + df[self.x_col].max()) / 2
            df = df.with_columns((2 * x_mid - pl.col(self.x_col)).alias(self.x_col))
            
        df = self._standname(df, self.id_col, self.t_col, self.x_col, self.y_col)

        if self.kwargs.get('convert_spatial_to', None) not in (None, ""):
            df = self._convert_spatial(df)
        if self.kwargs.get('convert_time_to', None) not in (None, ""):
            df = self._convert_time(df)
         
        return df


    def _read_file(self, filepath: str) -> Tuple[pl.DataFrame, dict]:

        _, ext = op.splitext(str(filepath).lower().split('?')[0])

        match ext:
            case '.xml':
                return self._read_trackmate_xml(filepath)
            case '.csv' | '.xls' | '.xlsx':
                return self._read_table(filepath, ext)
            case _:
                raise FileFormatError(f"{ext} is not supported. Supported formats include: .csv, .xls, .xlsx, .xml")

    @staticmethod
    def _fetch(filepath):
        """Return a local path or, for URLs, the file downloaded once into memory."""
        if not (isinstance(filepath, str) and filepath.startswith(('http://', 'https://'))):
            return filepath

        errors = []

        # 1) stdlib
        try:
            with urlopen(filepath) as resp:
                return io.BytesIO(resp.read())
        except Exception as e:
            errors.append(f"urllib: {e}")

        # 2) requests (bundles certifi)
        try:
            import requests
            r = requests.get(filepath, timeout=60)
            r.raise_for_status()
            return io.BytesIO(r.content)
        except Exception as e:
            errors.append(f"requests: {e}")

        # 3) urllib3 with certifi
        try:
            import urllib3, certifi
            http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
            r = http.request("GET", filepath, timeout=60.0)
            if r.status != 200:
                raise IOError(f"HTTP {r.status}")
            return io.BytesIO(r.data)
        except Exception as e:
            errors.append(f"urllib3: {e}")

        raise InputError(
            f"Could not download {filepath}. Attempts:\n  " + "\n  ".join(errors)
        )

    def _read_trackmate_xml(self, filepath) -> Tuple[pl.DataFrame, dict]:
        """Parse a TrackMate project XML and return spots and a merged metadata dict."""
        source = self._fetch(filepath)
        root = ET.parse(source).getroot()
        model = root.find('Model')

        spatialunits = model.attrib.get('spatialunits')
        timeunits = model.attrib.get('timeunits')

        dim_to_unit = {
            'POSITION': spatialunits, 'LENGTH': spatialunits,
            'TIME': timeunits, 'VELOCITY': f'{spatialunits}/{timeunits}',
            'RATE': f'1/{timeunits}', 'ANGLE_RATE': f'rad/{timeunits}',
            'INTENSITY': 'counts', 'ANGLE': 'rad',
            'QUALITY': 'quality', 'NONE': '', 'COST': '', 'STRING': '',
        }

        # feature -> unit, from <FeatureDeclarations>
        metadata = {}
        feature_declarations = model.find('FeatureDeclarations')
        if feature_declarations is not None:
            for category in feature_declarations:
                for feat in category:
                    metadata[feat.attrib['feature']] = dim_to_unit.get(feat.attrib.get('dimension'))

        # spots -> one DataFrame straight from the attribute dicts
        rows = [
            spot.attrib
            for frame in model.find('AllSpots')
            for spot in frame
        ]
        df = pl.DataFrame(rows)
        df = df.with_columns(pl.col('ID').cast(pl.Int64))
        num_cols = [c for c in df.columns if c not in ('name', 'ID')]
        df = df.with_columns([
            pl.col(c).cast(pl.Float64, strict=False) for c in num_cols
        ])

        # tag each spot with its TRACK_ID via the track edges
        src_ids, trk_ids = [], []
        for track in model.find('AllTracks'):
            tid = int(track.attrib['TRACK_ID'])
            for edge in track:
                a = edge.attrib
                src_ids += [a['SPOT_SOURCE_ID'], a['SPOT_TARGET_ID']]
                trk_ids += [tid, tid]

        if src_ids:
            mapping = (
                pl.DataFrame({
                    'ID': np.asarray(src_ids, dtype=np.int64),
                    'TRACK_ID': np.asarray(trk_ids, dtype=np.int64),
                })
                .unique(subset='ID', keep='first')
            )
            df = df.join(mapping, on='ID', how='left')
        else:
            df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias('TRACK_ID'))

        # move TRACK_ID to the first column
        df = df.select(['TRACK_ID'] + [c for c in df.columns if c != 'TRACK_ID'])

        # merge image calibration into the same metadata dict
        image_data = root.find('Settings/ImageData')
        if image_data is not None:
            metadata.update(image_data.attrib)

        metadata['spatialunits'] = spatialunits if self.kwargs.get('spatial_unit', None) is None else self.kwargs.get('spatial_unit')
        metadata['timeunits'] = timeunits if self.kwargs.get('time_unit', None) is None else self.kwargs.get('time_unit')
        metadata['timeinterval'] = self._calculate_time_interval(df) if self.t_col in df.columns else float('nan')
        metadata['nframes'] = df[self.t_col].n_unique() if self.t_col in df.columns else None
        metadata['columns'] = df.columns

        self.timeunit = metadata['timeunits']
        self.spatialunit = metadata['spatialunits']
        self.timeinterval = metadata['timeinterval']

        return df, {op.basename(str(filepath)): metadata}

    def _read_table(self, filepath, ext, *, metadata_row_index=2, skiprows=4,
                    encodings=("utf-8", "cp1252", "latin1", "iso8859_15"), **kwargs) -> Tuple[pl.DataFrame, dict]:
        try:
            source = self._fetch(filepath)

            if ext in ('.xls', '.xlsx'):
                column_names, units_row, df = self._read_excel_parts(
                    source, metadata_row_index, skiprows
                )
                metadata = {op.basename(str(filepath)): self._build_metadata(df, column_names, units_row)}
                return df, metadata

            # CSV path: try multiple encodings
            for enc in encodings:
                try:
                    if isinstance(source, io.BytesIO):
                        source.seek(0)
                    header = pl.read_csv(source, n_rows=0, encoding=enc)
                    column_names = header.columns

                    metadata_row = None
                    if metadata_row_index is not None:
                        if isinstance(source, io.BytesIO):
                            source.seek(0)
                        meta_df = pl.read_csv(
                            source, skip_rows=metadata_row_index, n_rows=1,
                            encoding=enc, has_header=True
                        )
                        metadata_row = list(meta_df.row(0)) if meta_df.height else None

                    if isinstance(source, io.BytesIO):
                        source.seek(0)
                    df = pl.read_csv(
                        source,
                        skip_rows=skiprows,
                        has_header=False,
                        new_columns=column_names,
                        encoding=enc,
                        infer_schema_length=10000,
                        ignore_errors=True,
                    )

                    metadata = {op.basename(str(filepath)): self._build_metadata(df, column_names, metadata_row)}
                    return df, metadata

                except UnicodeDecodeError:
                    continue

            raise FileFormatError(
                f"Failed to decode CSV file: {filepath}. Tried encodings: {encodings}."
            )

        except Exception as e:
            raise FileFormatError(f"{str(e)} -> Failed to read file: {filepath}.")


    def _read_excel_parts(self, filepath, metadata_row_index, skiprows) -> Tuple[List[str], Optional[List[str]], pl.DataFrame]:
        """Read header, units row, and data body from an Excel file."""
        header = pl.read_excel(filepath, read_options={"n_rows": 0})
        column_names = header.columns

        metadata_row = None
        if metadata_row_index is not None:
            meta_df = pl.read_excel(
                filepath,
                read_options={"skip_rows": metadata_row_index, "n_rows": 1, "has_header": False},
            )
            metadata_row = list(meta_df.row(0)) if meta_df.height else None

        df = pl.read_excel(
            filepath,
            read_options={"skip_rows": skiprows, "has_header": False, "new_columns": column_names},
        )
        return column_names, metadata_row, df


    def _build_metadata(self, df: pl.DataFrame, column_names, metadata_row) -> Dict[str, Optional[str]]:
        """Build a {column: unit} mapping from a raw metadata row."""
        metadata = {}
        if metadata_row is None:
            return {col: None for col in column_names}

        for col, u in zip(column_names, metadata_row):
            if isinstance(u, str):
                u = u.strip()
                if u.startswith('(') and u.endswith(')'):
                    u = u[1:-1]
            else:
                u = ''
            metadata[col] = u

        metadata['spatialunits'] = metadata.get(self.x_col, '') if self.kwargs.get('spatial_unit', None) is None else self.kwargs.get('spatial_unit')
        metadata['timeunits'] = metadata.get(self.t_col, '') if self.kwargs.get('time_unit', None) is None else self.kwargs.get('time_unit')
        metadata['timeinterval'] = self._calculate_time_interval(df)
        metadata['nframes'] = df[self.t_col].n_unique()
        metadata['columns'] = column_names

        self.timeunit = metadata['timeunits']
        self.spatialunit = metadata['spatialunits']
        self.timeinterval = metadata['timeinterval']

        return metadata


    def _calculate_time_interval(self, df: pl.DataFrame) -> Optional[float]:
        t = (
            df[self.t_col]
            .cast(pl.Float64, strict=False)
            .drop_nulls()
            .unique()
            .sort()
            .to_numpy()
        )
        timeintervals = np.diff(t)

        if timeintervals.size == 0:
            return float('nan')
        else:
            positive = timeintervals[timeintervals > 0]
            base = float(positive.min()) if positive.size else float(timeintervals.min())

            if base > 0:
                ratios = timeintervals / base
                is_regular = np.all(np.isclose(ratios, np.round(ratios), atol=1e-6))
            else:
                is_regular = False

            if is_regular:
                return base
            else:
                result = float(np.median(timeintervals))
                warn((f"Non-uniformly spaced time point data -> will probably lead to incorrect data computation.\n"
                               f"Observed time steps:\n{timeintervals}\nUsing: {result}"), InputWarning, 2)
                return result


    def get_columns(self, path: str) -> List[str]:
        """
        Returns a list of column names from the DataFrame.
        """
        df, _ = self._read_file(path)
        return df.columns
    

    def match_columns(self, columns: List[str], lookfor: List[str]) -> str:
        """
        Looks for matches with any of the provided strings.
        - First tries exact matches.
        - Then checks if the column starts with any of given terms.
        - Finally checks if any term is a substring of the column name.
        If no match is found, returns None.
        """

        normalized_columns = [
            (col, str(col).replace('', ' ').strip().lower() if col is not None else '') for col in columns
        ]
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col == look.lower():
                    return col
                
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col.startswith(look.lower()):
                    return col
                
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if look.lower() in norm_col:
                    return col
        return None
    

    def _guard_replicates(self, rep_lbl, file_info, data: pl.DataFrame, _rep_guard_list) -> pl.DataFrame:

        if rep_lbl in _rep_guard_list:

            count = _rep_guard_list.count(rep_lbl)
            data = data.with_columns(
                (pl.lit(f"{count}_") + pl.col('track_id').cast(pl.Utf8)).alias('track_id')
            )

            warn(message=f"Multiple ({count+1}) replicate labels: '{rep_lbl}' \n-> adding prefix: {count} to replicate label: {rep_lbl}\nFile info: {file_info}",
                          category=LabelWarning,
                          stacklevel=2)
            
            self.rep_multiplicates = True
        
        return data
    
    
    def _py_numeric_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Attempt to cast every column to Float64, keeping the original
        column when the cast would lose data (non-numeric columns)."""
        exprs = []
        for col in df.columns:
            casted = df[col].cast(pl.Float64, strict=False)
            # only convert if no new nulls were introduced
            if casted.null_count() == df[col].null_count():
                exprs.append(casted.alias(col))
        if exprs:
            df = df.with_columns(exprs)
        return df


    def _clean_name(self, name: str) -> str:
        name = str(name)
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        name = name.strip().lower()
        return name
    

    def _standname(self, df: pl.DataFrame, id_col: str, t_col: str, x_col: str, y_col: str) -> pl.DataFrame:
        return df.rename({id_col: 'track_id', t_col: 'time_point', x_col: 'x_coordinate', y_col: 'y_coordinate'})


    def _convert_spatial(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.spatialunit in (None, ""):
            warn('No spatial units found in input files.\n Please specify the spatial units using `spatial_unit="<unit>"`.', InputWarning, stacklevel=2)
            return df
        
        temp_args = get_aliases(
            {'from': self.spatialunit, 'to': self.kwargs.get('convert_spatial_to')},
            self.metadata.UNIT_ALIASES
        )
        from_unit = temp_args['from']
        to_unit = temp_args['to']

        if from_unit is None or to_unit is None:
            warn(
                message=f"Unsupported spatial conversion: {self.spatialunit} -> {self.kwargs.get('convert_spatial_to')}. Skipping conversion.",
                category=LabelWarning,
                stacklevel=2
            )
            return df

        from_unit = self.metadata._get_alias(from_unit)
        to_unit = self.metadata._get_alias(to_unit)

        if from_unit == to_unit:
            return df

        factor = calc.UNIT_TO_MICRONS[from_unit] / calc.UNIT_TO_MICRONS[to_unit]

        df = df.with_columns([
            (pl.col("x_coordinate") * factor).alias("x_coordinate"),
            (pl.col("y_coordinate") * factor).alias("y_coordinate"),
        ])

        self.metadata['spatialunits'] = to_unit

        return df
    

    def _convert_time(self, df: pl.DataFrame) -> pl.DataFrame:

        if self.timeunit in (None, ""):
            warn('No time units found in input files.\n Please specify the time units using `time_unit="<unit>"`.', InputWarning, stacklevel=2)
            return df
        
        temp_args = get_aliases(
            {'from': self.timeunit, 'to': self.kwargs.get('convert_time_to')},
            self.metadata.UNIT_ALIASES
        )
        from_unit = temp_args['from']
        to_unit = temp_args['to']

        if from_unit is None or to_unit is None:
            warn(
                message=f"Unsupported time conversion: {self.timeunit} -> {self.kwargs.get('convert_time_to')}. Skipping conversion.",
                category=LabelWarning,
                stacklevel=2
            )
            return df

        from_unit = self.metadata._get_alias(from_unit)
        to_unit = self.metadata._get_alias(to_unit)

        if from_unit == to_unit:
            return df

        factor = calc.UNIT_TO_SECONDS[from_unit] / calc.UNIT_TO_SECONDS[to_unit]

        df = df.with_columns((pl.col("time_point") * factor).alias("time_point"))

        self.metadata['timeinterval'] = self.timeinterval * factor
        self.metadata['timeunits'] = to_unit

        return df
    

    def _filter_metadata(self, metadata: Dict[str, dict]) -> Dict[str, dict]:
        """
        Keep only the essential column units (id/t/x/y) plus any `retain`
        columns, alongside the special summary keys.
        """
        keep_cols = [self.id_col, self.t_col, self.x_col, self.y_col]
        if self.retain:
            keep_cols += list(self.retain)

        special_keys = ['spatialunits', 'timeunits', 'timeinterval', 'nframes', 'columns']

        filtered = {}
        for file_name, meta in metadata.items():
            new_meta = {col: meta[col] for col in keep_cols if col in meta}
            for k in special_keys:
                if k in meta:
                    new_meta[k] = meta[k]
            filtered[file_name] = new_meta
        return filtered


load_data = DataLoader().load_data
get_columns = DataLoader().get_columns
match_columns = DataLoader().match_columns