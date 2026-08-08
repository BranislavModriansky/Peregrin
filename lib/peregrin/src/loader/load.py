import re
from sys import path
import traceback
import warnings

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os.path as op
from typing import Dict, List, Optional, Tuple
from os import PathLike


from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *

from ..compute.stats import stats
from ..various import get_aliases


class DataLoader:

    ALIASES = {
        
    }

    CATEGORIES = {
        'set': 5,
        'subset': 4,
        'group': 3,
        'subgroup': 2,
        'subsubgroup': 1
    }

    def __init__(self): ...


    def load_data(
        self, 
        files: PathLike[str] | list | dict = None,
        colnames: dict = {
            'id': 'TRACK_ID', 
            't':  'POSITION_T', 
            'x':  'POSITION_X', 
            'y':  'POSITION_Y'
        }, 
        *,
        retain: Optional[list[str]] = None,
        **kwargs
    ) -> pd.DataFrame:

        """
        Load tracking data from any number of files into a single DataFrame, 
        while assigning condition and replicate labels. This method is used
        to prepare data for further handling in the computations, using the 
        `peregrin` library.
        
        Parameters
        ----------
        files : list[PathLike[str]] | dict
            Either a list of file paths or a dictionary with keys as category indicies and values either as dicts (subcategories) or lists of file paths.
            Data can be categorized up to 5 levels deep having the following structure:

        ```
        { set: { subset: { group: { subgroup: { subsubgroup: file }}}}}

        Input example 1 (3-level categorization using lists of file paths):
        [ [ ['path/to/file01.csv', 'path/to/file02.csv'], 
            ['path/to/file03.csv', 'path/to/file04.csv']], 
          [ ['path/to/file05.csv', 'path/to/file06.csv'], 
            ['path/to/file07.csv', 'path/to/file08.csv']]]

        Input example 2 (3-level categorization using dictionaries):
        {'condition1': {'bio_replicate1': ['path/to/tech_replicate_A-10-10-10.csv', 'path/to/tech_replicate_B-10-10-10.csv'],
                        'bio_replicate2': ['path/to/tech_replicate_A-11-10-10.csv', 'path/to/tech_replicate_B-11-10-10.csv']},
         'condition2': {'bio_replicate1': ['path/to/tech_replicate_A-18-10-10.csv', 'path/to/tech_replicate_B-18-10-10.csv'],
                        'bio_replicate2': ['path/to/tech_replicate_A-20-10-10.csv', 'path/to/tech_replicate_B-20-10-10.csv']}}

        Input example 3 (4-level categorization using dictionaries):
        {'A': {'A1': {'A1a': {'A1an': ['path/to/file01.csv', 'path/to/file02.csv'],
                              'A1am': ['path/to/file03.csv', 'path/to/file04.csv']},
                      'A1b': {'A1bn': ['path/to/file05.csv', 'path/to/file06.csv'],
                              'A1bm': ['path/to/file07.csv', 'path/to/file08.csv']}},
               'A2': {'A2a': {'A2an': ['path/to/file09.csv', 'path/to/file10.csv'],
                              'A2am': ['path/to/file11.csv', 'path/to/file12.csv']},
                      'A2b': {'A2bn': ['path/to/file13.csv', 'path/to/file14.csv'],
                              'A2bm': ['path/to/file15.csv', 'path/to/file16.csv']}}},
         'B': {'B1': {'B1a': {'B1an': ['path/to/file17.csv', 'path/to/file18.csv'],
                              'B1am': ['path/to/file19.csv', 'path/to/file20.csv']},
                      'B1b': {'B1bn': ['path/to/file21.csv', 'path/to/file22.csv'],
                              'B1bm': ['path/to/file23.csv', 'path/to/file24.csv']}},
               'B2': {'B2a': {'B2an': ['path/to/file25.csv', 'path/to/file26.csv'],
                              'B2am': ['path/to/file27.csv', 'path/to/file28.csv']},
                      'B2b': {'B2bn': ['path/to/file29.csv', 'path/to/file30.csv'],
                              'B2bm': ['path/to/file31.csv', 'path/to/file32.csv']}}}}
        
        ```
         See more detailed documentation of this method here: https://branislavmodriansky.github.io/peregrin/lib/overview.html

        colnames : dict
            A dictionary specifying the column names for track identifiers, time points, and x/y coordinates. 
            With the default {'id': None, 't': None, 'x': None, 'y': None}, the method will NOT attempt to 
            automatically detect these columns.

        t_unit : str, optional, default 's'
            The unit of time used in the input data. Supported units include: 'ms', 's', 'min', 'h', 'day'.
        
        retain : list[str], optional, default None
            A list of column names from the original input data to be retained. If None, only the essential columns 
            (track_id, time_point, x_coordinate, y_coordinate) are extracted, while all other columns are discarded.

        convert_time_to : str, optional, default None
            If provided, time will be converted from the input unit `t_unit` to the specified target unit. 
            Supported units include: 'ms', 's', 'min', 'h', 'day'.

        mirror_y : bool, optional, default False
            If True, the method will mirror the y-coordinates across their midpoint - useful for correcting mirrored y-coordinates in exported data.
        
        mirror_x : bool, optional, default False
            If True, the method will mirror the x-coordinates across their midpoint - useful for correcting mirrored x-coordinates in exported data.

        merge : Literal['all', 'sets', 'subsets', 'groups', 'subgroups', None], default 'all'
            Specifies how to merge the loaded data:

        split_filename : bool, optional, default False
            If True, the method will split file names during subsubgroup labeling using the `split_filename_position` and `split_filename_delimiter` parameters.

        split_filename_position : Literal['first', 'last'] | int, optional, default 'last'
            Determines how to split file names during subsubgroup labeling when `split_filename` is True.
            Either one of 'first' or 'last' to split at the first or last occurrence of the `split_character`, or an integer to split at a specific index.

        split_filename_delimiter : str, optional, default '-'
            The character used to split file names for automatic label extraction when `split_filename` is True.
        ```
        'all' - merge all data into a single DataFrame (default)
        'sets' - merge data within each set, but keep sets separate
        'subsets' - merge data within each subset, but keep subsets separate
        'groups' - merge data within each group, but keep groups separate
        'subgroups' - merge data within each subgroup, but keep subgroups separate
        None - do not merge; return data as a dictionary of dictionaries with DataFrames
        ```

        Returns
        -------
        pd.DataFrame
            A single DataFrame containing all loaded data, with condition and replicate labels assigned to each row.
        
        """

        self.retain = retain
        self.kwargs = kwargs

        self.id_col = colnames['id']
        self.t_col  = colnames['t']
        self.x_col  = colnames['x']
        self.y_col  = colnames['y']

        # Time / transform options (wired as kwargs)
        self.convert_time_to = kwargs.get('convert_time_to', None)
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
            stats.input_metadata.update(metadata)
            df = self._extract(df)

            for cat, label in zip(used_categories, labels):
                df[cat] = label

            records.append(df)

        stats.input_metadata.check()

        return self._merge(records, used_categories)



    def merge_leaf_lists(tree, depth):
        current = [tree]

        # Go to the parents of the leaf lists
        for _ in range(depth - 2):
            current = [item for lst in current for item in lst]

        # Merge each leaf list
        for i, leaf in enumerate(current):
            current[i] = [pd.concat(leaf, ignore_index=True)]

        return tree  

    

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
            return pd.DataFrame()

        if self.merge == 'all':
            return pd.concat(records, ignore_index=True)

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

        # How many of the available (bottom-up) categories to key on.
        full_order = ['set', 'subset', 'group', 'subgroup', 'subsubgroup']
        # index of the deepest keying category within full_order
        
        # only keep categories that actually exist in the data
        key_cats = [c for c in full_order[:target] if c in used_categories]

        result = {}
        for df in records:
            keys = [str(df[c].iloc[0]) for c in key_cats]

            node = result
            for k in keys[:-1]:
                node = node.setdefault(k, {})

            last = keys[-1]
            if self.merge is None:
                node[last] = df
            else:
                node.setdefault(last, []).append(df)

        # Concatenate leaf lists (unless merge is None)
        if self.merge is not None:
            self._concat_leaves(result)

        return result

    def _concat_leaves(self, node):
        """Recursively concatenate lists of DataFrames stored in a nested dict."""
        for key, val in node.items():
            if isinstance(val, list):
                node[key] = pd.concat(val, ignore_index=True)
            elif isinstance(val, dict):
                self._concat_leaves(val)


    def _iter_list_tree(self, tree: list, depth: int, labels: tuple = ()):
        """
        Yield (labels, filepath) leaves from a nested list tree.
        Non-file levels are labelled with their 1-based index (as str);
        the file level is labelled with the file's basename.
        """
        if depth == 1:
            # This list holds file paths
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
            # This dict maps filename -> filepath, OR is a list of paths
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

        With `split_filename=False` (default) the full basename is used.
        With `split_filename=True` the stem is split on `split_character`
        according to `split_how`:
            'first' -> keep the part before the first occurrence
            'last'  -> keep the part before the last occurrence
            int i   -> keep the i-th part (0-based) of the split
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
    

    def _load_from_list(self, file_tree_list: list, depth: int) -> pd.DataFrame:

        current = [file_tree_list]

        for _ in range(depth - 1):
            current = [item for lst in current for item in lst]

        return [f for group in current for f in group]
    
    def _load_from_dict(self, file_tree_dict: dict, depth: int) -> pd.DataFrame:
        current = [file_tree_dict]

        for _ in range(depth - 1):
            current = [item for d in current for item in d.values()]

        return [f for group in current for f in group]


    def _load_metadata(self, metadata: Dict[str, any]) -> None:
            """
            Load metadata from a dictionary into the global object.
            """
            ...
            

    def _extract(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        if not all(col in df.columns for col in [self.id_col, self.t_col, self.x_col, self.y_col]):
            missing = [col for col in [self.id_col, self.t_col, self.x_col, self.y_col] if col not in df.columns]
            raise ColumnsNotFoundError(f"missing columns: {missing}")

        if self.retain is None:
            df = df[[self.id_col, self.t_col, self.x_col, self.y_col]].apply(pd.to_numeric, errors='coerce')
        else:
            df = df[[self.id_col, self.t_col, self.x_col, self.y_col] + self.retain].copy()
            for c in [self.id_col, self.t_col, self.x_col, self.y_col]:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            self._py_numeric_df(df)

        df = df.dropna(subset=[self.id_col, self.t_col, self.x_col, self.y_col]).reset_index(drop=True)
        
        if self.mirror_y:
            y_mid = (df[self.y_col].min() + df[self.y_col].max()) / 2
            df[self.y_col] = 2 * y_mid - df[self.y_col]
        if self.mirror_x:
            x_mid = (df[self.x_col].min() + df[self.x_col].max()) / 2
            df[self.x_col] = 2 * x_mid - df[self.x_col]
            
        df = self._standname(df, self.id_col, self.t_col, self.x_col, self.y_col)

        # run time conversion for any requested target unit handled by _convert_time
        if self.convert_time_to not in (None, ""):
            df = self._convert_time(df)
         
        return df


    def _read_file(self, filepath: str) -> pd.DataFrame:
            
        _, ext = op.splitext(filepath.lower())

        match ext:
            case '.xml':
                return self._read_trackmate_xml(filepath)
            case '.csv' | '.xls' | '.xlsx':
                return self._read_table(filepath, ext)
            case _:
                raise FileFormatError(f"{ext} is not supported. Supported formats include: .csv, .xls, .xlsx, .xml")


    def _read_trackmate_xml(self, filepath):
        """Parse a TrackMate project XML and return spots and a merged metadata dict."""
        root = ET.parse(filepath).getroot()
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
        for category in model.find('FeatureDeclarations'):
            for feat in category:
                metadata[feat.attrib['feature']] = dim_to_unit.get(feat.attrib.get('dimension'))

        # spots
        df = pd.DataFrame([
            spot.attrib
            for frame in model.find('AllSpots')
            for spot in frame
        ])
        df['ID'] = df['ID'].astype(np.int64)
        num_cols = [c for c in df.columns if c != 'name']
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

        # tag each spot with its TRACK_ID via the tracks (single pass, vectorized map)
        src_ids, trk_ids = [], []
        for track in model.find('AllTracks'):
            tid = int(track.attrib['TRACK_ID'])
            for edge in track:
                a = edge.attrib
                src_ids += [a['SPOT_SOURCE_ID'], a['SPOT_TARGET_ID']]
                trk_ids += [tid, tid]

        mapping = pd.Series(
            np.asarray(trk_ids, dtype=np.int64),
            index=np.asarray(src_ids, dtype=np.int64),
        )
        mapping = mapping[~mapping.index.duplicated()]
        df['TRACK_ID'] = mapping.reindex(df['ID'].values).values

        # move TRACK_ID to the first column
        df.insert(0, 'TRACK_ID', df.pop('TRACK_ID'))


        # merge image calibration into the same metadata dict
        metadata.update(root.find('Settings/ImageData').attrib)
        metadata['spatialunits'] = spatialunits
        metadata['timeunits'] = timeunits

        self.timeunit = timeunits
        self.timeinterval = metadata['timeinterval']

        metadata = {str(filepath.split(op.sep)[-1]): metadata}

        return df, metadata
            

    def _read_table(self, filepath, ext, *, metadata_row_index=2, skiprows=4, encodings=("utf-8", "cp1252", "latin1", "iso8859_15"), **kwargs) -> Tuple[pd.DataFrame, dict]:
        try:
            if ext in ('.xls', '.xlsx'):
                column_names, units_row, df = self._read_excel_parts(
                    filepath, metadata_row_index, skiprows
                )
                metadata = {str(filepath.split(op.sep)[-1]): self._build_metadata(df, column_names, units_row)}
                return df, metadata

            # CSV path: try multiple encodings
            for enc in encodings:
                try:
                    column_names = pd.read_csv(filepath, nrows=0, encoding=enc).columns.tolist()

                    metadata_row = None
                    if metadata_row_index is not None:
                        metadata_row = pd.read_csv(
                            filepath, skiprows=metadata_row_index, nrows=1, encoding=enc
                        ).iloc[0].tolist()

                    df = pd.read_csv(
                        filepath, names=column_names, skiprows=skiprows,
                        encoding=enc, low_memory=False
                    )

                    metadata = {str(filepath.split(op.sep)[-1]): self._build_metadata(df, column_names, metadata_row)}

                    return df, metadata

                except UnicodeDecodeError:
                    continue

            raise FileFormatError(
                f"Failed to decode CSV file: {filepath}. Tried encodings: {encodings}."
            )

        except Exception as e:
            raise FileFormatError(f"{str(e)} -> Failed to read file: {filepath}.")


    def _read_excel_parts(self, filepath, metadata_row_index, skiprows) -> Tuple[List[str], Optional[List[str]], pd.DataFrame]:
        """Read header, units row, and data body from an Excel file."""
        column_names = pd.read_excel(filepath, nrows=0).columns.tolist()

        metadata_row = None
        if metadata_row_index is not None:
            metadata_row = pd.read_excel(
                filepath, skiprows=metadata_row_index, nrows=1, header=None
            ).iloc[0].tolist()

        df = pd.read_excel(filepath, names=column_names, skiprows=skiprows, header=None)
        return column_names, metadata_row, df


    def _build_metadata(self, df, column_names, metadata_row) -> Dict[str, Optional[str]]:
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

        metadata['spatialunits'] = metadata.get(self.x_col, '')
        metadata['timeunits'] = metadata.get(self.t_col, '')
        metadata['timeinterval'] = self._calculate_time_interval(df)
        metadata['nframes'] = df[self.t_col].nunique()

        self.timeunit = metadata['timeunits']
        self.timeinterval = metadata['timeinterval']

        return metadata


    def _calculate_time_interval(self, df: pd.DataFrame) -> Optional[float]:
        df.sort_values(self.t_col, inplace=True)
        timeintervals = np.diff(df[self.t_col].unique())

        if timeintervals.size == 0:
            return float('nan')
        else:
            # Base step = smallest positive interval (one frame).
            positive = timeintervals[timeintervals > 0]
            base = float(positive.min()) if positive.size else float(timeintervals.min())

            if base > 0:
                ratios = timeintervals / base
                # Uniform if every gap is an (near-)integer multiple of the base step
                # -> tolerates dropped frames (e.g. a single 120 among 60s).
                is_regular = np.all(np.isclose(ratios, np.round(ratios), atol=1e-6))
            else:
                is_regular = False

            if is_regular:
                return base
            else:
                result = float(np.median(timeintervals))
                warnings.warn((f"Non-uniformly spaced time point data -> will probably lead to incorrect data computation.\n"
                               f"Observed time steps:\n{timeintervals}\nUsing: {result}"), InputWarning, 2)
                return result


    def get_columns(self, path: str) -> List[str]:
        """
        Returns a list of column names from the DataFrame.
        """
        df, _ = self._read_file(path)  # or pd.read_excel(path), depending on file type
        return df.columns.tolist()
    

    def match_columns(self, columns: List[str], lookfor: List[str]) -> str:
        """
        Looks for matches with any of the provided strings.
        - First tries exact matches.
        - Then checks if the column starts with any of given terms.
        - Finally checks if any term is a substring of the column name.
        If no match is found, returns None.
        """

        # Normalize columns for matching
        normalized_columns = [
            (col, str(col).replace('', ' ').strip().lower() if col is not None else '') for col in columns
        ]
        # Try exact matches first
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col == look.lower():
                    return col
                
        # Then try startswith
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col.startswith(look.lower()):
                    return col
                
        # Then try substring
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if look.lower() in norm_col:
                    return col
        return None
    

    def _guard_replicates(self, rep_lbl, file_info, data, _rep_guard_list) -> pd.DataFrame:

        if rep_lbl in _rep_guard_list:

            count = _rep_guard_list.count(rep_lbl)
            data['track_id'] = data['track_id'].apply(lambda x: f"{count}_{x}")

            warnings.warn(message=f"Multiple ({count+1}) replicate labels: '{rep_lbl}' \n-> adding prefix: {count} to replicate label: {rep_lbl}\nFile info: {file_info}",
                          category=LabelWarning,
                          stacklevel=2)
            
            self.rep_multiplicates = True
        
        return data
    
    
    def _py_numeric_df(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            try: 
                df[col] = pd.to_numeric(df[col], errors='raise')
            except Exception as e: 
                continue  # quietly move on if conversion fails


    def _clean_name(self, name: str) -> str:
        name = str(name)
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        name = name.strip().lower()
        return name
    

    def _standname(self, df, id_col: str, t_col: str, x_col: str, y_col: str) -> pd.DataFrame:
        return df.rename(columns={id_col: 'track_id', t_col: 'time_point', x_col: 'x_coordinate', y_col: 'y_coordinate'})
    

    def _convert_time(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_to_seconds = {
            "ms": 1e-3,
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
            "day": 86400.0,
        }

        temp_args = get_aliases(
            {'from': self.timeunit, 'to': self.convert_time_to},
            stats.input_metadata.UNIT_ALIASES
        )
        from_unit = temp_args['from']
        to_unit = temp_args['to']

        if from_unit is None or to_unit is None:
            warnings.warn(
                message=f"Unsupported time conversion: {self.timeunit} -> {self.convert_time_to}. Skipping conversion.",
                category=LabelWarning,
                stacklevel=2
            )
            return df

        if from_unit == to_unit:
            return df

        factor = unit_to_seconds[from_unit] / unit_to_seconds[to_unit]

        df["time_point"] = df["time_point"] * factor

        stats.input_metadata['timeinterval'] = self.timeinterval * factor
        stats.input_metadata['timeunits'] = to_unit

        return df


load_data = DataLoader().load_data
get_columns = DataLoader().get_columns
match_columns = DataLoader().match_columns