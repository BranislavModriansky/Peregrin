from pathlib import Path
import pandas as pd
import numpy as np
import itertools


def craft_dummy_dataframe(
    n_tracks: int = 6,
    n_time_points: int = 100,
    seed: int | None = 20,
    categories: dict[str, list[str]] | None = None,
    save: bool | str = False,          # False | 'single' | 'split' | 'tree'
    out_dir: str | Path = ".",
    filename: str = "dummy_cell_tracks",
    n_files: int = 1,                  # files per leaf ('tree'/'split' modes)
) -> tuple[pd.DataFrame, dict | str | None]:
    """
    Generate synthetic persistent-random-walk cell trajectories, organised
    into an arbitrary category hierarchy.

    Parameters
    ----------
    categories : dict, optional
        Ordered mapping of category level -> labels, e.g.::

            {
                'set':      ['set_A', 'set_B'],
                'subset':   ['subset_1', 'subset_2'],
                'group':    ['ctrl', 'treated'],
                'subgroup': ['rep1', 'rep2'],
            }

        Any subset of levels works (1-4). Defaults to 2 sets x 2 subsets.
        `n_tracks` trajectories are generated per leaf combination.

    save :
        - False    -> no files written
        - 'single' -> one CSV with all category columns
        - 'split'  -> one CSV per category combination (flat directory),
                      named e.g. dummy_cell_tracks__set_A__subset_1.csv
        - 'tree'   -> directory tree out_dir/set_A/subset_1/.../<filename>.csv

    Returns
    -------
    (df, paths)
        `df` is the combined DataFrame. `paths` is:
        - 'single' -> the file path (str)
        - 'split'/'tree' -> a nested dict of file paths mirroring the
          category hierarchy, directly consumable by `load_data(paths)`
        - False -> None
    """
    if categories is None:
        categories = {
            'set': ['set_A', 'set_B'],
            'subset': ['subset_1', 'subset_2'],
        }

    allowed = ['set', 'subset', 'group', 'subgroup']
    levels = [lvl for lvl in allowed if lvl in categories]
    if not levels:
        raise ValueError(f"categories must use keys from {allowed}")

    rng = np.random.default_rng(seed)
    records = []
    global_track_id = 0

    def _simulate_track(track_id: int) -> list[tuple]:
        # start = rng.integers(0, n_time_points // 4)
        start = 0
        # length = rng.integers(n_time_points // 2, n_time_points + 1)
        length = n_time_points + 1
        end = min(start + length, n_time_points)

        x, y = rng.uniform(0, 500), rng.uniform(0, 500)
        speed = rng.uniform(2.0, 8.0)
        persistence = rng.uniform(0.6, 0.98)
        angle = rng.uniform(0, 2 * np.pi)

        rows = []
        for t in range(start, end):
            rows.append((track_id, t, x, y))
            angle += (1 - persistence) * rng.uniform(-np.pi, np.pi)
            step = rng.normal(speed, speed * 0.2)
            x += step * np.cos(angle) + rng.normal(0, 0.5)
            y += step * np.sin(angle) + rng.normal(0, 0.5)
        return rows

    # Cartesian product over all category levels -> one leaf per combination
    combos = list(itertools.product(*(categories[lvl] for lvl in levels)))
    for combo in combos:
        for _ in range(n_tracks):
            global_track_id += 1
            for track_id, t, x, y in _simulate_track(global_track_id):
                records.append(combo + (track_id, t, x, y))

    df = pd.DataFrame(
        records,
        columns=levels + ['track_id', 'time_point', 'x_coordinate', 'y_coordinate'],
    )
    df = df.sort_values(levels + ['track_id', 'time_point']).reset_index(drop=True)
    df['track_uid'] = df.groupby(levels + ['track_id']).ngroup()

    def clear_directory(path: Path) -> None:
        """Delete all files and subdirectories in the given directory."""
        if path.exists() and path.is_dir():
            for item in path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    clear_directory(item)
                    item.rmdir()

    def _split_tracks(sub: pd.DataFrame) -> list[pd.DataFrame]:
        """Split a leaf's tracks into up to n_files roughly equal chunks."""
        ids = sub['track_id'].unique()
        chunks = np.array_split(ids, min(n_files, len(ids)))
        return [sub[sub['track_id'].isin(chunk)] for chunk in chunks if len(chunk)]

    # ------------------------------------------------------------------ save
    paths = None
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if save in (True, 'single'):
        single_dir = out_dir / 'single'
        clear_directory(single_dir)
        single_dir.mkdir(parents=True, exist_ok=True)
        path = single_dir / f"{filename}.csv"
        df.to_csv(path, index=False)
        paths = str(path)

    elif save == 'split':
        paths = {}
        for combo, sub in df.groupby(levels, sort=False):
            combo = combo if isinstance(combo, tuple) else (combo,)
            stem = filename + "".join(f"__{c}" for c in combo)

            split_dir = out_dir / 'split'
            clear_directory(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for j, part in enumerate(_split_tracks(sub)):
                path = split_dir / f"{stem}_{j}.csv"
                part.drop(columns=levels).to_csv(path, index=False)
                files.append(str(path))
            node = paths
            for c in combo[:-1]:
                node = node.setdefault(c, {})
            node[combo[-1]] = files if n_files > 1 else files[0]

    elif save == 'tree':
        paths = {}
        tree_dir = out_dir / 'tree'
        clear_directory(tree_dir)

        for combo, sub in df.groupby(levels, sort=False):
            combo = combo if isinstance(combo, tuple) else (combo,)
            leaf_dir = tree_dir.joinpath(*combo)
            leaf_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for j, part in enumerate(_split_tracks(sub)):
                path = leaf_dir / f"{filename}_{j}.csv"
                part.drop(columns=levels).to_csv(path, index=False)
                files.append(str(path))
            node = paths
            for c in combo[:-1]:
                node = node.setdefault(c, {})
            node[combo[-1]] = files if n_files > 1 else files[0]

    elif save is not False:
        raise ValueError(f"Invalid save mode: {save!r}. Use False, 'single', 'split', or 'tree'.")

    return df, paths