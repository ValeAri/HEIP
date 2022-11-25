import os
import argparse
from functools import partial
from pathlib import Path
from typing import List, Union
from tqdm.contrib.concurrent import process_map

from src.merging import CellMerger


def merge_tiles(
    in_dir: Union[str, Path], overwrite: bool = False, use_pbar: bool = True
) -> None:
    """Merges a set of geojson cell annotation .json files.

    NOTE: Assumes that the input directory contains a `cells` directory that contains
          all the .json annotation files. This saves the output file to the `in_dir`.
          The the name of the output file will be `{in_dir}_cells.json`. If in need of
          overwriting existing output json files set `overwrite=True`.

    Parameters
    ----------
        in_dir : Path or str
            Path to the directory containing ".json" geojson annotation files.
        overwrite: bool, default=False
            If True, will overwrite the `{in_dir}_cells.json` file if it exists in the
            input folder.
        use_pbar : bool, default=True
            Flag, whether to use progress bar.
    """
    try:
        in_dir = Path(in_dir)
        fname = in_dir.name
        in_dir = in_dir / "cells"
        res_dir = in_dir.parent
        fn = (res_dir / f"{fname}_cells").with_suffix(".json")

        if not fn.exists() or overwrite:
            c = CellMerger(in_dir=in_dir)
            c.merge(fname=fn, use_pbar=use_pbar)
        else:
            print(
                f"Found an existing {fn.name} in {res_dir}. Skipping the merging. "
                "If in need of rewriting the files, set `overwrite=True` \n"
            )

    except KeyboardInterrupt:
        print("Detected keyboard interrupt.")
    except Exception as e:
        print(
            "Could not merge .json files in {} due to error:\n {}".format(
                in_dir.name, e
            )
        )
        return


def merge_gson_folders(
    in_folders: List[Path],
    num_workers: int = -1,
    overwrite: bool = False,
    use_pbar: bool = True,
) -> None:
    """Merge a list of gson folders in parallel.

    Parameters
    ----------
        in_folders : List[Tuple[Path, str]]
            A list of Path, str tuples, containing the dir paths and the dir name that
            will be used as the output filename.
        num_workers : int, default=-1
            Number of worker processes. Defaults to -1 which uses all available threads.
        use_pbar : bool, default=True
            Flag, whether to use progress bar to update after every iteration.
            If true, the progress jumps between the processes making the progress
            possibly hard to follow. If False, the progress bar updates only when a
            process is finished.
    """
    func = partial(merge_tiles, use_pbar=use_pbar, overwrite=overwrite)

    if num_workers <= 0:
        num_workers = os.cpu_count()
        if len(in_folders) < num_workers:
            num_workers = len(in_folders)

    print(f"Merging {len(in_folders)} geojson input folders.")
    process_map(func, in_folders, max_workers=num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="The directory containing the folders that contain the gson files",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="Flag wheter to overwrite existing output .json files in the folders.",
    )
    parser.add_argument(
        "--use_pbar",
        type=int,
        default=1,
        help="Flag wheter to update pbar after every iteration.",
    )

    args = parser.parse_args()
    res_dir = Path(args.in_dir)
    in_dirs = sorted(d for d in res_dir.iterdir() if d.is_dir())

    merge_gson_folders(
        in_dirs, use_pbar=bool(args.use_pbar), overwrite=bool(args.overwrite)
    )
