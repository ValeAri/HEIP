import argparse
import sys
sys.path.insert(0,'../src')
from merging import CellMerger
from pathlib import Path
from typing import Dict, Any

def mergingTiles(args: Dict[str, Any]) -> None:
    """Merge instant segmentation tiles result in one .json document."""
    fn = Path(args.result_dir) / args.fname_cells

    c = CellMerger(in_dir=args.in_dir)
    c.merge(fname=fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="The path of the input folder with the tiles.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="The output path where the megerd json file will be saved.",
    )
    parser.add_argument(
        "--fname_cells",
        type=str,
        required=True,
        help="The merge file name.",
    )

    args = parser.parse_args()
    mergingTiles(args)
