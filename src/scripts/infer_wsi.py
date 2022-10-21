import torch
import argparse
from typing import Dict, Any
from pathlib import Path

from cellseg_models_pytorch.inference import SlidingWindowInferer
from ..unet import get_seg_model, convert_state_dict, MODEL_PARTS


def run_infer_wsi_patches(args: Dict[str, Any]) -> None:
    """Run inference with HEIP model.

    Runs inference to multiple wsi patches by looping through a folder
    with the following structure:

    (The coordinates are optional but recommended since they are needed for merging.)

    wsi_patches/
    │
    ├──sample1_patches/
    │    ├── patch_x-0_y-0.png
    │    ├── patch_x-0_y-1000.png
    │    ├── patch_x-0_y-2000.png
    │    ├── patch_x-1000_y-0.png
    │    ├── patch_x-1000_y-1000.png
    │    .
    │    .
    │    .
    │    └── patch_x-n_y-n.png
    |
    ├──sample2_patches/
    │    ├── patch_x-0_y-0.png
    │    ├── patch_x-0_y-1000.png
    │    ├── patch_x-0_y-2000.png
    │    ├── patch_x-1000_y-0.png
    │    ├── patch_x-1000_y-1000.png
    │    .
    │    .
    │    .
    │    └── patch_x-n_y-n.png
    │
    .
    .
    .
    └──

    NOTE: Resulting masks are save as .json files that can be read with QuPath.
    The result folder has the following structure:

    result_dir/
    │
    ├──sample1_patches/
    |    |
    |    └── cells/
    |         ├── patch_x-0_y-0.json
    |         ├── patch_x-0_y-1000.json
    |         ├── patch_x-1000_y-0.json
    |         ├── patch_x-1000_y-1000.json
    |         .
    |         .
    |         .
    |         └── patch_x-n_y-n.json
    |
    ├──sample2_patches/
    |    |
    |    └── cells/
    |         ├── patch_x-0_y-0.json
    |         ├── patch_x-0_y-1000.json
    |         ├── patch_x-1000_y-0.json
    |         ├── patch_x-1000_y-1000.json
    |         .
    |         .
    |         .
    |         └── patch_x-n_y-n.json
    |
    .
    .
    .
    └──

    """
    ckpt = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)

    model = get_seg_model()
    new_state_dict = convert_state_dict(
        MODEL_PARTS, model.state_dict(), ckpt["state_dict"]
    )
    model.load_state_dict(new_state_dict, strict=True)

    # set classes
    cell_types = None
    if args.classes_type is not None:
        cell_types = {c: i for i, c in enumerate(args.classes_type.split(","))}

    tissue_types = None
    if args.classes_tissue is not None:
        tissue_types = {c: i for i, c in enumerate(args.classes_tissue.split(","))}

    res_dir = Path(args.res_dir)
    for d in Path(args.in_dir).iterdir():
        print(f"inference: {d}")

        inferer = SlidingWindowInferer(
            model,
            d,
            out_activations={"type": "softmax", "inst": "softmax", "omnipose": None},
            out_boundary_weights={"inst": False, "type": False, "omnipose": True},
            patch_size=(args.patch_size, args.patch_size),
            stride=args.stride,
            padding=args.padding,
            instance_postproc="omnipose",
            batch_size=args.batch_size,
            save_dir=save_dir,
            save_format=".json",
            geo_format=args.geo_format,
            offsets=bool(args.offsets),
            classes_type=cell_types,
            classes_sem=tissue_types,
            device=args.device,
            n_devices=args.n_devices,
        )
        inferer.infer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        metavar="INDIR",
        help="The path to the folder containing the wsi patches.",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        required=True,
        metavar="RESDIR",
        help="The path to the folder where results will be saved.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        metavar="RESDIR",
        help="The path to the checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        metavar="DEVICE",
        help="One of cpu, cuda",
    )
    parser.add_argument(
        "--n_devices",
        type=int,
        required=True,
        metavar="N_DEVICES",
        help="Number of GPUS/CPUS when running model forward.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        metavar="EXPNAME",
        help="The experiment name.",
    )
    parser.add_argument(
        "--exp_version",
        type=str,
        default=None,
        metavar="EXPVER",
        help="The experiment version.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        metavar="BATCHS",
        help="Batch size.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=120,
        metavar="PAD",
        help="Padding.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=80,
        metavar="Stride",
        help="Sliding window stride.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        metavar="PATCH",
        help="Patch size.",
    )
    parser.add_argument(
        "--geo_format",
        type=str,
        default="qupath",
        metavar="GEOFRMT",
        help="The geojson format. One of 'qupath', 'simple'.",
    )
    parser.add_argument(
        "--offsets",
        type=int,
        default=1,
        metavar="OFFSETS",
        help="Flag, whether the filename coords are added to the json ploygon coords.",
    )
    parser.add_argument(
        "--classes_type",
        type=str,
        required=None,
        metavar="TYPECLS",
        help=(
            "comma separated list of the cell type classes. ",
            "Has to be in order and include bg class",
        ),
    )
    parser.add_argument(
        "--classes_tissue",
        type=str,
        default=None,
        metavar="TISSCLS",
        help=(
            "comma separated list of the tissue type classes. ",
            "Has to be in order and include bg class",
        ),
    )

    args = parser.parse_args()
    run_infer_wsi_patches(args)
