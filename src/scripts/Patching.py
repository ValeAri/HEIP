# imports
import histoprep as hp
import os

def patching(args: Dict[str, Any]) -> None:
	""" Creation of patches from WSI """

	cutter = hp.SlideReader(args.sampl_name)
	metadata = cutter.save_tiles(
    output_dir=args.output_dir,
    coordinates=cutter.get_tile_coordinates( # Patch cutter info
        width=args.width, #patches dimension
        overlap=args.overlap,  #No overlap
        max_background=args.max_background), #Background accepted
        image_format=args.image_format, #Format of the saved patches
        quality= args.quality, #Quality of the saved patch images
	)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument(
        "--sampl_name",
        type=str,
        required=True,
        help="The sample name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path to the output dir.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1000,
        help="The width of the patches.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="The overlap of the patches.",
    )
    parser.add_argument(
        "--max_background",
        type=float,
        default=0.95,
        help="The percentage of background accepted.",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default='png',
        help="The format of the image (patches).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=100,
        help="The quality of the image (patches).",
    )


    args = parser.parse_args()
    patching(args)