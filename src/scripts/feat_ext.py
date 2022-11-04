import argparse
import pandas as pd
import geopandas as gpd
import shapely
import os
import src.func_utils as fu

from typing import Dict, Any


def feat_ext(args: Dict[str, Any]) -> None:
    """Extract features from input geojson file."""
    gson = os.path.join(args.input_dir, args.sample_name + "_cells.json")
    cell_doc = pd.read_json(gson)
    cell_doc["geometry"] = cell_doc["geometry"].apply(shapely.geometry.shape)
    cells = gpd.GeoDataFrame(cell_doc).set_geometry("geometry")

    # Function to extract information regarding the numbers of Neoplastic,
    # Inflammatory and Connective cells present in the WSI slide
    data, _, _ = fu.cell_counts(cells)

    # Function to extract the Shannon Index
    shannon_index = fu.shannon_index(data)
    data["ShannonIndex"] = shannon_index

    # Saving data
    data_dict = pd.DataFrame(data, index=[0])
    data_dict.to_csv(
        os.path.join(args.output_dir, args.sample_name, "CellComposition.csv")
    )

    # Function to obtain all the features in one variable
    cell_feat = fu.cell_features(cells)
    cell_feat.to_csv(
        os.path.join(args.output_dir, args.sample_name, "MorphoFeatures.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The input_dir name.",
    )
    parser.add_argument(
        "--sample_name",
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

    args = parser.parse_args()
    feat_ext(args)
