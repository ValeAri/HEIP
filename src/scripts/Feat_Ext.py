# Imports
from shapely.geometry import LineString
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import os
import src.func_utils as fu
import sys

def Feat_Ext(args: Dict[str, Any]) -> None:
	#import the json file with all the cell (shapely geometry)
	gson=os.path.join(args.input_dir,args.sampl_name+'_cells.json')
	cellDoc = pd.read_json(gson)
	cellDoc["geometry"] = cellDoc["geometry"].apply(shapely.geometry.shape)
	cells = gpd.GeoDataFrame(cellDoc).set_geometry('geometry')

	# Function to extract information regarding the numbers of Neoplastic, Inflammatory and Connective cells present in the WSI slide
	data,cells_list,poly_list=fu.cell_counting(cells)
	#print('Neoplastic cells: %i\nInflammatory cells: %i\nConnective cells: %i' %(data['neoplastic'],data['inflammatory'],data['connective']))
	# Function to extract the Shannon Index
	ShannonIndex=fu.Shannon_indx(data)        
	#print('Shannon Index: %.3f'% ShannonIndex)
	data['ShannonIndex']=ShannonIndex
	#Saving data
	data_dict=pd.DataFrame(data, index=[0])
	data_dict.to_csv(os.path.join(args.output_dir,sampl_name,'CellComposition.csv'))
	# Function to obtain all the features in one variable
	cell_feat=fu.cell_features(cells)
	cell_feat.to_csv(os.path.join(output_dir,sampl_name,'MorphoFeatures.csv'))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",s
        type=str,
        required=True,
        help="The input_dir name.",
    )
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
    
    args = parser.parse_args()
    feat_ext(args)