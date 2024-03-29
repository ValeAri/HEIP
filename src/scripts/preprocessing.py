# imports
import histoprep as hp
import os
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pickle import TRUE
import shutil


def preprocessing:
    # Load of metadata and thumbail image
    metadata=pd.read_csv(os.path.join(args.input_dir,'tile_metadata.csv'))
    thumbnail_path=os.path.join(args.input_dir,'thumbnail.jpeg')
    thumbnail=Image.open(thumbnail_path).convert('RGB')
    annotated_thumbnail = thumbnail.copy()
    annotated = ImageDraw.Draw(annotated_thumbnail)
    
    
    #Filter the patch with not informative infomation. The parameters and thresholds should be adjust according with dataset
    filtered_coordinates=[]
    filtered_xy=[]
    for i in range(0,len(metadata)):
        if metadata['hue_q10'][i]>args.hue_q10 or metadata['black_pixels'][i]>args.black_pixels or metadata['brightness_q10'][i]<args.brightness_q10 \
        or metadata['saturation_q50'][i]<args.saturation_q50 or metadata['laplacian_std'][i]<args.laplacian:

            filtered_coordinates.append(i)
            filtered_xy.append([metadata['y'][i],metadata['x'][i]])

    # Saving of the image with the eliminated patches
    width=args.width
    downsample = args.downsample
    w = h = int(width/downsample)
    for i in filtered_coordinates:
        x_d = round(metadata['x'][i]/downsample)
        y_d = round(metadata['y'][i]/downsample)
        annotated.rectangle([x_d, y_d, x_d+w, y_d+h]
                            , outline='red', width=2)
    # Deletion of the patches also in the medata file
    metadata = metadata.drop(labels=filtered_coordinates, axis=0)
    metadata=metadata.reset_index()
    annotated_thumbnail.save(os.path.join(output_dir,'thumbnail_eliminatedTile.jpeg'))
    #Deletion of the low quality patches
    #print('Inizial number of Tiles: ',len(os.listdir(os.path.join(output_dir,'tiles'))))
    #print('Number of tiles to eliminate: ',len(filtered_xy))
    for coord in filtered_xy:
        name_coord=output_dir+'/tiles/'+'x'+str(coord[1])+'_y'+str(coord[0])+'_w'+str(width)+'_h'+str(width)+'.png'
        os.remove(name_coord)
    #Code to identify all the tissue present in the slide, expecially if there are more than one!
    thumb=hp.SlideReader(os.path.join(output_dir,"thumbnail.jpeg"))
    image,mask =thumb.get_tissue_mask()
    
    kernel_size = (args.kernel_size,args.kernel_size) #Change the kernel size
    mask2 = cv2.dilate(mask, np.ones(kernel_size, np.uint8), iterations=5)
    #plt.imshow(mask2)
    contours, __ = cv2.findContours(
        mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(x) for x in contours])
    maxarea=max(areas) 
    minarea=maxarea-(maxarea*40/100) #Change how much reduce the maximal area

    #Find the bounding boxes of each tissue
    cc=[]
    for cnts in contours:
        aa=cv2.contourArea(cnts)
        if aa>=minarea:
            cc.append(cnts)
    
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in cc]
    #If there is one tissue in the slide, this code is used to change the name of each patch to the required format. Instead, if there are more tissue in the slide, this code allows
    #to save the patches in separated folders according with the tissues
    for i in range(0,len(metadata)):
        j=1
        x_d = round(metadata['x'][i]/downsample)
        y_d = round(metadata['y'][i]/downsample)
        
        if len(bounding_boxes)>1:
            # Load the thumbnail image
            thumbnail_path=os.path.join(output_dir,'thumbnail.jpeg')
            thumbnail=Image.open(thumbnail_path).convert('RGB')
            annotated_thumbnail = thumbnail.copy()
            annotated = ImageDraw.Draw(annotated_thumbnail)
            w = h = int(width/downsample)
            color=['red','blue','green','yellow','black','magenta']
            for box in bounding_boxes:
                if (x_d>=box[0] and x_d<=box[0]+box[2]) and (y_d>=box[1] and y_d<=box[1]+box[3]):
                    annotated.rectangle([x_d, y_d, x_d+w, y_d+h]
                                            , outline=color[j-1], width=4)
                    folder=os.path.join(output_dir,'tiles_n'+str(j))
                    if not os.path.exists(folder): 
                        os.makedirs(folder)
                    #Name sostitution not only for moving the patches to the relative folder but also to change the name to the correct format required for the following steps
                    old_name_coord=output_dir+'/tiles/'+'x'+str(metadata['x'][i])+'_y'+str(metadata['y'][i])+'_w'+str(width)+'_h'+str(width)+'.png'
                    new_name_coord=folder+'/'+'x-'+str(metadata['x'][i])+'_y-'+str(metadata['y'][i])+'.png'
                    shutil.move(old_name_coord, new_name_coord)
                j+=1
            annotated_thumbnail.save(os.path.join(output_dir,'thumbnail_diff_tissue.jpeg'))

        elif len(bounding_boxes)==1:
            #Name sostitution to change the name to the correct format required for the following steps
                old_name_coord=output_dir+'/tiles/'+'x'+str(metadata['x'][i])+'_y'+str(metadata['y'][i])+'_w'+str(width)+'_h'+str(width)+'.png'
                new_name_coord=output_dir+'/tiles/'+'x-'+str(metadata['x'][i])+'_y-'+str(metadata['y'][i])+'.png'
                shutil.move(old_name_coord, new_name_coord)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",s
        type=str,
        required=True,
        help="The input_dir name.",
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
        "--downsample",
        type=int,
        default=64,
        help="The downsample of WSI thumbnail.",
    )
    parser.add_argument(
        "--kernel_size",
        type=float,
        default=30,
        help="The kernel size for the dilatation function for get all the tissue present in the slide",
    )
    parser.add_argument(
        "--hue_q10",
        type=int,
        default='144',
        help="The hue_q=0.1 parameter from Histoprep metadata.",
    )
    parser.add_argument(
        "--black_pixels",
        type=float,
        default=0.2,
        help="The black_pixels parameter from Histoprep metadata.",
    )
    parser.add_argument(
        "--brightness_q10",
        type=int,
        default=45,
        help="The brightness_q=0.1 parameter from Histoprep metadata.",
    )
    parser.add_argument(
        "--saturation_q50",
        type=int,
        default=30,
        help="The saturation_q=0.5 parameter from Histoprep metadata.",
    )
    parser.add_argument(
        "--laplacian",
        type=int,
        default=8,
        help="The sharpness_max parameter from Histoprep metadata .",
    )

    args = parser.parse_args()
    preprocessing(args)
