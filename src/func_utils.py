import numpy as np
import pandas as pd
from shapely.geometry import LineString
import math


# Function to count how many neoplastic, inflammatory and connective cells are in the entire WSI
def cell_counting(ref_cells):#(List of all cells(polygons) in the WSI)
    cells_list=[]
    poly_list=[]
    for i in range(0,len(ref_cells)):
        poly_list.append(ref_cells.geometry[i])
        cells_list.append(ref_cells.properties[i]['classification']['name'])
        
    unique, counts = np.unique(np.array(cells_list), return_counts=True)
    data=dict(zip(['neoplastic','inflammatory','connective'],[counts[np.where(unique=="neoplastic")][0]
        ,counts[np.where(unique=="inflammatory")][0]
        ,counts[np.where(unique=="connective")][0]]))
    return data,poly_list,cells_list

# Function to extract features from the cells in WSI
def cell_features(ref_cells): #(List of all cells(polygons) in the WSI)
    poly_list=[]
    for i in range(0,len(ref_cells)):
        geom=[ref_cells.geometry[i]]
        minor_axis,major_axis,aspect_ratio=geoAxis(geom)

        poly_list.append([ref_cells.properties[i]['classification']['name'],\
        geoArea(geom)[0],\
        geoSolidity(geom)[0],\
        geoEccentricity(geom)[0],\
        geoRotation(geom)[0],\
        minor_axis[0],\
        major_axis[0],\
        aspect_ratio[0],\
        geoRoundness(geom)[0],\
        geoPerimeter(geom)[0]])
        
    df=pd.DataFrame(poly_list,columns=['CellType','Area','Solidity','Eccentricity','Rotation','MinorAxis','MajorAxis','Aspect ratio','Roundness','Perimeter'])
    return(df)


# Function to calculate the normalized number of the neoplastic, inflammatory and connective cells
def N_F_C_norm(data):
    N = sum(data.values())
    norm_data=data
    norm_data = {k: v / N for k, v in norm_data.items()}
    return norm_data


# Function to calculate the shannon index from the data (dictionary)
def Shannon_indx(data):           
    def p(n, N):
        if n==0:
            return 0
        else:
            return (float(n)/N) * np.log10(float(n)/N)
            
    N = sum(data.values())
    return -sum(p(n, N) for n in data.values() if n is not 0)



#Morphological cells characteritic from shapely geometry
#Area
def geoArea(cells_list):
    areas=[]
    for i in range(0,len(cells_list)):
        areas.append(cells_list[i].area)
    return areas
#Solidity
def geoSolidity(cells_list):
    solidit=[]
    for i in range(0,len(cells_list)):
        solidit.append((cells_list[i].area)/((cells_list[i].convex_hull).area))
    return solidit
#Eccentricity
def geoEccentricity(cells_list):
    eccentr=[]
    for i in range(0,len(cells_list)):
        points = list(zip(*cells_list[i].minimum_rotated_rectangle.exterior.coords.xy))
        lengths = [LineString((points[j], points[j+1])).length for j in range(len(points) - 1)]

        # get major/minor axis measurements
        min_axis = min(lengths)
        max_axis = max(lengths)
        a = max_axis/2
        b = min_axis/2

        eccentr.append(np.sqrt(np.square(a)-np.square(b))/a)
    return eccentr
#Rotation
def geoRotation(cells_list):
    rot_ang=[]
    for i in range(0,len(cells_list)):

        rect=np.array(cells_list[i].minimum_rotated_rectangle.exterior.coords)
        edges = []
        for d in np.diff(rect,axis=0):
            length = np.sqrt(d[0]**2 + d[1]**2)
            angle = np.arctan2(d[1], d[0])
            edges.append([length, angle])
        
        edges=np.array(edges)
        angle_value=edges[np.argmax(edges[:,0])][1]
        #Check if the angle is between -pi/2 and pi/2
        if angle_value> np.pi/2:
            angle_value-= np.pi
        elif angle_value< -np.pi/2:
            angle_value+= np.pi
        rot_ang.append(angle_value)

    return rot_ang
#Minor/Major Axis and Aspect Ratio
def geoAxis(cells_list):
    minor_axis=[]
    major_axis=[]
    aspect_ratio=[]
    for i in range(0,len(cells_list)):
        mbr_points = list(zip(*cells_list[i].minimum_rotated_rectangle.exterior.coords.xy))

        # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]

        # get major/minor axis measurements
        minor_axis.append(min(mbr_lengths))
        major_axis.append(max(mbr_lengths))
        aspect_ratio.append(max(mbr_lengths)/min(mbr_lengths))

    return minor_axis, major_axis, aspect_ratio
#Roundness
def geoRoundness(cells_list):
    roundness=[]
    for i in range(0,len(cells_list)):
       mbr_points = list(zip(*cells_list[i].minimum_rotated_rectangle.exterior.coords.xy))
       # calculate the length of each side of the minimum bounding rectangle
       mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]
       # get major/minor axis measurements
       major_axis=max(mbr_lengths)
       roundness.append(4*(cells_list[i].area/(math.pi*(major_axis**2))))
    return roundness
#Perimeter
def geoPerimeter(cells_list):
    perimeter=[]
    for i in range(0,len(cells_list)):
        perimeter.append(cells_list[i].length)
    return perimeter
