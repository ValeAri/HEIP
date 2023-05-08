import math
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Tuple
from shapely.geometry import LineString, Polygon

__all__ = [
    "cell_counts",
    "cell_features",
    "cell_percentages",
    "shannon_index",
    "geo_area",
    "geo_solidity",
    "geo_eccentricity",
    "geo_rotation",
    "geo_roundness",
    "geo_axis",
    "geo_perimeter",
    "geoSferVolume",
    "geoEllisVolume"
]


def cell_counts(
    ref_cells: gpd.GeoDataFrame,
) -> Tuple[Dict[str, int], List[Polygon], List[str]]:
    """Count the cells of different types.

    I.e. count how many neoplastic, inflammatory and connective cells are
    in the entire WSI.

    Parameters
    ----------
        ref_cells : gpd.GeoDataFrame
            A geo-dataframe containing the info of the cells.

    Returns
    -------
        Tuple[Dict[str, int], List[Polygon], List[str]]:
            Dict mapping celltypes to counts,
            a list of the cell polygos (len = n_cells)
            a list of the celltypes (len = n_cells)
    """
    cells_list = []
    poly_list = []
    for i in range(0, len(ref_cells)):
        poly_list.append(ref_cells.geometry[i])
        cells_list.append(ref_cells.properties[i]["classification"]["name"])

    unique, counts = np.unique(np.array(cells_list), return_counts=True)
    data = dict(
        zip(
            ["neoplastic", "inflammatory", "connective"],
            [
                counts[np.where(unique == "neoplastic")][0],
                counts[np.where(unique == "inflammatory")][0],
                counts[np.where(unique == "connective")][0],
            ],
        )
    )
    return data, poly_list, cells_list


def cell_features(ref_cells: gpd.GeoDataFrame) -> pd.DataFrame:
    """Extract morphological feature sfrom the cells.

    Parameters
    ----------
        ref_cells : gpd.GeoDataFrame
            A geo-dataframe containing the info of the cells.

    Returns
    -------
        pd.DataFrame:
            A dataframe containing the features.
    """
    poly_list = []
    for i in range(0, len(ref_cells)):
        geom = [ref_cells.geometry[i]]
        minor_axis, major_axis, aspect_ratio = geo_axis(geom)

        poly_list.append(
            [
                ref_cells.properties[i]["classification"]["name"],
                geo_area(geom)[0],
                geoSferVolume(geom)[0],\
                geoEllisVolume(geom)[0],\
                geo_solidity(geom)[0],
                geo_eccentricity(geom)[0],
                geo_rotation(geom)[0],
                minor_axis[0],
                major_axis[0],
                aspect_ratio[0],
                geo_roundness(geom)[0],
                geo_perimeter(geom)[0],
            ]
        )

    df = pd.DataFrame(
        poly_list,
        columns=[
            "CellType",
            "Area",
            "Sf_Volume",
            "El_Volume",
            "Solidity",
            "Eccentricity",
            "Rotation",
            "MinorAxis",
            "MajorAxis",
            "Aspect ratio",
            "Roundness",
            "Perimeter",
        ],
    )
    return df


def cell_percentages(data: Dict[str, int]) -> Dict[str, float]:
    """Calculate the percentage of different celltypes.

    I.e the cell percentages for neoplastic, inflammatory and connective cell counts

    Parameters
    ----------
        data : Dict[str, int]
            Dict mapping celltypes to counts.

    Returns
    -------
        Dict[str, float]:
            A Dict mapping celltypes to normalized counts
    """
    N = sum(data.values())
    norm_data = data
    norm_data = {k: v / N for k, v in norm_data.items()}

    return norm_data


def shannon_index(data: Dict[str, int]) -> float:
    """Compute shannon index from cellcounts per type.

    Parameters
    ----------
        data : Dict[str, int]
            Dict mapping celltypes to counts.

    Returns
    -------
        float:
            The shannon index value.
    """

    def p(n, N):
        if n == 0:
            return 0
        else:
            return (float(n) / N) * np.log10(float(n) / N)

    N = sum(data.values())
    return -sum(p(n, N) for n in data.values() if n != 0)


def geo_area(cells_list: List[Polygon]) -> List[float]:
    """Compute the cell areas.

    Parameters
    ----------
        cells_list : List[Polygon]
            A list of shapely polygons representing cells.

    Returns
    -------
        List[float]:
            The cell areas in a list.
    """
    areas = []
    for i in range(0, len(cells_list)):
        areas.append(cells_list[i].area)
    return areas


def geo_solidity(cells_list: List[Polygon]) -> List[float]:
    """Compute the solidity feature of the cells.

    Parameters
    ----------
        cells_list : List[Polygon]
            A list of shapely polygons representing cells.

    Returns
    -------
        List[float]:
            The cell solidities in a list.
    """
    solidit = []
    for i in range(0, len(cells_list)):
        solidit.append((cells_list[i].area) / ((cells_list[i].convex_hull).area))

    return solidit


def geo_eccentricity(cells_list: List[Polygon]) -> List[float]:
    """Compute the eccentricity of the cells.

    Parameters
    ----------
        cells_list : List[Polygon]
            A list of shapely polygons representing cells.

    Returns
    -------
        List[float]:
            The cell eccentricities in a list.
    """
    eccentr = []
    for i in range(0, len(cells_list)):
        points = list(zip(*cells_list[i].minimum_rotated_rectangle.exterior.coords.xy))
        lengths = [
            LineString((points[j], points[j + 1])).length
            for j in range(len(points) - 1)
        ]

        # get major/minor axis measurements
        min_axis = min(lengths)
        max_axis = max(lengths)
        a = max_axis / 2
        b = min_axis / 2

        eccentr.append(np.sqrt(np.square(a) - np.square(b)) / a)

    return eccentr


def geo_rotation(cells_list: List[Polygon]) -> List[float]:
    """Compute the rotation of the cells.

    Parameters
    ----------
        cells_list : List[Polygon]
            A list of shapely polygons representing cells.

    Returns
    -------
        List[float]:
            The cell rotations in a list.
    """
    rot_ang = []
    for i in range(0, len(cells_list)):

        rect = np.array(cells_list[i].minimum_rotated_rectangle.exterior.coords)
        edges = []
        for d in np.diff(rect, axis=0):
            length = np.sqrt(d[0] ** 2 + d[1] ** 2)
            angle = np.arctan2(d[1], d[0])
            edges.append([length, angle])

        edges = np.array(edges)
        angle_value = edges[np.argmax(edges[:, 0])][1]
        # Check if the angle is between -pi/2 and pi/2
        if angle_value > np.pi / 2:
            angle_value -= np.pi
        elif angle_value < -np.pi / 2:
            angle_value += np.pi
        rot_ang.append(angle_value)

    return rot_ang


def geo_axis(cells_list: List[Polygon]) -> Tuple[List[float], List[float], List[float]]:
    """Compute minor and major axes and aspect ratio of the cells.

    Parameters
    ----------
        cells_list : List[Polygon]
            A list of shapely polygons representing cells.

    Returns
    -------
        Tuple[List[float], List[float], List[float]]
            The minor axis of the cells in a list.
            The major axis of the cells in a list.
            The aspect ratio of the cells in a list.
    """
    minor_axis = []
    major_axis = []
    aspect_ratio = []
    for i in range(0, len(cells_list)):
        mbr_points = list(
            zip(*cells_list[i].minimum_rotated_rectangle.exterior.coords.xy)
        )

        # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [
            LineString((mbr_points[i], mbr_points[i + 1])).length
            for i in range(len(mbr_points) - 1)
        ]

        # get major/minor axis measurements
        minor_axis.append(min(mbr_lengths))
        major_axis.append(max(mbr_lengths))
        aspect_ratio.append(max(mbr_lengths) / min(mbr_lengths))

    return minor_axis, major_axis, aspect_ratio


def geo_roundness(cells_list: List[Polygon]) -> List[float]:
    """Compute the roundness of the cells.

    Parameters
    ----------
        cells_list : List[Polygon]
            A list of shapely polygons representing cells.

    Returns
    -------
        List[float]:
            The cell roundnesses in a list.
    """
    roundness = []
    for i in range(0, len(cells_list)):
        mbr_points = list(
            zip(*cells_list[i].minimum_rotated_rectangle.exterior.coords.xy)
        )
        # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [
            LineString((mbr_points[i], mbr_points[i + 1])).length
            for i in range(len(mbr_points) - 1)
        ]
        # get major/minor axis measurements
        major_axis = max(mbr_lengths)
        roundness.append(4 * (cells_list[i].area / (math.pi * (major_axis**2))))

    return roundness


def geo_perimeter(cells_list: List[Polygon]) -> List[float]:
    """Compute the perimeters of the cells.

    Parameters
    ----------
        cells_list : List[Polygon]
            A list of shapely polygons representing cells.

    Returns
    -------
        List[float]:
            The cell perimeters in a list.
    """
    perimeter = []
    for i in range(0, len(cells_list)):
        perimeter.append(cells_list[i].length)

    return perimeter
def geoSferVolume(cells_list: List[Polygon]) -> List[float]:
    """Compute the volume with sphere formula of the cells.

    Parameters
    ----------
        cells_list : List[Polygon]
            A list of shapely polygons representing cells.

    Returns
    -------
        List[float]:
            The cell perimeters in a list.
    """
    volume=[]
    for i in range(0,len(cells_list)):
        areas=cells_list[i].area
        radius = math.sqrt(areas / (4 * math.pi))
        V=(4/3)*math.pi * (radius ** 3)
        volume.append(V)
    return volume

def geoEllisVolume(cells_list: List[Polygon]) -> List[float]:
    """Compute the volume with ellipsoid formula of the cells.

    Parameters
    ----------
        cells_list : List[Polygon]
            A list of shapely polygons representing cells.

    Returns
    -------
        List[float]:
            The cell perimeters in a list.
    """
    volume=[]
    minor_axis,major_axis,aspect_ratio=geo_axis(cells_list)
    for i in range(0,(len(minor_axis))): 
        V= (4/3) * math.pi * ((minor_axis[i]/2) ** 2) * major_axis[i]/2
        volume.append(V)
    return volume
