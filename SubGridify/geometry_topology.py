'''
Geometry and topology operations on meshes
'''
import numpy as np

from .parameters import MAJOR_RADIUS_OF_EARTH

__all__ = [
    "meshResolution",
    "projectMeshToMercator",
    "triarea",
    "isInside",
]

def determine_vertex_element_connectivity(meshObject, maxConnectedVertex):
    '''
    Determine the connectivity of vertices to elements

    Parameters
    ----------
    meshObject : mesh from readMesh function
    maxConnectedVertex : maximum number of connected vertices

    Returns
    ------- 
    vertexConnect : connectivity of vertices to elements
    countArray : number of connected elements for each vertex

    '''
    numNode = meshObject[2]
    numEle = meshObject[3]
    meshConnectivity = meshObject[1].triangles

     # vertex connectivity
    vertexConnect = np.zeros((numNode, maxConnectedVertex)).astype(int)

    # keep track of how many connected elements
    countArray = np.zeros(numNode).astype(int)
    # loop through vertices to get connected elements
    for i in range(numEle):
        # find connected elements
        nm0 = meshConnectivity[i, 0]
        nm1 = meshConnectivity[i, 1]
        nm2 = meshConnectivity[i, 2]
        # fill in connectivity
        vertexConnect[nm0, countArray[nm0]] = i
        vertexConnect[nm1, countArray[nm1]] = i
        vertexConnect[nm2, countArray[nm2]] = i
        # fill count array
        countArray[nm0] += 1
        countArray[nm1] += 1
        countArray[nm2] += 1

    return vertexConnect, countArray

def find_max_connected_vertex(meshObject):
    '''
    Find the most connected vertex in a mesh

    Parameters
    ----------
    meshObject : mesh from readMesh function

    '''
    meshConnectivity = meshObject[1].triangles
    counts = np.bincount(meshConnectivity.flatten())
    commonVertex = np.argmax(counts)
    maxConnectedVertex = np.count_nonzero(meshConnectivity == commonVertex)
    return maxConnectedVertex

def meshResolution(meshObject, outputFilename):
    '''
    Calculate the mesh resolution and write to a file

    Parameters
    ----------
    meshObject : mesh from readMesh function
    outputFilename : name of the output file
    '''
    # get the mesh connectivity
    meshConnectivity = meshObject[1].triangles
    # get the x and y of mesh Vertices
    allVertLon = meshObject[0]["Longitude"]
    allVertLat = meshObject[0]["Latitude"]
    # create empty list of lists
    connectedVertices = [[] for _ in range(meshObject[2])]
    # create array to hold distances
    distToConnectedVertices = np.zeros(meshObject[2])
    # get vertex connectivity
    for i in range(meshObject[3]):
        currMeshConn = meshConnectivity[i]

        nm0 = currMeshConn[0]
        nm1 = currMeshConn[1]
        nm2 = currMeshConn[2]

        connectedVertices[nm0] = connectedVertices[nm0] + [
            currMeshConn[1],
            currMeshConn[2],
        ]
        connectedVertices[nm1] = connectedVertices[nm1] + [
            currMeshConn[0],
            currMeshConn[2],
        ]
        connectedVertices[nm2] = connectedVertices[nm2] + [
            currMeshConn[0],
            currMeshConn[1],
        ]

    # get unique values and calulate distances
    for i in range(meshObject[2]):
        # get unique values
        connectedVertices[i] = np.array(connectedVertices[i])
        connectedVertices[i] = np.unique(connectedVertices[i])

        # get x y of vertex of interest

        vertLon = allVertLon[i]
        vertLat = allVertLat[i]

        # now get the x y of the connected vertices

        conVertLon = allVertLon[connectedVertices[i]]
        conVertLat = allVertLat[connectedVertices[i]]

        # now calculate distances

        conVertDistances = (
            np.sqrt((conVertLon - vertLon) ** 2 + (conVertLat - vertLat) ** 2)
            * 111
            / 0.001
        )  # convert to meters

        # average these

        convertDistancesMean = np.mean(conVertDistances)

        # now add this to larger array

        distToConnectedVertices[i] = convertDistancesMean

    with open(outputFilename, "w+") as meshRes:
        meshRes.write("Averaged distance surrounding each vertex\n")

        for i in range(meshObject[2]):
            meshRes.write(str(i) + "\t" + str(distToConnectedVertices[i]) + "\n")
        


def projectMeshToMercator(lat, lon):
    """
    Project lat/lon to Mercator coordinates.
    """
    x = MAJOR_RADIUS_OF_EARTH* np.radians(lon)
    scale = x / lon
    y = (
        180.0
        / np.pi
        * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
        * scale
    )
    return x, y


def triarea(x1, y1, x2, y2, x3, y3):
    """
    Function to calculate the area of a triangle given the coordinates of the vertices
    Parameters
    ----------
    """
    area = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
    return area


def isInside(x1, y1, x2, y2, x3, y3, x, y, difCriteria):
    """
    Function to determine if a point is inside a triangle given the coordinates of the vertices
    """
    A = triarea(x1, y1, x2, y2, x3, y3)
    A1 = triarea(x, y, x2, y2, x3, y3)
    A2 = triarea(x1, y1, x, y, x3, y3)
    A3 = triarea(x1, y1, x2, y2, x, y)

    ADiff = abs(A - (A1 + A2 + A3))

    mask = ADiff < difCriteria

    return mask

