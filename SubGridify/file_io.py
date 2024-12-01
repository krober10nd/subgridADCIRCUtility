"""
File input out functions
"""

import os
import re

import matplotlib.tri as mtri
import netCDF4 as nc
import pandas as pd
import xarray as xr
import numpy as np

__all__ = [
    "readMesh",
    "importDEM",
    "importSubgridLookup",
    "readfort63",
    "readMaxele",
    "readManning",
    "read_control_file",
    "getSubgridVariables",
    "writeSubgridLookup",
]


def writeSubgridLookup(
    meshObject,
    outputFilename,
    desiredPhiList,
    depthsVertForLookup,
    HWVertForLookup,
    HGVertForLookup,
    cfVertForLookup,
    vertexUseList,
    cmfVertForLookup,
    cadvVertForLookup,
):
    '''
    Write the subgrid lookup table to a netcdf file
    '''
    numNode = meshObject[2]

    ncFile = nc.Dataset(outputFilename, mode="w", format="NETCDF4")

    # create dimensions
    ncFile.createDimension("numPhi", 11)  # number of possible phi values
    ncFile.createDimension("numNode", numNode)  # number of nodes in mesh

    # create variables
    # write variable dimensions transposed because FORTRAN will read them that way
    # only need to do it for the 3D arrays, handle 2 and 1D a different way.

    # wetAreaFraction set
    phiSet = ncFile.createVariable("phiSet", np.float32, "numPhi")

    # create array for depth in reduced vertex table
    wetFractionDepthVarVertex = ncFile.createVariable(
        "wetFractionVertex", np.float32, ("numNode", "numPhi")
    )

    # vertex wet total water depth
    wetTotWatDepthVarVertex = ncFile.createVariable(
        "wetTotWatDepthVertex", np.float32, ("numNode", "numPhi")
    )

    # vertex grid total water depth
    gridTotWatDepthVarVertex = ncFile.createVariable(
        "gridTotWatDepthVertex", np.float32, ("numNode", "numPhi")
    )

    # vertex coefficient of friction level 0
    cfVarVertex = ncFile.createVariable(
        "cfVertex", np.float32, ("numNode", "numPhi")
    )

    # variables showing which vertices are contained within
    # the subgrid area
    binaryVertexListVariable = ncFile.createVariable(
        "binaryVertexList", np.int32, ("numNode")
    )

    # vertex coefficient of friction level 1
    cmfVarVertex = ncFile.createVariable(
        "cmfVertex", np.float32, ("numNode", "numPhi")
    )
    # vertex advection correction
    cadvVarVertex = ncFile.createVariable(
        "cadvVertex", np.float32, ("numNode", "numPhi")
    )
    # create a binary list of vertices within the subgrid
    # if inside set to 1 otherwise 0
    binaryVertexList = np.zeros(len(vertexUseList)).astype("int")
    binaryVertexList[vertexUseList == False] = 1

    phiSet[:] = desiredPhiList
    wetFractionDepthVarVertex[:, :] = depthsVertForLookup
    wetTotWatDepthVarVertex[:, :] = HWVertForLookup
    gridTotWatDepthVarVertex[:, :] = HGVertForLookup
    cfVarVertex[:, :] = cfVertForLookup
    binaryVertexListVariable[:] = binaryVertexList
    cmfVarVertex[:, :] = cmfVertForLookup
    cadvVarVertex[:, :] = cadvVertForLookup

    ncFile.close()
    return f"Subgrid lookup table written to {outputFilename}"


def getSubgridVariables(meshObject, subgridVariableDict, desiredWaterLevel=0.0):
    """
    Function to interpolate subgrid variables to a desired water level

    Parameters
    ----------
    meshObject : mesh object from readMesh function
    subgridVariableDict: dictionary of all subgrid variables as numpy arrays
    desiredWaterLevel : elevation to interpolate subgrid variables from default = 0.0 m
    """
    # allocate arrays for each subgrid variable
    wetFraction = np.zeros(meshObject[2])
    wetTotWatDepth = np.zeros(meshObject[2])
    gridTotWatDepth = np.zeros(meshObject[2])
    cf = np.zeros(meshObject[2])
    cmf = np.zeros(meshObject[2])
    cadv = np.zeros(meshObject[2])

    for i in range(meshObject[2]):  # loop through vertices
        numGreater = 0

        for j in range(len(subgridVariableDict["phiSet"])):
            if subgridVariableDict["wetFractionDepth"][i, j] > desiredWaterLevel:
                numGreater += 1  # count that the cell is greater

        if numGreater == len(subgridVariableDict["phiSet"]):  # always wet
            wetFraction[i] = subgridVariableDict["phiSet"][0]
            wetTotWatDepth[i] = subgridVariableDict["wetTotWatDepth"][i, 0]
            gridTotWatDepth[i] = subgridVariableDict["gridTotWatDepth"][i, 0]
            cf[i] = subgridVariableDict["cf"][i, 0]
            cmf[i] = subgridVariableDict["cmf"][i, 0]
            cadv[i] = subgridVariableDict["cadv"][i, 0]

        elif numGreater == 0:  # always dry
            wetFraction[i] = subgridVariableDict["phiSet"][-1]
            wetTotWatDepth[i] = subgridVariableDict["wetTotWatDepth"][i, -1] + (
                desiredWaterLevel - subgridVariableDict["wetFractionDepth"][i, -1]
            )
            gridTotWatDepth[i] = subgridVariableDict["gridTotWatDepth"][i, -1] + (
                desiredWaterLevel - subgridVariableDict["wetFractionDepth"][i, -1]
            )
            cf[i] = subgridVariableDict["cf"][i, -1]
            cmf[i] = subgridVariableDict["cmf"][i, -1]
            cadv[i] = subgridVariableDict["cadv"][i, -1]

        else:
            wetFraction[i] = (
                desiredWaterLevel
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) / (
                subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
            ) * (
                subgridVariableDict["phiSet"][
                    len(subgridVariableDict["phiSet"]) - numGreater
                ]
                - subgridVariableDict["phiSet"][
                    len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) + subgridVariableDict[
                "phiSet"
            ][
                len(subgridVariableDict["phiSet"]) - numGreater - 1
            ]

            wetTotWatDepth[i] = (
                desiredWaterLevel
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) / (
                subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
            ) * (
                subgridVariableDict["wetTotWatDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
                - subgridVariableDict["wetTotWatDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) + subgridVariableDict[
                "wetTotWatDepth"
            ][
                i, len(subgridVariableDict["phiSet"]) - numGreater - 1
            ]

            gridTotWatDepth[i] = (
                desiredWaterLevel
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) / (
                subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
            ) * (
                subgridVariableDict["gridTotWatDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
                - subgridVariableDict["gridTotWatDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) + subgridVariableDict[
                "gridTotWatDepth"
            ][
                i, len(subgridVariableDict["phiSet"]) - numGreater - 1
            ]

            cf[i] = (
                desiredWaterLevel
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) / (
                subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
            ) * (
                subgridVariableDict["cf"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
                - subgridVariableDict["cf"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) + subgridVariableDict[
                "cf"
            ][
                i, len(subgridVariableDict["phiSet"]) - numGreater - 1
            ]

            cmf[i] = (
                desiredWaterLevel
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) / (
                subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
            ) * (
                subgridVariableDict["cmf"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
                - subgridVariableDict["cmf"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) + subgridVariableDict[
                "cmf"
            ][
                i, len(subgridVariableDict["phiSet"]) - numGreater - 1
            ]

            cadv[i] = (
                desiredWaterLevel
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) / (
                subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
                - subgridVariableDict["wetFractionDepth"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
            ) * (
                subgridVariableDict["cadv"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater
                ]
                - subgridVariableDict["cadv"][
                    i, len(subgridVariableDict["phiSet"]) - numGreater - 1
                ]
            ) + subgridVariableDict[
                "cadv"
            ][
                i, len(subgridVariableDict["phiSet"]) - numGreater - 1
            ]

    dict = {
        "wetFraction": wetFraction,
        "wetTotWatDepth": wetTotWatDepth,
        "gridTotWatDepth": gridTotWatDepth,
        "cf": cf,
        "cmf": cmf,
        "cadv": cadv,
    }

    return dict



def read_control_file(controlFilename):
    '''
    Read in the control file and return the variables
    '''
    with open(controlFilename) as ctrF:
       ctrF.readline()
       # change to shintaros r.strip with re to allow for spaces
       line = ctrF.readline().rstrip()
       line = re.split(" *= *", line)
       # get output file name
       outputFilename = line[1]
       line = ctrF.readline().rstrip()
       line = re.split(" *= *", line)
       # get mesh filename
       meshFilename = line[1]
       # read in mannings stuff
       line = ctrF.readline().rstrip()
       line = re.split(" *= *", line)
       defaultManning = line[1]  # if true just use the manning table in the code
       if defaultManning == "False":  # otherwise we need to read a table in
           line = ctrF.readline().rstrip()
           line = re.split(" *= *", line)
           manningsnTableFilename = line[1]  # get the mannings n table filename
       # now read in the elevation array
       line = ctrF.readline().rstrip()
       line = re.split(" *= *", line)
       minSurElev = float(line[1])
       line = ctrF.readline().rstrip()
       line = re.split(" *= *", line)
       maxSurElev = float(line[1])
       line = ctrF.readline().rstrip()
       line = re.split(" *= *", line)
       elevDisc = float(line[1])
       line = ctrF.readline().rstrip()
       line = re.split(" *= *", line)
       numDEMs = int(line[1])
       # get list of elevation datasets
       demFilenameList = []
       for i in range(numDEMs):
           line = ctrF.readline().rstrip()
           line = re.split(" *= *", line)
           demFilenameList.append(line[0])
       line = ctrF.readline().rstrip()
       line = re.split(" *= *", line)
       numLCs = int(line[1])
       # get list of landcover datasets
       landcoverFilenameList = []
       for i in range(numLCs):
           line = ctrF.readline().rstrip()
           line = re.split(" *= *", line)
           landcoverFilenameList.append(line[0])

    file_dict = {"outputFilename": outputFilename,
                 "meshFilename": meshFilename, 
                 "defaultManning": defaultManning, 
                 "manningsnTableFilename": manningsnTableFilename, 
                 "minSurElev": minSurElev, 
                 "maxSurElev": maxSurElev, 
                 "elevDisc": elevDisc, 
                 "numDEMs": numDEMs, 
                 "demFilenameList": demFilenameList, 
                 "numLCs": numLCs, 
                 "landcoverFilenameList": landcoverFilenameList}

    return file_dict


def readManning(manningsnFilename):
    """
    Read in mannings n file and return the values in a dictionary
    """
    assert os.path.exists(manningsnFilename), f"File {manningsnFilename} not found."
    manningsValues = open(manningsnFilename, "r")
    manningsnTable = {}
    for value in manningsValues:
        line = value.split()
        manningsnTable[int(line[0])] = float(line[1])

    return manningsnTable


def readMaxele(maxEleFilename):
    """
    Read in maxele file and return the maxele array

    Parameters
    ----------
    maxEleFilename : str
        Name of the maxele file with path

    """
    assert os.path.exists(maxEleFilename), f"File {maxEleFilename} not found."
    ds = nc.Dataset(maxEleFilename)
    maxEle = np.asarray(ds["zeta_max"][:])
    return maxEle


def readfort63(fort63Filename):
    """
    Read in fort63 file and return the fort63 array
    """
    assert os.path.exists(fort63Filename), f"File {fort63Filename} not found."
    ds = nc.Dataset(fort63Filename)
    zs = np.asarray(ds["zeta"][:])
    return zs


def importSubgridLookup(subgridFilename):
    """
    Function to import the subgrid lookup table and return the dictionary

    Parameters
    ----------
    subgridFilename : str
        Name of the subgrid file with path
    """
    assert os.path.exists(subgridFilename), f"File {subgridFilename} not found."
    data = nc.Dataset(subgridFilename)
    dict = {}
    dict["wetFractionDepth"] = np.array(data["wetFractionVertex"])
    dict["wetTotWatDepth"] = np.array(data["wetTotWatDepthVertex"])
    dict["gridTotWatDepth"] = np.array(data["gridTotWatDepthVertex"])
    dict["cf"] = np.array(data["cfVertex"])
    dict["cmf"] = np.array(data["cmfVertex"])
    dict["cadv"] = np.array(data["cadvVertex"])
    dict["phiSet"] = np.array(data["phiSet"])
    return dict


def importDEM(fileName):
    """
    Function to import a DEM using xarray and return the x, y, z arrays and resolutions.

    Parameters
    ----------
    fileName : str
        Name of the DEM file with

    """
    # check if file exists
    assert os.path.exists(fileName), f"File {fileName} not found."

    dem_data = xr.open_dataset(fileName)
    # TODO: Add support for when the variable name is not 'Band1'
    dem_band = list(dem_data.data_vars)[
        0
    ]  # Assuming the first data variable is the DEM
    z_array = dem_data[dem_band].values
    x_coords = dem_data[dem_band].x.values
    y_coords = dem_data[dem_band].y.values

    x_res = np.abs(x_coords[1] - x_coords[0])
    y_res = np.abs(y_coords[1] - y_coords[0])

    # Ensure NaNs are handled
    nodata_value = dem_data[dem_band].attrs.get("_FillValue", None)
    if nodata_value is not None:
        z_array[z_array == nodata_value] = np.nan

    return x_coords, y_coords, z_array, x_res, y_res


def readMesh(meshFilename):
    """
    Function to read in a mesh file and return the mesh object

    Parameters
    ----------
    meshFilename : str
        Name of the mesh file with path
    """
    # check if file exists
    assert os.path.exists(meshFilename), f"File {meshFilename} not found."

    x = []
    y = []
    z = []
    vertNum = []
    eleNum = []
    triangles = []

    with open(meshFilename) as gridFile:
        gridFile.readline()
        line = gridFile.readline().split()
        numEle = int(line[0])
        numVert = int(line[1])

        # import coordinates of points and elevations
        for i in range(numVert):
            line = gridFile.readline().split()
            vertNum.append(int(line[0]))

            x.append(float(line[1]))
            y.append(float(line[2]))
            z.append(float(line[3]))

        # NOTE: the -1 in the triangles assembly is to make it 0 indexed
        for i in range(numEle):
            line = gridFile.readline().split()
            eleNum.append(int(line[0]))
            triangles.append([int(line[2]) - 1, int(line[3]) - 1, int(line[4]) - 1])

    # triangulate mesh
    triang = mtri.Triangulation(x, y, triangles)

    # put xyz into dataframe for ease of use
    gridXYZ = pd.DataFrame(
        {"Vertex Number": vertNum, "Latitude": y, "Longitude": x, "Elevation": z}
    )

    return gridXYZ, triang, numVert, numEle
