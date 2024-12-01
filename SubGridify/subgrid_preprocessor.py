# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:17:03 2020

@author: jlwoodr3
@coauthor: krober (K. Roberts)

"""
# systme imports
import sys
import time

# third party imports
import numpy as np
from scipy.interpolate import griddata

# local imports
from . import file_io as io
from . import geometry_topology as geom
from . import parameters as param 

# unpack parameters (should be immutable)
landCoverValues = param.landCoverValues
DEFAULT_MANNING = param.DEFAULT_MANNING
landCoverToManning = param.landCoverToManning
MIN_CF = param.MIN_CF

__all__ = ['plotSubgridVariable', 
           'downscaleResults', 
           'calculateSubgridCorrection']


def plotSubgridVariable(meshObject, subgridVariable, levels=20):
    '''
    Function to plot a subgrid variable on the mesh

    Parameters
    ----------
    meshObject : mesh object from readMesh function
    subgridVariable : subgrid variable to plot
    levels : number of levels to plot
    '''
    import cmocean
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(9, 9))
    ax1.set_aspect("equal")
    tcf = ax1.tricontourf(
        meshObject[1],
        subgridVariable,
        cmap=cmocean.cm.rain,
        levels=levels,
        extend="both",
    )
    ax1.triplot(meshObject[1], color="k", linestyle="-", linewidth=0.25)
    cbar = fig1.colorbar(tcf, extendrect=True)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Elevation (m)", rotation=270, fontsize=14)
    ax1.set_xlabel("Longitude", fontsize=20)
    ax1.set_ylabel("Latitude", fontsize=20)
    return fig1, ax1


def downscaleResults(meshObject, demObject, result, interpMethod="nearest"):
    '''
    Perform downscaling of results to the DEM resolution

    Parameters
    ----------
    meshObject : mesh from readMesh function
    demObject : dem from importDEM function
    result : either maxEle or single timestep of fort63 results
        obtained from their respective read tools
    interpMethod : method you want to interpolate with
        default is NN but could also do linear or cubic

    Returns
    -------
    interp : interpolated results

    '''

    interp = griddata(
        np.array((meshObject[0]["Longitude"], meshObject[0]["Latitude"])).T,
        result,
        (demObject[0], demObject[1]),
        method=interpMethod,
    )
    # perform downscaling
    interpMinusDEM = interp - demObject[2]
    interp[interpMinusDEM < 0] = np.nan

    return interp  



def calculateSubgridCorrection(controlFilename):
    '''
    This function calculates the subgrid correction factors for a given mesh and set of DEMs    

    Parameters
    ----------
    controlFilename : string
        name of the control file

    '''
    controlFileDict = io.read_control_file(controlFilename)

    # TODO KJR: re-introduce custom landcover values

    # Unpack the control file dictionary
    minSurElev = controlFileDict['minSurElev']
    maxSurElev = controlFileDict['maxSurElev']
    elevDisc = controlFileDict['elevDisc']
    meshFilename = controlFileDict['meshFilename']
    demFilenameList = controlFileDict['demFilenameList']
    landcoverFilenameList = controlFileDict['landcoverFilenameList']

    # surface elevation array for calcuations
    surfaceElevations = np.round(
        np.arange(minSurElev, maxSurElev + elevDisc, elevDisc), 2
    ).astype("float32")
    num_SfcElevs = len(surfaceElevations)

    # now read in the mesh
    mesh = io.readMesh(meshFilename)
    meshConnectivity = mesh[1].triangles
    meshLon = np.asarray(mesh[0]["Longitude"]).astype("float32")
    meshLat = np.asarray(mesh[0]["Latitude"]).astype("float32")
    numNode = mesh[2]
    numEle = mesh[3]

    # first mind the maximum number of elements connected to a vertex
    maxConnectedVertex = geom.find_max_connected_vertex(mesh)

    # KJR: 8 is used because each sub area will have 4 vertices so listing lon then lat
    vertexData = np.empty((numNode, maxConnectedVertex, 8))
    vertexData[:, :, :] = np.nan

    vertexConnect, countArray = geom.determine_vertex_element_connectivity(mesh)

    # fill this vertex Data Array
    for i in range(numNode):
        # find connected elements (this is the slowest part)
        connectedElements = vertexConnect[i, : countArray[i]]
        # fill in vertex data
        for j in range(len(connectedElements)):
            # get vertices of subelement
            ele = connectedElements[j]
            # other vertices besides the one in question
            otherVertices = meshConnectivity[ele, meshConnectivity[ele, :] != i]
            # order vertices how you want
            # start with vertex in question
            nm0 = i
            nm1 = otherVertices[0]
            nm2 = otherVertices[1]
            # get lon and lat of vertices
            vertexNumbers = [nm0, nm1, nm2]
            vertexLon = meshLon[vertexNumbers]
            vertexLat = meshLat[vertexNumbers]
            # now get the centroid of the element
            centroidLon = (vertexLon[0] + vertexLon[1] + vertexLon[2]) / 3
            centroidLat = (vertexLat[0] + vertexLat[1] + vertexLat[2]) / 3
            # get mid point of each vertex connected to vertex of interest
            midPointLon1 = (meshLon[nm0] + meshLon[nm1]) / 2
            midPointLon2 = (meshLon[nm0] + meshLon[nm2]) / 2
            midPointLat1 = (meshLat[nm0] + meshLat[nm1]) / 2
            midPointLat2 = (meshLat[nm0] + meshLat[nm2]) / 2
            # now add this data to vertex array
            subAreaPerimeter = np.array(
                (
                    meshLon[nm0],
                    midPointLon1,
                    centroidLon,
                    midPointLon2,
                    meshLat[nm0],
                    midPointLat1,
                    centroidLat,
                    midPointLat2,
                )
            )
            vertexData[i, j, :] = subAreaPerimeter

    # get an array of max and min vertex area coordinates
    vertexAreaMinLon = np.nanmin(vertexData[:, :, :4], axis=(1, 2))
    vertexAreaMaxLon = np.nanmax(vertexData[:, :, :4], axis=(1, 2))
    vertexAreaMinLat = np.nanmin(vertexData[:, :, 4:], axis=(1, 2))
    vertexAreaMaxLat = np.nanmax(vertexData[:, :, 4:], axis=(1, 2))

    # TODO: Explain each of these arrays with a comment
    # preallocate arrays to store vertex subgrid data
    wetFractionVertex = np.empty((numNode, num_SfcElevs))*np.nan

    totWatDepthVertex = np.empty(wetFractionVertex.shape)*np.nan

    wetTotWatDepthVertex = np.empty(wetFractionVertex.shape)*np.nan

    cfVertex = np.empty(wetFractionVertex.shape)*np.nan

    cmfVertex = np.empty(wetFractionVertex.shape)*np.nan

    cadvVertex = np.empty(wetFractionVertex.shape)*np.nan

    # create a list of vertices contained within the subgrid area
    vertexUseList = np.ones(numNode, dtype=bool)

    # keep track of total calc time
    startTotal = time.time()

    for i in range(len(demFilenameList)):

        # all variables the same as before
        elevationData = io.importDEM(demFilenameList[i])
        landcoverData = io.importDEM(landcoverFilenameList[i])

        # get data out of dems
        bathyTopo = elevationData[0].astype("float32")
        lon = elevationData[3]
        lat = elevationData[4]

        elevationData = None  # deallocate

        manningsn = landcoverData[0].astype("float32")  # array of mannings n values

        landcoverData = None  # deallocate

        # convert landcover values to mannings n values
        for value in landCoverValues:
            manningsn[manningsn == value] = landCoverToManning[value]

        # set nan values to 0.02
        manningsn[np.isnan(manningsn)] = DEFAULT_MANNING

        # get bounds of raster data
        demMinLon = np.min(lon)
        demMaxLon = np.max(lon)

        demMinLat = np.min(lat)
        demMaxLat = np.max(lat)

        # see what vertices lie within this dem
        minLonWithin = np.array(vertexAreaMinLon > demMinLon)
        maxLonWithin = np.array(vertexAreaMaxLon < demMaxLon)

        minLatWithin = np.array(vertexAreaMinLat > demMinLat)
        maxLatWithin = np.array(vertexAreaMaxLat < demMaxLat)

        # get all within and only use vertices that have not already been used
        allWithin = (
            minLonWithin
            * maxLonWithin
            * minLatWithin
            * maxLatWithin
            * vertexUseList
        )
        idxAllWithin = np.where(allWithin)[0]

        # update vertex use list
        vertexUseList[idxAllWithin] = False

        # now loop through contained vertices and perform calculations
        # for vertex in idxAllWithin:
        for j in range(len(idxAllWithin)):

            start = time.time()

            # find how many connected elements
            conElementCount = countArray[idxAllWithin[j]]

            # create array to hold vertex area data
            vertexSubArea = np.zeros((conElementCount, 1))

            # temporarily allocate for each subarea variable
            tempwetFractionData = np.zeros((conElementCount, num_SfcElevs))
            # just set the rest equal to this because we are just preallocating 0 arrays
            temptotWatDepthData = np.zeros((conElementCount, num_SfcElevs))
            tempwetTotWatDepthData = np.zeros((conElementCount, num_SfcElevs))
            tempcfData = np.zeros((conElementCount, num_SfcElevs))
            tempcmfData = np.zeros((conElementCount, num_SfcElevs))
            tempcadvData = np.zeros((conElementCount, num_SfcElevs))

            # now loop through connected elements
            for k in range(conElementCount):

                # get the sub area perimeter for the particular element
                subAreaPerimeterLon = vertexData[idxAllWithin[j], k, :4]
                subAreaPerimeterLat = vertexData[idxAllWithin[j], k, 4:]

                # cut down dem and landcover to each sub area
                # get locations of bounds
                minLonDEMWithin = lon > np.min(subAreaPerimeterLon)
                maxLonDEMWithin = lon < np.max(subAreaPerimeterLon)
                lonWithinIdx = np.where(minLonDEMWithin * maxLonDEMWithin)[0]
                minCol = np.min(
                    lonWithinIdx
                )  #  - bufferCells # create a cell buffer
                maxCol = np.max(lonWithinIdx)  #  + bufferCells
                demLonCut = lon[minCol:maxCol]
                minLatDEMWithin = lat > np.min(subAreaPerimeterLat)
                maxLatDEMWithin = lat < np.max(subAreaPerimeterLat)
                latWithinIdx = np.where(minLatDEMWithin * maxLatDEMWithin)[0]
                minRow = np.min(
                    latWithinIdx
                )  
                maxRow = np.max(latWithinIdx)  #  + bufferCells
                demLatCut = lat[minRow:maxRow]
                demBathyTopoCut = bathyTopo[minRow:maxRow, minCol:maxCol]
                manningsnCut = manningsn[minRow:maxRow, minCol:maxCol]
                lonGrid, latGrid = np.meshgrid(demLonCut, demLatCut)

                # split into 2 triangles
                triLon0 = subAreaPerimeterLon[:3]
                triLat0 = subAreaPerimeterLat[:3]

                # convert to meters
                tri0Meters = geom.projectMeshToMercator(
                    triLat0, triLon0
                )
                tri0Area = geom.triarea(
                    tri0Meters[0][0],
                    tri0Meters[1][0],
                    tri0Meters[0][1],
                    tri0Meters[1][1],
                    tri0Meters[0][2],
                    tri0Meters[1][2],
                )

                insideTri0 = geom.isInside(
                    triLon0[0],
                    triLat0[0],
                    triLon0[1],
                    triLat0[1],
                    triLon0[2],
                    triLat0[2],
                    lonGrid,
                    latGrid,
                    0.00000001,
                )

                triLon1 = subAreaPerimeterLon[[0, 2, 3]]
                triLat1 = subAreaPerimeterLat[[0, 2, 3]]
                tri1Meters = geom.projectMeshToMercator(
                    triLat1, triLon1
                )

                tri1Area = geom.triarea(
                    tri1Meters[0][0],
                    tri1Meters[1][0],
                    tri1Meters[0][1],
                    tri1Meters[1][1],
                    tri1Meters[0][2],
                    tri1Meters[1][2],
                )

                insideTri1 = geom.isInside(
                    triLon1[0],
                    triLat1[0],
                    triLon1[1],
                    triLat1[1],
                    triLon1[2],
                    triLat1[2],
                    lonGrid,
                    latGrid,
                    0.00000001,
                )

                # now combine the two triangles and find the points inside

                insideSubElement = np.logical_or(insideTri0, insideTri1)

                # count the number of subgrid cells within the subelement

                cellsInSubElement = np.count_nonzero(insideSubElement)

                # if there are no cells within the element the DEM is too coarse
                # you must decrease the DEM resolution in this area
                if cellsInSubElement == 0:
                    sys.exit("DEM {0} resolution too coarse!".format(i))

                # get just he bathy topo inside the sub element

                # bathyTopoInsideSubElement = demBathyTopoCut*insideSubElement
                # take out unnecessary calc
                bathyTopoInsideSubElement = demBathyTopoCut[
                    insideSubElement == True
                ]

                # get area of sub element
                vertexSubArea[k] = (
                    tri0Area + tri1Area
                )  # used for area weighting later

                # TODO: JLW: remove these lines
                manningsnInside = manningsnCut[insideSubElement == True]

                # get the total water depth at each surface elevation
                temptotWatDepth = (
                    surfaceElevations[:, None] - bathyTopoInsideSubElement
                )

                # count the number of wet cells

                wetCellsInSubArea = temptotWatDepth > 0.0001

                wetCellsInSubAreaCount = np.sum(wetCellsInSubArea, axis=1).astype(
                    "float32"
                )

                # now set tot water depth of dry cells to nan

                temptotWatDepth[temptotWatDepth < 0.0001] = np.nan

                # add to wet frac array

                tempwetFractionData[k, :] = (
                    wetCellsInSubAreaCount / cellsInSubElement
                )

                # add to total water depth array
                temptotWatDepthData[k, :] = (
                    np.nansum(temptotWatDepth, axis=1) / cellsInSubElement
                )

                # get wet total water depth and coefficient of friction

                # find the mannings for only wet areas then 0 the rest for
                # use in calculations

                # manningsnCutNoNaNWet = manningsnCutNoNaN * wetCellsInSubArea
                manningsnCutNoNaNWet = manningsnInside * wetCellsInSubArea

                tempcf = (9.81 * manningsnCutNoNaNWet**2) / (
                    temptotWatDepth ** (1 / 3)
                )
                # set 0 tempcf to nan to prevent 0 divide
                tempcf[tempcf == 0] = np.nan

                # make wet cells in sub area count nan when == 0 so we don't get divide by 0 issue
                wetCellsInSubAreaCount[wetCellsInSubAreaCount == 0.0] = np.nan
                # calculate wet total water depth
                tempwetTotWatDepthData[k, :] = (
                    np.nansum(temptotWatDepth, axis=1) / wetCellsInSubAreaCount
                )
                # calculate bottom friction coefficient
                tempcfData[k, :] = (
                    np.nansum(tempcf, axis=1) / wetCellsInSubAreaCount
                )
                # calculate rv for use in advection and bottom friction correction
                rv = tempwetTotWatDepthData[k, :] / (
                    np.nansum(
                        (temptotWatDepth ** (3 / 2)) * (tempcf ** (-1 / 2)), axis=1
                    )
                    / wetCellsInSubAreaCount
                )
                # calcualte advection correction
                tempcadvData[k, :] = (
                    (1 / tempwetTotWatDepthData[k, :])
                    * (
                        np.nansum(temptotWatDepth**2 / tempcf, axis=1)
                        / wetCellsInSubAreaCount
                    )
                    * rv**2
                )
                # calculate bottom friction correction
                tempcmfData[k, :] = (
                    tempwetTotWatDepthData[k, :] * rv**2
                )  # this is correct I need <H>W * Rv**2
                # now fill the nans from this calculation which represent depths were nothing was wet
                # set wet total water depth to 0
                tempwetTotWatDepthData[
                    k, np.isnan(tempwetTotWatDepthData[k, :])
                ] = 0.0
                # set nan values to cf calculated from mean mannings n and 8 cm of water
                # tempcfData[k,np.isnan(tempcfData[k,:])] = 9.81*np.mean(manningsnCutNoNaN)**2/(0.08**(1/3))
                # tempcmfData[k,np.isnan(tempcmfData[k,:])] = 9.81*np.mean(manningsnCutNoNaN)**2/(0.08**(1/3))
                tempcfData[k, np.isnan(tempcfData[k, :])] = (
                    9.81 * np.mean(manningsnInside) ** 2 / (0.08 ** (1 / 3))
                )
                tempcmfData[k, np.isnan(tempcmfData[k, :])] = (
                    9.81 * np.mean(manningsnInside) ** 2 / (0.08 ** (1 / 3))
                )
                # set advection correction equal to 1.0
                tempcadvData[k, np.isnan(tempcadvData[k, :])] = 1.0

            # once the sub elements surrounding each vertex have been looped through, put all of he data on the elements
            areaTotalVertex = np.sum(vertexSubArea[:, 0])

            # check = np.sum(tempwetFractionData*vertexSubArea,axis=0)/areaTotalVertex
            wetFractionVertex[idxAllWithin[j], :] = (
                np.sum(tempwetFractionData * vertexSubArea, axis=0)
                / areaTotalVertex
            )
            totWatDepthVertex[idxAllWithin[j], :] = (
                np.sum(temptotWatDepthData * vertexSubArea, axis=0)
                / areaTotalVertex
            )
            wetTotWatDepthVertex[idxAllWithin[j], :] = (
                np.sum(tempwetTotWatDepthData * vertexSubArea, axis=0)
                / areaTotalVertex
            )
            cfVertex[idxAllWithin[j], :] = (
                np.sum(tempcfData * vertexSubArea, axis=0) / areaTotalVertex
            )
            cmfVertex[idxAllWithin[j], :] = (
                np.sum(tempcmfData * vertexSubArea, axis=0) / areaTotalVertex
            )
            cadvVertex[idxAllWithin[j], :] = (
                np.sum(tempcadvData * vertexSubArea, axis=0) / areaTotalVertex
            )
            # finish vertex loop and print time
            end = time.time()
            print(
                "Finish vertex {} in DEM {} took : {} s".format(
                    idxAllWithin[j], i, end - start
                )
            )

    # set minimums for bottom friction
    cfVertex[cfVertex < MIN_CF] = MIN_CF 
    cmfVertex[cmfVertex < MIN_CF] =MIN_CF 

    # total time
    endTotal = time.time()

    print("All calulations took {} s".format(endTotal - startTotal))

    # now I need to condense the lookup tables to only have 11 values corresponding to phi=0 to phi=1
    # KJR: why 11?
    start = time.time()
    desiredPhiList = np.linspace(0, 1, 11)
    depthsVertForLookup = np.ones((len(wetFractionVertex[:]), 11)) * -99999
    HGVertForLookup = np.ones((len(wetFractionVertex[:]), 11)) * -99999
    HWVertForLookup = np.ones((len(wetFractionVertex[:]), 11)) * -99999
    cfVertForLookup = np.ones((len(wetFractionVertex[:]), 11)) * -99999
    cmfVertForLookup = np.ones((len(wetFractionVertex[:]), 11)) * -99999
    cadvVertForLookup = np.ones((len(wetFractionVertex[:]), 11)) * 1.0

    # only loop through Nodes in subgrid area
    # get list of which vertices are inside the subgrid area
    vertsInSubArea = np.where(vertexUseList == False)[
        0
    ]  # confusing that I use false but it was for multiplication earlier

    for vert in vertsInSubArea:
        currPhiArray = wetFractionVertex[vert, :]

        # make sure that the phi array also gets fully wet and then proceed
        # otherwise just skip

        # for phi == 0 you want to find exactly where that is in the currPhiArray
        equalTo0 = np.where(currPhiArray == 0.0)[0]

        if len(equalTo0) != 0:  # if 0.0 exists in the array
            depthsVertForLookup[vert, 0] = surfaceElevations[equalTo0[-1]]
            HGVertForLookup[vert, 0] = totWatDepthVertex[vert, equalTo0[-1]]
            HWVertForLookup[vert, 0] = wetTotWatDepthVertex[vert, equalTo0[-1]]
            cfVertForLookup[vert, 0] = cfVertex[vert, equalTo0[-1]]
            cmfVertForLookup[vert, 0] = cmfVertex[vert, equalTo0[-1]]
            cadvVertForLookup[vert, 0] = cadvVertex[vert, equalTo0[-1]]

        else:  # so if it never gets fully dry set everything to the value corresponding to the first surface elevations
            depthsVertForLookup[vert, 0] = surfaceElevations[0]
            HGVertForLookup[vert, 0] = totWatDepthVertex[vert, 0]
            HWVertForLookup[vert, 0] = wetTotWatDepthVertex[vert, 0]
            cfVertForLookup[vert, 0] = cfVertex[vert, 0]
            cmfVertForLookup[vert, 0] = cmfVertex[vert, 0]
            cadvVertForLookup[vert, 0] = cadvVertex[vert, 0]

        # now check for when phi == 1.0 and find exactly where that is

        equalTo1 = np.where(currPhiArray == 1.0)[0]

        if len(equalTo1) != 0:  # if 1.0 exists in the array
            depthsVertForLookup[vert, -1] = surfaceElevations[equalTo1[0]]
            HGVertForLookup[vert, -1] = totWatDepthVertex[vert, equalTo1[0]]
            HWVertForLookup[vert, -1] = wetTotWatDepthVertex[vert, equalTo1[0]]
            cfVertForLookup[vert, -1] = cfVertex[vert, equalTo1[0]]
            cmfVertForLookup[vert, -1] = cmfVertex[vert, equalTo1[0]]
            cadvVertForLookup[vert, -1] = cadvVertex[vert, equalTo1[0]]

        else:  # if there is nothing that is equal to 1 (so never gets fully wet, just set everything to correspind to the last surface elevation)
            depthsVertForLookup[vert, -1] = surfaceElevations[-1]
            HGVertForLookup[vert, -1] = totWatDepthVertex[vert, -1]
            HWVertForLookup[vert, -1] = wetTotWatDepthVertex[vert, -1]
            cfVertForLookup[vert, -1] = cfVertex[vert, -1]
            cmfVertForLookup[vert, -1] = cmfVertex[vert, -1]
            cadvVertForLookup[vert, -1] = cadvVertex[vert, -1]

        # now for everything else

        for k in range(1, len(desiredPhiList) - 1):
            desiredPhi = desiredPhiList[k]
            greaterThan = np.where(currPhiArray > desiredPhi)[0]

            if (
                len(greaterThan) == 0
            ):  # so if nothing in the currPhiArray is greater than the desired phi
                # set everything to correspond to the last surface elevation

                depthsVertForLookup[vert, k] = surfaceElevations[-1]
                HGVertForLookup[vert, k] = totWatDepthVertex[vert, -1]
                HWVertForLookup[vert, k] = wetTotWatDepthVertex[vert, -1]
                cfVertForLookup[vert, k] = cfVertex[vert, -1]
                cmfVertForLookup[vert, k] = cmfVertex[vert, -1]
                cadvVertForLookup[vert, k] = cadvVertex[vert, -1]

            elif (
                greaterThan[0] == 0
            ):  # so if the first currphi index is greater than the desired phi
                # set everything to correspond to the first surfaceelevation

                depthsVertForLookup[vert, k] = surfaceElevations[0]
                HGVertForLookup[vert, k] = totWatDepthVertex[vert, 0]
                HWVertForLookup[vert, k] = wetTotWatDepthVertex[vert, 0]
                cfVertForLookup[vert, k] = cfVertex[vert, 0]
                cmfVertForLookup[vert, k] = cmfVertex[vert, 0]
                cadvVertForLookup[vert, k] = cadvVertex[vert, 0]

            else:  # this is where we interpolate
                greaterThan = greaterThan[0]
                lessThan = greaterThan - 1

                depthsVertForLookup[vert, k] = (
                    (desiredPhi - currPhiArray[lessThan])
                    / (currPhiArray[greaterThan] - currPhiArray[lessThan])
                ) * (
                    surfaceElevations[greaterThan] - surfaceElevations[lessThan]
                ) + (
                    surfaceElevations[lessThan]
                )
                HGVertForLookup[vert, k] = (
                    (desiredPhi - currPhiArray[lessThan])
                    / (currPhiArray[greaterThan] - currPhiArray[lessThan])
                ) * (
                    totWatDepthVertex[vert, greaterThan]
                    - totWatDepthVertex[vert, lessThan]
                ) + (
                    totWatDepthVertex[vert, lessThan]
                )
                HWVertForLookup[vert, k] = (
                    (desiredPhi - currPhiArray[lessThan])
                    / (currPhiArray[greaterThan] - currPhiArray[lessThan])
                ) * (
                    wetTotWatDepthVertex[vert, greaterThan]
                    - wetTotWatDepthVertex[vert, lessThan]
                ) + (
                    wetTotWatDepthVertex[vert, lessThan]
                )
                cfVertForLookup[vert, k] = (
                    (desiredPhi - currPhiArray[lessThan])
                    / (currPhiArray[greaterThan] - currPhiArray[lessThan])
                ) * (cfVertex[vert, greaterThan] - cfVertex[vert, lessThan]) + (
                    cfVertex[vert, lessThan]
                )
                cmfVertForLookup[vert, k] = (
                    (desiredPhi - currPhiArray[lessThan])
                    / (currPhiArray[greaterThan] - currPhiArray[lessThan])
                ) * (cmfVertex[vert, greaterThan] - cmfVertex[vert, lessThan]) + (
                    cmfVertex[vert, lessThan]
                )
                cadvVertForLookup[vert, k] = (
                    (desiredPhi - currPhiArray[lessThan])
                    / (currPhiArray[greaterThan] - currPhiArray[lessThan])
                ) * (cadvVertex[vert, greaterThan] - cadvVertex[vert, lessThan]) + (
                    cadvVertex[vert, lessThan]
                )

    end = time.time()
    print(
        "Reduction of partially wet vertices finished and took {} s".format(
            end - start
        )
    )

    # Create a new netcdf file with the subgrid data
    outputFilename = controlFilename.split(".")[0] + "_subgrid.nc"

    io.writeSubgridFile(outputFilename, 
                        outputFilename, 
                        desiredPhiList, 
                        depthsVertForLookup, 
                        HGVertForLookup, 
                        HWVertForLookup, 
                        cfVertForLookup, 
                        vertexUseList, 
                        cmfVertForLookup, 
                        cadvVertForLookup)

