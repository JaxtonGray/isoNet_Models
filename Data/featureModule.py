# This script will be used to extract features to add to the dataset. It will be a collection of functions that 
# will be used to extract features from the dataset. The type of features to be added will be determined by the
# Numbers inputed when calling the function in the command line.  It will export new dataset called dataTrain.csv and dataTest.csv
# The features that can be added are (NOTE: More will be added as needed):
# 1. KPN: Koppen Climate Classification
# (NOTE: Precipitation and Temperature data are already included in the base dataset)

# Importing Libraries
import pandas as pd
import numpy as np
import geopandas as gpd # For ease of use with geospatial data
import os
import sys

# Import the base dataset
def importData():
    # Importing the dataset
    trainGNIP = pd.read_csv('GNIP/GNIP_train.csv')
    testGNIP = pd.read_csv('GNIP/GNIP_test.csv')

    # Converting the dataset to a geodataframe
    trainGNIP = gpd.GeoDataFrame(trainGNIP, geometry=gpd.points_from_xy(trainGNIP.Lon, trainGNIP.Lat))
    testGNIP = gpd.GeoDataFrame(testGNIP, geometry=gpd.points_from_xy(testGNIP.Lon, testGNIP.Lat))

    return trainGNIP, testGNIP

# Function to add Koppen Climate Classification
def addKPN(trainGNIP, testGNIP):
    kpnGrid = gpd.read_file('KPN/KPN_1976-2000.geojson')

    # # Open the text file with the gridcode conversions and convert it to a dictionary with this structure
    # {'A': [61,62]...} and so forth
    gridCodeDict = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[]}
    with open('KPN_Gridcode_Convert.txt') as f:
        lineList = f.readlines()

    lineList = [x.strip() for x in lineList]
    lineList = [x.split(' ... ') for x in lineList]

    for line in lineList:
        if line[1][0] in gridCodeDict.keys():
            gridCodeDict[line[1][0]].append(int(line[0]))


def main():
    # Import the dataset
    trainGNIP, testGNIP = importData()
    trainGNIP.head()
main()