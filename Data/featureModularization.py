# This script will be used as a way of modularizing the data so that I can add and remove features as needed
# It will work by using numbers to represent the features and those numbers will be added to the call to the script
# which will then add the features to the dataset. This will allow me to easily add and remove features as needed by just not having those numbers in the call to the script
# The Features are as follows:
# 1. KPN

# Import libraries
from glob import glob
import sys
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio

# Load the dataset
def loadDataset(path):
    data = pd.read_csv(path)
    data['Date'] = pd.to_datetime(data['Date'])
    df = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Lon, data.Lat))
    return df

# KPN Classification Script
################################################################################
# This script will be used to add the koppen-geiger climate classification to
# the dataset. The script will do the following:
# 1. Load the dataset
# 2. Cycle through each row in the dataset and get the latitude, longitude, and date
# 3. Use the latitude and longitude to get the climate classification from the corresponding time period
# 4. Add the climate classification to the dataset
# 5. Read in the legend and convert the climate classification from a number to a string (A, B, C, D, E)
# 6. One-hot encode the climate classification
# 7. Save the dataset
################################################################################

# Read in the climate classification raster directories and assign them to the corresponding time period in
# a dictionary. Storing the specific raster to a key represeting the time period max min. The following will be a list:
# 1. The raster object
# 2. The first and only band of the raster as a numpy array
def readRasters():
    # Create a dictionary to hold the rasters
    files = glob(r'KPN\*') # Get all the files in the current directory
    files.remove(r'KPN\2041_2070') # Remove the directories that are future projections
    files.remove(r'KPN\2071_2099') # Remove the directories that are future projections
    rasters = {}
    for file in files:
        if os.path.isdir(file):
            with rio.open(file + r'\koppen_geiger_0p1.tif') as src:
                rasters[(int(file.split('_')[0].split('\\')[1]), int(file.split('_')[1]))] = [src, src.read(1)] 
        else:
            continue
    return rasters
    

# Get the climate classification for the dataframe by iterating through it and using the correct
# raster for the time period and latitude and longitude of the row
def getKPN(df, rasters):
    df.set_crs(rasters[(1961, 1990)][0].crs, inplace=True)

    for i, point in df.iterrows():
        # Find the correct raster for the time period
        for time in rasters.keys():
            if point['Date'].year >= time[0] and point['Date'].year <= time[1]:
                date = time
            elif point['Date'].year > 2020:
                date = (1991, 2020)
            elif point['Date'].year < 1901:
                date = (1901, 1930)

        # Get the climate classification
        row, col = rasters[date][0].index(point.geometry.x, point.geometry.y)
        df.at[i, 'KPN'] = rasters[date][1][row, col]

    # Remove values that are equal to 0, ocean values or values that are not in the raster
    df = df[df['KPN'] != 0]

    return df.reset_index()

# Now to load the legend and convert the KPN from a number to a string (A, B, C, D, E)
# Return a dictionary with the key as the string and the value as the numbers as a list
def loadLegend():
    with open(r'KPN\legend.txt') as f:
        legend = f.readlines()
        legend = [line.strip().split(':') for line in legend][3:33] # Remove header and footer
        legend = [(line[0].strip(), line[1].strip()[0]) for line in legend]
        legendDict = {}

        for line in legend:
            if line[1] in legendDict.keys():
                legendDict[line[1]].append(int(line[0]))
            else:
                legendDict[line[1]] = [int(line[0])]
                
    return legendDict

# Cycle through the dataframe and convert the KPN to the string representation
def convertKPN(df, legend):
    df = df.copy()
    df['temp'] = df['KPN'].astype(str)
    for key, values in legend.items():
        df.loc[df['KPN'].isin(values), 'temp'] = key
    df.drop(columns=['KPN'], inplace=True)
    df.rename(columns={'temp': 'KPN'}, inplace=True)
    return df

# One-hot encode the KPN
def oneHotEncodeKPN(df):
    df = df.copy()
    df = pd.concat([df, pd.get_dummies(df['KPN'], prefix='KPN', dtype=int)], axis=1)
    df.drop(columns=['KPN'], inplace=True)
    
    return df

def addKPN(df):
    # Read in the rasters
    rasters = readRasters()
    # Get KPN for the dataset
    dfKPN = getKPN(df, rasters)
    # Load the legend
    legend = loadLegend()
    # Convert the KPN to the string representation
    dfKPN = convertKPN(dfKPN, legend)
    # One-hot encode the KPN
    dfKPN = oneHotEncodeKPN(dfKPN)

    return dfKPN


################################################################################

# Main function that will be called by the script determining which features to add
def addFeatures(df):
    features = ['KPN']

    for feat in features:
        if feat == 'KPN':
            df = addKPN(df)
        else:
            continue
    
    return df.drop(columns=['geometry', 'index'])

def main():
    # Load training and test datasets
    dfTrain = loadDataset(r'GNIP\GNIP_Train.csv')
    dfTest = loadDataset(r'GNIP\GNIP_Test.csv')

    # Add the features
    dfTrain = addFeatures(dfTrain)
    dfTest = addFeatures(dfTest)

    # Save the datasets
    dfTrain.to_csv(r'DataTrain.csv', index=False)
    dfTest.to_csv(r'DataTest.csv', index=False)

main()