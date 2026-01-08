# This script will be used as a way of modularizing the data so that I can add and remove features as needed
# It will work by using numbers to represent the features and those numbers will be added to the call to the script
# which will then add the features to the dataset. This will allow me to easily add and remove features as needed by just not having those numbers in the call to the script
# The Features are as follows:
# 1. KPN

# Import libraries
from glob import glob
import os, pathlib, datetime
import pandas as pd
import geopandas as gpd
import rasterio as rio
import xarray as xr

# Load the dataset
def loadDataset(path):
    data = pd.read_csv(path)
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
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
def readKPNRasters(dir=r'KPN'):
    # Create a dictionary to hold the rasters
    folders = glob(os.path.join(dir, '*')) # Get all the files in the current directory
    rasters = {}
    for folder in folders:
        # Ensure the folder is a valid time period
        if os.path.isdir(folder) and int(pathlib.Path(folder).name.split('_')[0]) < datetime.datetime.now().year: 
            with rio.open(os.path.join(folder, 'koppen_geiger_0p1.tif')) as src:
                rasters[(int(folder.split('_')[0].split('\\')[1]), int(folder.split('_')[1]))] = [src, src.read(1)] 
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
    
    # Change KPN of 0 to 'O' for Ocean, which is not in the raster data
    df.replace({'KPN': {0: 'O'}}, inplace=True)

    return df.reset_index(drop=True)

# Now to load the legend and convert the KPN from a number to a string (A, B, C, D, E)
# Return a dictionary with the key as the string and the value as the numbers as a list
def loadKPNLegend(dir=r'KPN'):
    with open(os.path.join(dir, 'legend.txt')) as f:
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
    
    return df.reset_index(drop=True)

# One-hot encode the KPN
def oneHotEncodeKPN(df):
    df = df.copy()
    df = pd.concat([df, pd.get_dummies(df['KPN'], prefix='KPN', dtype=int)], axis=1)
    df.drop(columns=['KPN'], inplace=True)
    
    return df.reset_index(drop=True)

def addKPN(df, dir=r'KPN'):
    # Read in the rasters
    rasters = readKPNRasters(dir)
    # Get KPN for the dataset
    dfKPN = getKPN(df, rasters)
    # Load the legend
    legend = loadKPNLegend(dir)
    # Convert the KPN to the string representation
    dfKPN = convertKPN(dfKPN, legend)
    # One-hot encode the KPN
    dfKPN = oneHotEncodeKPN(dfKPN)

    return dfKPN
# End of KPN Classification Script

# Atmospheric Data Script
################################################################################
# This part will be used to add atmospheric variables
# 1. Read in the correct variable name from mapping
# 2. Open all netcdfs containing that variable
# 3. Combine the datasets by coordinates
# 4. For each row in the dataframe, get the value of the variable at the corresponding lat, lon, and date
# 5. Add the variable to the dataframe
################################################################################
# Function that opens all the netcdf datasets containing a specific variable
def open_datasets(variable_name, dir_path=r'HydroGFD/data_files/'):
    # Find all files matching the pattern
    files = glob(f'{dir_path}/{variable_name}*.nc')

    # Open multiple datasets and combine them by coordinates, overriding attributes as they are not consistent and unnecessary at this stage
    # dataset = xr.open_mfdataset(files, combine='by_coords', combine_attrs='override')
    dataset = xr.combine_by_coords([xr.open_dataset(f) for f in files], combine_attrs='override')
    return dataset

# Grab the nearest value from the given dataset and dataframe row
def attach_nearest_value(ds, df, var):
    # Lambda function that will get the nearest value for a given row
    nearest_value = lambda ds, row: ds.sel(
        time=row['Date'].strftime('%Y-%m-%d'),
        lat=row['Lat'],
        lon=row['Lon'],
        method='nearest'
    )[var].item()

    # Apply the function to each row in the dataframe
    return df.apply(lambda row: nearest_value(ds, row), axis=1)

# Add atmospheric data to the dataframe
def addAtmosData(df, feature, dir_path):
    # From the feature given, like precip, get the corresponding variable name in the dataset
    var_map = {
        'Precipitation': 'prAdjust',
        'Temperature': 'tasAdjust',
    }
    var_name = var_map.get(feature)

    # Open the datsets for the given variables
    ds = open_datasets(var_name, dir_path)

    # Attach the nearest values to the dataframe
    df[feature] = attach_nearest_value(ds, df, var_name)
    return df

# Main function that will be called by the script determining which features to add
def addFeatures(df):
    features = ['KPN', 'Precipitation', 'Temperature']
    functions = {
        'KPN': addKPN,
        'Precipitation': lambda df: addAtmosData(df, 'Precipitation', r'HydroGFD/data_files/'),
        'Temperature': lambda df: addAtmosData(df, 'Temperature', r'HydroGFD/data_files/'),
    }
    
    for feature in features:
        df = functions[feature](df)
    return df.drop(columns=['geometry', 'index'], errors='ignore')

if __name__ == "__main__":
    # Load training and test datasets
    dfTrain = loadDataset(r'GNIP\GNIP_Train.csv')
    dfTest = loadDataset(r'GNIP\GNIP_Test.csv')

    # Load the leave-one-out dataset
    dfLoo = loadDataset(r'Leave_Out_Points\Leave_Out_Points_GNIP (2025-07-22).csv')

    # Add the features
    dfTrain = addFeatures(dfTrain)
    dfTest = addFeatures(dfTest)
    dfLoo = addFeatures(dfLoo)

    # Save the datasets
    dfTrain.to_csv(r'DataTrain.csv', index=False)
    dfTest.to_csv(r'DataTest.csv', index=False)
    dfLoo.to_csv(r'Leave_Out_Points\DataLeaveOut.csv', index=False)