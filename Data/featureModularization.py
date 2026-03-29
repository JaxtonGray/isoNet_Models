# This script will be used as a way of modularizing the data so that I can add and remove features as needed
# It will work by using numbers to represent the features and those numbers will be added to the call to the script
# which will then add the features to the dataset. This will allow me to easily add and remove features as needed by just not having those numbers in the call to the script
# The Features are as follows:
# 1. KPN
# 2. Altitude
# 3. Precipitation
# 4. Temperature
# 5. Teleconnection Indices (NAO and ENSO)

# Import libraries
from glob import glob
import os, pathlib, datetime, logging
import numpy as np
from scipy.ndimage import distance_transform_edt
import pandas as pd
import geopandas as gpd
import rasterio as rio
import xarray as xr

# Set up logging
# Check to make sure logging directory exists and works
os.makedirs(os.path.join('..', 'Logs'), exist_ok=True)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join('..', 'Logs', 'featureModularization.log'), 'w+')
formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Load the dataset
def loadDataset(path):
    logger.info(f'Loading dataset from {path}')
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
        # Ensure the folder is a directory and not a file
        if os.path.isdir(folder):
            folderYearStart, folderYearEnd = int(pathlib.Path(folder).name.split('_')[0]), int(pathlib.Path(folder).name.split('_')[1])
            
            # Only read in rasters for time periods that have already occurred
            if folderYearStart < datetime.datetime.now().year: 
                with rio.open(os.path.join(folder, 'koppen_geiger_0p1.tif')) as src:
                    rasters[(folderYearStart, folderYearEnd)] = [src, src.read(1)] 
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
    logger.info('Adding KPN feature to dataframe')
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
# 6. If the value is NaN, find the nearest valid grid point within 1 grid cell and use that value instead
# 7. For Antarctic points, open the ERA5 api dataset and get the value from there instead
################################################################################
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
    dataset = xr.combine_by_coords([xr.open_dataset(f, engine='h5netcdf') for f in files], combine_attrs='override')
    return dataset

# Function to find the nearest valid grid point for a given point and time within a specified buffer 
# and grid selection distance, if the original point is missing data. 
def find_nearest_valid_grid_xarrayds(ds, point, time, var, buffer=5, grid_select=2):
    # Arguments
    # ds: xarray dataset containing the variable of interest (e.g., 'prAdjust')
    # point: shapely Point object with the coordinates of the target location
    # time: the specific time for which to find the nearest valid grid point
    # var: the variable name to extract from the dataset
    # buffer: the size of the area around the point to consider for finding valid grid points
    # grid_select: the maximum distance (in grid units) to consider when selecting valid grid points
    # Returns
    # The value of the nearest valid grid point for the specified time
    dsFiltered = ds.sel(
        lon=slice(point.x - buffer, point.x + buffer), 
        lat=slice(point.y - buffer, point.y + buffer),
        time=time)[var]
    
    mask = np.isnan(dsFiltered.values)
    dist, ids = distance_transform_edt(mask, return_indices=True)

    # Add a condition to only consider points within the specified grid_select distance
    valid_mask = np.where(dist < grid_select, True, False)

    # Combine the original mask with the valid_mask to ensure we only consider valid points within the grid_select distance
    mask = np.logical_and(mask, valid_mask) 
    
    data = dsFiltered.values.copy()
    data[mask] = data[tuple(ids[:, mask])]
    
    filled_arr = xr.DataArray(data, coords=dsFiltered.coords, dims=dsFiltered.dims, attrs=dsFiltered.attrs)
    
    return filled_arr.sel(lon=point.x, lat=point.y, method='nearest').values.item()

# Grab nearest grid point values for variable in dataset for all rows via vectorized indexing. For rows that are NaN, apply 
# find_nearest_valid_grid to get the nearest valid grid point value within the specified buffer and grid selection distance.
def attach_nearest_value_vectorized(ds, df, var):
    # Extract the variable values at the nearest grid points for all rows in the dataframe
    times_da = xr.DataArray(df['Date'].dt.strftime('%Y-%m-%d').values, dims='points')
    lats_da = xr.DataArray(df['Lat'].values, dims='points')
    lons_da = xr.DataArray(df['Lon'].values, dims='points')

    # Use xarray's sel method with vectorized indexing to get the nearest values for all rows at once
    nearest_values = ds.sel(
        time=times_da,
        lat=lats_da,
        lon=lons_da,
        method='nearest'
    )[var]

    # If any values are NaN, apply the find_nearest_valid_grid function to those specific rows
    for i, value in enumerate(nearest_values.values):
        if np.isnan(value):
            point = df.iloc[i]['geometry']
            time = df.iloc[i]['Date'].strftime('%Y-%m-%d')
            nearest_values.values[i] = find_nearest_valid_grid_xarrayds(ds, point, time, var)

    return nearest_values

# Define a function that takes in a df, finds the Null Antarctica data, and fills with found value in the Antarctica DataFrame
def fillAntarcticaData(df, feature, antDF_Path = r'ERA5_Antarctica\GNIP_With_ERA5.csv'):
    # If the workflow is followed the antDF should contain all the missing points below -60 Lat, so we can just do a spatial join to fill in the missing data
    # Load antarctic data
    antDF = pd.read_csv(antDF_Path)
    antDF['Date'] = pd.to_datetime(antDF['Date'], utc=True)
    antGDF = gpd.GeoDataFrame(antDF, geometry=gpd.points_from_xy(antDF.Lon, antDF.Lat))

    # The antGDF has total average daily precipitation data in meters, I need it in kg/m2/s, which is roughly 86.4 times smaller
    # Since the data is in meters per day, we can convert it to kg/m2/s by dividing by 86.4
    # We will also rename the temperature, and precipitation columns to match the original df
    antGDF['Precipitation'] = antGDF['tp'] / 86.4
    antGDF['Temperature'] = antGDF['t2m']
    antGDF = antGDF.drop(columns=['tp', 't2m'])
    
    # Grab all the antarctic points from the original df, and then do a spatial join to fill in the missing data
    antGDF_df = df[df['Lat'] <= -60]
    # Spatial join to fill in the missing data
    joinedDF = gpd.sjoin(antGDF_df, antGDF, how='left', on_attribute=['Date']
                         ).rename(columns={'Precipitation_right':'Precipitation', 'Temperature_right': 'Temperature'})
    nonNull_values = joinedDF[~joinedDF['Precipitation'].isnull() & ~joinedDF['Temperature'].isnull()][['Precipitation', 'Temperature']]
    
    # Iter through the rows, and since the data has the same index as the original df, we can fill in the missing values 
    # in the original df with the values from the joinedDF
    filledDF = df.copy()
    for i, row in nonNull_values.iterrows():
        filledDF.at[i, feature] = row[feature]
    
    return filledDF

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
    df[feature] = attach_nearest_value_vectorized(ds, df, var_name)

    # For Antarctic points, fill in the data from the ERA5 Antarctica dataset
    df = fillAntarcticaData(df, feature)

    return df
# End of Atmospheric Data Script

# Teleconnection Indices Script
#############################################
# This part will read in the long format of the teleconnection indices and add them to the dataframe
# 1. Read in the teleconnection indices CSV file
# 2. Attach the indices to the dataframe based on the Year and Month
#############################################

# Open the dataset containing the teleconnection indices
def openTeleInd(dir=os.path.join('Teleconnection_Indices', 'teleconnection_indices.csv')):
    logger.info(f'Opening teleconnection indices from {dir}')
    tele_df = pd.read_csv(dir)
    return tele_df

# Add Teleconnection Indices to the dataframe
def addTeleconnectionData(df):
    logger.info('Adding teleconnection indices to dataframe')

    # Open the teleconnection indices dataset
    teleDF = openTeleInd()

    # Extract the NAO and ENSO indices based on Year and Month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df = df.merge(teleDF, how='left', left_on=['Year', 'Month'], right_on=['Year', 'Month'])
    df.drop(columns=['Year', 'Month'], inplace=True)

    return df
    

# Altitude Data Script 
#############################################
# This part will be used to add the Copernicus 30m DEM altitude data
# 1. Load the pre-downloaded GeoJson file for the given dataset
# 2. For each row in the dataframe, get the altitude value at the corresponding lat and lon
# 3. Add the altitude value to the dataframe
#############################################
# Open the related GeoJson file
def loadAltitudeData(dir='Altitude/data_files'):
    # Get all files in the directory
    altFiles = glob(os.path.join(dir, '*'))
    
    # Load each shapefile into a GeoDataFrame and combine them
    gdfs = []
    for file in altFiles:
        gdf = gpd.read_file(file)
        gdfs.append(gdf)
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    # Return the combined GeoDataFrame with the unique geometry entries
    return combined_gdf.drop_duplicates(subset='geometry').reset_index(drop=True)

# Find an altitude for each point in the dataframe
def findAltitude(point, alt_gdf):
    # Find the altitude for a given point by checking which polygon it intersects in the altitude GeoDataFrame
    matched = alt_gdf[alt_gdf.intersects(point)]

    # If a match is found, return the altitude value
    if not matched.empty:
        return matched['Altitude'].item()
    else:
        return pd.NA  # Return NA if no match is found
    
# Add altitude data to the dataframe
def addAltitudeData(df, dir='Altitude/data_files'):
    logger.info(f'Adding altitude data from {dir}')
    # Load the altitude data
    alt_gdf = loadAltitudeData(dir)
    
    # Apply the findAltitude function to each row in the dataframe
    df['Alt'] = df['geometry'].apply(lambda point: findAltitude(point, alt_gdf))
    
    return df


# Main function that will be called by the script determining which features to add
def addFeatures(df):
    logger.info('Adding features to dataframe')
    #features = ['Altitude', 'Precipitation', 'Temperature', 'Teleconnection']
    features = ['KPN']
    functions = {
        'KPN': addKPN,
        'Precipitation': lambda df: addAtmosData(df, 'Precipitation', r'HydroGFD/data_files/'),
        'Temperature': lambda df: addAtmosData(df, 'Temperature', r'HydroGFD/data_files/'),
        'Altitude': addAltitudeData,
        'Teleconnection': addTeleconnectionData
    }
    
    for feature in features:
        logger.info(f'Adding feature: {feature}')
        df = functions[feature](df)
    return df.drop(columns=['geometry', 'index'], errors='ignore')

if __name__ == "__main__":
    # Determine if you need to append the features or start fresh
    appendTrain = os.path.exists(r'DataTrain.csv')
    appendTest = os.path.exists(r'DataTest.csv')
    appendLoo = os.path.exists(os.path.join('Leave_Out_Points', r'DataLeaveOut.csv'))

    if appendLoo and appendTest and appendTrain: 
        pathTrain = r'DataTrain.csv'
        pathTest = r'DataTest.csv'
        pathLoo = os.path.join('Leave_Out_Points', r'DataLeaveOut.csv')
        logger.info('Appending features to existing datasets')
    else:
        pathTrain = os.path.join('GNIP', 'GNIP_Train.csv')
        pathTest = os.path.join('GNIP', 'GNIP_Test.csv')
        pathLoo = os.path.join('Leave_Out_Points', 'Leave_Out_Points_GNIP (2025-07-22).csv')
        logger.info('Creating new datasets with features')

    # Load training and test datasets
    logger.info('Loading training and test datasets')
    dfTrain = loadDataset(pathTrain)
    dfTest = loadDataset(pathTest)
    # Load the leave-one-out dataset
    dfLoo = loadDataset(pathLoo)
    logger.info('Datasets loaded successfully')

    logger.info('Adding features to datasets')
    # Add the features
    dfTrain = addFeatures(dfTrain)
    dfTest = addFeatures(dfTest)
    dfLoo = addFeatures(dfLoo)
    logger.info('Features added successfully')

    # Save the datasets
    logger.info('Saving datasets to CSV files')
    dfTrain.to_csv(r'DataTrain.csv', index=False)
    dfTest.to_csv(r'DataTest.csv', index=False)
    dfLoo.to_csv(os.path.join('Leave_Out_Points', 'DataLeaveOut.csv'), index=False)