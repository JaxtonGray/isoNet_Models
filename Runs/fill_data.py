# This script will be used to fill in data for runs with the different models
# It will require the arguments that are the file path
# Said file will need the following columns:
# - Lat: The latitude of the location (float)
# - Lon: The longitude of the location (float)
# - Date: The date for the run (string in the format YYYY-MM-DD)

# The follwing will occur:
# - The script will read in the data from the file
# - For each row, it will use the latitude, longitude, and date to fill in the missing data for the 
# 1. Temperature (float)
# 2. Precipitation (float)
# 3. KPN (One-hot encoded, A, B, C, D, E)
# 4. Altitude (float)
# 5. Teleconnection Indices (ENSO, NAO) (float)

import os, sys, glob, pathlib, datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import rasterio as rio
from scipy.ndimage import distance_transform_edt

# Climate Data Import
# 1. Open all NetCDF files and combine them into a single xarray dataset
# 2. Function that will find the nearest valid grid point for a given point and time within a specified buffer
# 3. Function to grab nearest grid point values for variable in dataset for all rows via vectorized indexing.

# Read in the temperature and precipitation data from the appropriate files
def read_climate_data(dir_path: str = os.path.join('..', 'Data', 'HydroGFD', 'data_files')) -> xr.Dataset:
    # Grab all the NetCDF files in the directory
    files = glob.glob(os.path.join(dir_path, '*.nc'))

    # Open each file then combine them into a single dataset (assuming they have the same variables and dimensions)
    ds = xr.combine_by_coords([xr.open_dataset(file, engine = 'h5netcdf') for file in files], combine_attrs = 'override')
    return ds

# Function to find the nearest valid grid point for a given point and time within a specified buffer 
# and grid selection distance, if the original point is missing data. 
def find_nearest_valid_grid_xarrayds(ds: xr.Dataset, 
                                     point: gpd.GeoDataFrame.geometry, 
                                     time: pd.Timestamp, var: str, 
                                     buffer=5, grid_select=2) -> float:
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
def attach_nearest_value_vectorized(ds: xr.Dataset, 
                                    df: pd.DataFrame | gpd.GeoDataFrame, 
                                    var: str) -> float:
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
# End of Climate Data Import


# KPN Data Import
# 1. Read in the KPN rasters for the appropriate time periods
# 2. Find the nearest non-zero KPN point for a given raster, within a given grid selection distance.
# 3. Get the climate classification for the dataframe by iterating through it and using the correct raster
# 4. Load the legend and convert the KPN from a number to a string (A, B, C, D, E)
# 5. One-hot encode the KPN values for use in the model.
# 6. Function to add the KPN data to the dataframe by calling the above functions in sequence.

# Read in the KPN rasters
def readKPNRasters(dir=os.path.join('..', 'Data', 'KPN')):
    # Create a dictionary to hold the rasters
    folders = glob.glob(os.path.join(dir, '*')) # Get all the files in the current directory
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

# Create a function to grab the nearest non-0 KPN point for a given raster, within a given grid selection distance. 
def find_nearest_non_zero_kpn(rasterArr, row, col, grid_select=2):
    # Args: 
    # rasterArr: a numpy array representing the KPN raster, where 0 is the non-KPN value
    # row: the row index of the point for which to find the nearest non-zero value
    # col: the column index of the point for which to find the nearest non-zero value
    # grid_select: the distance within which to search for the nearest non-zero point
        # - Default is 2, which means we will only consider points within less than 2 grid cells away from the original point.
    # Returns:
    # The value of the nearest non-zero point in the raster for the given row and column, 
    # within the specified grid selection distance.

    # Create a mask of the non-zero points in the raster
    mask = rasterArr == 0

    # Use distance_transform_edt to calculate the euclidean distance to the nearest non-zero point for each point in the raster
    distance, ids = distance_transform_edt(mask, return_indices=True)

    # Add a condition to only consider points within the specified grid_select distance
    valid_mask = np.where(distance < grid_select, True, False)

    # Combine the original mask with the valid_mask to ensure we only consider valid points within the grid_select distance
    mask = np.logical_and(mask, valid_mask)

    # Fill in the zero values in the raster with the nearest non-zero value using the indices returned by distance_transform_edt
    filled_raster = rasterArr.copy()
    filled_raster[mask] = rasterArr[tuple(ids[:, mask])]

    return filled_raster[row, col].item()

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
        kpnValue = rasters[date][1][row, col].item()

        # If the value is 0, check the nearest non-zero value within a grid selection distance of 2
        if kpnValue == 0:
            kpnValue = find_nearest_non_zero_kpn(rasters[date][1], row, col)
        df.at[i, 'KPN'] = kpnValue
    
    # Replace any remaining zero values with NaN
    df['KPN'] = df['KPN'].replace(0, np.nan)

    # Drop any rows that still have NaN values in the KPN column, as we will not be able to fill those in with the nearest non-zero value
    df = df.dropna(subset=['KPN'])

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

    # Check to make sure that all KPN categories are represented in the one-hot encoding,
    # even if they are not present in the dataset.
    # If a category is not present, add a column for it with all values as 0.
    for cat in ['A', 'B', 'C', 'D', 'E']:
        if f'KPN_{cat}' not in df.columns:
            df[f'KPN_{cat}'] = 0 
    
    return df.reset_index(drop=True)

def addKPN(df, dir=os.path.join('..', 'Data', 'KPN')):
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

# Altitude Data Import
# NOTE: The altitude data is imported through pre-made shape files, make sure that the shape files are made
# before running the script. This is done through the BLANK script.

# Function to read in the data from the file
def read_data(file_path: str) -> gpd.GeoDataFrame:
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        return gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Lon, data.Lat))
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Read in the necessary data
    gdf = read_data(r'Runs/Charlotte_Runs/combined_sites.csv')#read_data(sys.argv[1])
    #ds = read_climate_data(dir_path=os.path.join('Data', 'HydroGFD', 'data_files'))
    #rasters = readKPNRasters(dir=os.path.join('Data', 'KPN'))
    gdf = addKPN(gdf, dir=os.path.join('Data', 'KPN'))
    print(gdf.columns)