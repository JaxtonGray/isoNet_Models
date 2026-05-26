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

import os, sys, glob
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import rasterio as rio
from scipy.ndimage import distance_transform_edt

# Read in the temperature and precipitation data from the appropriate files
def read_climate_data(file_path: str = os.path.join('..', 'Data', 'HydroGFD', 'data_files')) -> xr.Dataset:
    # Grab all the NetCDF files in the directory
    files = glob.glob(os.path.join(file_path, '*.nc'))

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

# Function to read in the data from the file
def read_data(file_path: str) -> gpd.GeoDataFrame:
    try:
        data = pd.read_csv(file_path)
        return gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Lon, data.Lat))
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    gdf = read_data(sys.argv[1])
    ds = read_climate_data()
    print(ds)