# This script will be used to attach ERA5 data to any values in the Antarctica dataset. It will be used to test the process of attaching ERA5 data to the Antarctica dataset, and to ensure that the process is working correctly.
import os
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np

# This function will read in the total original dataset and find only the Antarctica data points
# Defined here as any data point with a latitude less than or equal to -60 degrees.
def read_antartica_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lon, df.Lat), crs='EPSG:4326')
    gdf_ant = gdf[gdf['Lat'] <= -60].reset_index(drop=True)
    return gdf_ant

# As the ERA5 data uses a longitude range of 0 to 360, we need to convert any longitudes in the 
# Antarctica dataset that are negative (i.e. in the range of -180 to 0) to the corresponding positive 
# values in the range of 180 to 360.
def convert_to_era5_longitude(lon):
    # Convert longitude from -180 to 180 range to 0 to 360 range
    if lon < 0:
        return lon + 360
    else:
        return lon

# This function will take in a latitude, longitude, and time, and will return the corresponding ERA5 data
# for that location and date. It will use the xarray library to query the ERA5 dataset.
def get_era5_data(lat, lon, date, ds, variable=['t2m', 'tp']):
    # Convert longitude to ERA5 format
    lon_era5 = convert_to_era5_longitude(lon)

    # Given that this is an hourly dataset, we will query for the entire day (24 hours) and then take the average over 3h and then over 1D to get the daily average for the given date.
    # Start and end times for the query
    date_start = pd.Timestamp(date).floor('D') # Start of the day
    date_end = date_start + pd.Timedelta(hours=23) # End of the day

    if isinstance(variable, list):
        # Query the ERA5 dataset for the given latitude, longitude, and time
        era5_data = ds[variable].sel(
            latitude=lat,
            longitude=lon_era5,
            method='nearest'
        ).sel(
            valid_time=slice(date_start, date_end)
        ).load().resample(valid_time='3h').mean().resample(valid_time='1D').mean()
    
    # If variable is a list, return a dictionary of variable names and their corresponding values
    if isinstance(variable, list):
        return {var: float(era5_data[var].values.item()) for var in variable}
    else:
        return float(era5_data.values.item())

if __name__ == "__main__":
    # First, we will read in the ERA5 data. This will be done using the earthdata.destine API
    # Which requires an account and an API key. Follow their instructions to set up an account and obtain an API key.
    ds = xr.open_dataset(
            "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
            storage_options={"client_kwargs":{"trust_env":True}},
            chunks={},
            engine="zarr",
        )

    # Read in the Antarctica dataset
    gdf_ant = read_antartica_data(os.path.join('..', 'GNIP', 'GNIP_Data (2025-07-22).csv'))

    # Cycle through each row in the Antarctica dataset and attach the corresponding ERA5 data
    for index, row in gdf_ant.iterrows():
        lat_i = float(row.Lat)
        lon_i = float(row.Lon)
        time_i = row.Date.strftime('%Y-%m-%d')
        
        era5_data = get_era5_data(lat_i, lon_i, time_i, ds)
        
        # Attach the ERA5 data to the Antarctica dataset
        for var, value in era5_data.items():
            gdf_ant.at[index, var] = value
    
    # Save the updated Antarctica dataset with ERA5 data attached
    gdf_ant.to_csv('GNIP_With_ERA5.csv', index=False)