# Script to write a JSON file containing the leave out points for isoNet training
# This will contain points specifically pulled from the GNIP data as well as the 
# excess data shared by Trish's colleagues
#%%
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#%%
# Function to read in the GNIP data and convert it to a GeoDataFrame
def read_gnip_data(file_path):
    data = pd.read_csv(file_path)
    
    # Set the columns to keep
    columns_to_keep = [
        'Latitude',
        'Longitude',
        'Altitude',
        'Sample Date',
        'Measurand Symbol',
        'Measurand Amount'
    ]
    # Select only the necessary columns
    df = data[columns_to_keep]

    # Rename columns for consistency and clarity
    df = df.rename(columns={
    'Latitude': 'Lat',
    'Longitude': 'Lon',
    'Altitude': 'Alt',
    'Sample Date': 'Date',
    'Measurand Symbol': 'Symbol',
    'Measurand Amount': 'Amount'
    })

    # Set the typing of each column
    # Convert Date Column to Datetime
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Set the typing of the other columns
    df['Lat'] = df['Lat'].astype(float)
    df['Lon'] = df['Lon'].astype(float)
    df['Alt'] = df['Alt'].astype(float)
    df['Symbol'] = df['Symbol'].astype(str)
    df['Amount'] = df['Amount'].astype(float)

    # Pivot the table to be long format
    df = df.pivot_table(index=['Lat', 'Lon', 'Alt', 'Date'],
                        columns='Symbol', 
                        values='Amount', 
                        aggfunc='first').reset_index()

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lon, df.Lat))
    return gdf
# %%
# Function to be used for reading in the excess data
# NOT AVAILABLE YET
#%%
# Function to set the rough coordinates for the leave out points
def set_leave_out_points():
    # Define the rough coordinates for the leave out points
    leave_out_coords = {
        "Point_1": {"Lat": 34.0522, "Lon": -118.2437},  # Los Angeles, CA
        "Point_2": {"Lat": 40.7128, "Lon": -74.0060},   # New York, NY
        "Point_3": {"Lat": 41.8781, "Lon": -87.6298},   # Chicago, IL
        "Point_4": {"Lat": 29.7604, "Lon": -95.3698},   # Houston, TX
        "Point_5": {"Lat": 33.4484, "Lon": -112.0740}   # Phoenix, AZ
    }
    return leave_out_coords