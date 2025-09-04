# Script to write a JSON file containing the leave out points for isoNet training
# This will contain points specifically pulled from the GNIP data as well as the 
# excess data shared by Trish's colleagues
#%%
from ctypes.wintypes import POINT
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Set Global variable defining the points that are being left out
LEAVE_OUT_POINTS = {
    "LowData" : {"coords" : Point(104.283, 52.3), "type": "Low density of observations", "source": "GNIP"},
    "HighData" : {"coords" : Point(7.584, 47.543), "type": "High density of observations", "source": "GNIP"},
    "DesertData" : {"coords" : Point(-2.17, 30.13), "type": "Arid region that is sparsely populated, testing for how model performs", "source": "GNIP"},
    "NorthernData" : {"coords" : Point(-105.117, 69.1), "type": "Northern region with unique climatic conditions and typically sparesly populated", "source": "GNIP"},
    "SouthernData" : {"coords" : Point(-105.117, -19.65), "type": "Southern region with distinct climatic conditions and diverse ecosystems", "source": "GNIP"},
    "lakeWoods" : {"coords" : Point(-93.72, 49.67), "type": "Experimental Lakes Area or Lake of the Woods, Insititue of Sustainable Development", "source": "GNIP"},
    "tibetPlat" : {"coords" : Point(91.133, 29.7), "type": "Tibetan Plateau, or as close as I could get it for this", "source": "GNIP"},

}
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
# Function to be used for reading in the Non-Gnip Data Sources
# NOT AVAILABLE YET
#%%