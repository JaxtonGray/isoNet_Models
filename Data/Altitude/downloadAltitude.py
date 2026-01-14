import requests, sys
from rasterio.io import MemoryFile
import pandas as pd
import geopandas as gpd

# First function will retrieve the DEM data and save it as a GeoTIFF file
def request_dem(api_key, left, bottom, right, top):
    # Define the API endpoint and parameters
    baseURL = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "COP30",
        "south": bottom,
        "north": top,
        "west": left,
        "east": right,
        "outputFormat": "GTiff",
        "API_Key": api_key # Replace with your actual API key ideally not sent via github (that's how you get scammed)
    }

    # Make the request to download the DEM data
    response = requests.get(baseURL, params=params)

    if response.status_code != 200:
        raise Exception(f"Error fetching DEM data: {response.status_code}, {response.text}")
    
    return response.content

# Function that will take a point and return bounding box coordinates
def get_bounding_box(point, buffer_distance):
    buffered = point.buffer(buffer_distance)
    bounds = buffered.bounds
    return bounds  # returns (minx, miny, maxx, maxy)

# Function that retreives altitude from response content
def dem_data_from_response(response_content):
    with MemoryFile(response_content) as memfile:
        with memfile.open() as dataset:
            dem_data = dataset.read(1)
            r = dataset
    return dem_data, r

# Return the elevation data from the DEM request
def get_elevation_data(dem_array, raster, x, y):
    return dem_array[raster.index(x, y)]

# Function that combines all steps to get elevation for a point with buffer
def fetch_elevation_for_point(api_key, point, buffer_distance):
    bounds = get_bounding_box(point, buffer_distance)
    dem_response = request_dem(api_key, *bounds)
    dem_array, raster = dem_data_from_response(dem_response)
    elevation = get_elevation_data(dem_array, raster, point.x, point.y)
    return float(elevation)

# Load the DataFrame containing information
df_file = pd.read_csv(sys.argv[1])
df_geoFile = gpd.GeoDataFrame(df_file, geometry=gpd.points_from_xy(df_file.Lon, df_file.Lat), crs='EPSG:4326')

# Extract unique geometries to avoid redundant API calls and reduce costs
unique_geom = df_geoFile.geometry.unique()
gdf = gpd.GeoDataFrame(unique_geom, columns=['geometry'], geometry='geometry')

# Read in the API key from a text file
with open('opentopography_api_key.txt', 'r') as file:
    api_key = file.read().strip()

# Apply the function to get elevation for each point
gdf['Elevation'] = gdf['geometry'].apply(lambda point: fetch_elevation_for_point(api_key, point, buffer_distance=0.01))

# Save the Elevation data with the for that specific file into a new CSV
# Extract the base name without extension from the input file path
base_name = sys.argv[1].split('/')[-1].split('.')[0]
output_csv = f"{base_name}_with_elevation.csv"

# Save only the unique geometry columns along with Elevation
gdf.to_csv(output_csv, index=False)