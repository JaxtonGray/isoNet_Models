import os, sys
import dask.array as da
import xarray as xr
import pandas as pd
import geopandas as gpd

# Function that takes in a geopandas dataframe and returns a new dataframe of only the unique coordinates
def get_unique_coordinates(gdf):
    unique_geom = gdf.geometry.unique()
    unique_gdf = gpd.GeoDataFrame(geometry=unique_geom, columns=['geometry'], crs=gdf.crs)
    return unique_gdf

# Function that uses the coordinates to extract elevation data from the dataset, if Dask arrays are used, this will be lazy loaded
def extract_elevation_data(x, y, dataset, var='dsm'):
    # Check to see if dataset variable is a Dask array
    if isinstance(dataset[var].data, da.Array):
        return dataset.sel(lon=x, lat=y, method="nearest")[var].compute().item()
    else:
        return dataset.sel(lon=x, lat=y, method="nearest")[var].item()

if __name__ == "__main__":
    # Read in the file that was given though the passed argument in the command line
    # Only unique geometries will be used to reduce the number of lookups
    df = pd.read_csv(sys.argv[1])
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Lon, df.Lat), crs="EPSG:4326"
    )
    unique_gdf = get_unique_coordinates(gdf)

    # Read in the authentication token from a file
    with open('auth_token.txt', 'r') as file:
        auth_token = file.read().strip()

    # Open the dataset with the authentication token
    ds = xr.open_dataset(
        f"https://edh:{auth_token}@data.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr",
        chunks = None,
        engine="zarr",
        decode_coords="all",
        mask_and_scale=False
    )
    
    # Extract elevation data for each unique coordinate
    unique_gdf['elevation'] = unique_gdf.geometry.apply(lambda point: extract_elevation_data(point.x, point.y, ds))

    # Extract the name of the input file without the extension and path
    file_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # Save the unique_gdf to a new file
    unique_gdf.to_file(f"{file_name}_elevation.geojson", driver="GeoJSON")