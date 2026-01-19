import os, sys
import xarray as xr
import pandas as pd
import geopandas as gpd

# Function that takes in a geopandas dataframe and returns a new dataframe of only the unique coordinates
def get_unique_coordinates(gdf):
    unique_geom = gdf.geometry.unique()
    unique_gdf = gpd.GeoDataFrame(geometry=unique_geom, columns=['geometry'], crs=gdf.crs)
    return unique_gdf
df = pd.read_csv(r"../DataTrain.csv")
gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Lon, df.Lat), crs="EPSG:4326"
    )

if __name__ == "__main__":
    # Read in the file given
    df = pd.read_csv(sys.argv[1])
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Lon, df.Lat), crs="EPSG:4326"
    )
    # Extract only the unique coordinates associated with the dataframe
    gdf_unique = get_unique_coordinates(gdf)

    # Retrieve the authentication token from the file in the directory (SHOULD NEVER BE SHARED IN THE GIT)
    with open('auth_token.txt', 'r') as f:
        auth_token = f.read().strip()

    # Open the dataset from EarthData
    ds = xr.open_dataset(
        f"https://edh:{auth_token}@data.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr",
        chunks = None,
        engine="zarr",
        decode_coords="all",
        mask_and_scale=False
    )

    # Vectorize the points in the unique gdf
    xs = xr.DataArray(gdf_unique.geometry.x.values, dim='points')
    ys = xr.DataArray(gdf_unqiue.geometry.y.values, dim='points')

    # Select all the nearest points in the dataset and attach them to the unique dataframe
    gdf_unqie['Altitude'] = (
        ds["dsm"]
        .sel(lon=xs, lat=ys, method="nearest")
        .values
    )

    # Save the file into the appropriate path
    input_name = os.path.split(input)[-1].split('.')[0]
    file_name = f'{file_name}_Alt.geojson'
    gdf_unique.to_file(f'data_files/{file_name}')