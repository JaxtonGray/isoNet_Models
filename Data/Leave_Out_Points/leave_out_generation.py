# Script to write a JSON file containing the leave out points for isoNet training
# This will contain points specifically pulled from the GNIP data as well as the 
# excess data shared by Trish's colleagues
#%%
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Set Global variable defining the points that are being left out
LEAVE_OUT_POINTS = {
    "lowData" : Point(104.283, 52.3),
    "highData" : Point(7.584, 47.543),
    "aridData" : Point(-2.17, 30.13),
    "northData" : Point(-105.117, 69.1),
    "equitData" : Point(6.72, 0.38),
    "southernData" : Point(-48.06, -22.66),
    "antData" : Point(-68.13, -67.57),
    "lakeWoods" : Point(-93.72, 49.67),
    "tibetPlat" : Point(91.133, 29.7),
    "ethiopiaHigh" : Point(39.77, 12.542),
    "hotWet" : Point(72.82, 18.96)
}

# Function to read in the GNIP data and convert it to a GeoDataFrame
def read_gnip_data(file_path):
    data = pd.read_csv(file_path)
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Lon, data.Lat))
    return gdf

# Function to remove leave out points from the GNIP data based on X and Y coordinates
# Returns: GeoDataFrame with leave out points removed, GeoDataFrame of leave out points found
def remove_leave_out_points(gdf, leave_out_points):
    # Collect matched frames in a list and concat once at the end. This
    # avoids concatenating against an empty/all-NA DataFrame which triggers
    # the FutureWarning about dtype inference.
    matched_list = []

    # Cycle through the points to leave out and find the matching points in the GNIP data
    for point_name, point_geom in leave_out_points.items():
        matched_points = gdf[gdf.intersects(point_geom)].copy()
        if not matched_points.empty:
            matched_points['LeaveOutPoint'] = point_name
            matched_list.append(matched_points)

            # Remove matched points from the original GeoDataFrame
            gdf = gdf[~gdf.intersects(point_geom)]

    # Concatenate all matched points into a single GeoDataFrame
    leave_out_gdf = pd.concat(matched_list, ignore_index=True) if matched_list else gpd.GeoDataFrame(columns=['LeaveOutPoint', *gdf.columns], geometry='geometry')
    return gdf, leave_out_gdf
#%%
if __name__ == "__main__":
    # Read in the GNIP data
    file_date = "2025-07-22"
    gnip_file_path = f"../GNIP/GNIP_Cleaned ({file_date}).csv"
    gnip_gdf = read_gnip_data(gnip_file_path)

    # Remove leave out points from GNIP data
    gnip_gdf_cleaned, leave_out_gdf = remove_leave_out_points(gnip_gdf, LEAVE_OUT_POINTS)
# %%

