# Script to write a JSON file containing the leave out points for isoNet training
# This will contain points specifically pulled from the GNIP data as well as the 
# excess data shared by Trish's colleagues
#%%
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Import train_test_split for creating training and validation sets
from sklearn.model_selection import train_test_split

# Set Global variable defining the points that are being left out
LEAVE_OUT_POINTS = {
    "lowData" : Point(104.2833333, 52.3),
    "highData" : Point(7.58371693, 47.54260867),
    "aridData" : Point(5.6, 23.27),
    "northData" : Point(-105.117, 69.1),
    "equitData" : Point(6.72, 0.38),
    "southernData" : Point(-48.06, -22.66),
    "antData" : Point(-68.13, -67.57),
    "lakeWoods" : Point(-93.72, 49.67),
    "tibetPlat" : Point(91.133, 29.7),
    "ethiopiaHigh" : Point(39.77, 12.542),
    "hotWet" : Point(72.82, 18.9)
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
    # Initialze an empty geodataframe that has the same columns as the input gdf
    leave_out_gdf = gpd.GeoDataFrame(columns=gdf.columns)

    # Cycle through the the Leave out points and sort out the leaveout points
    for label, point in leave_out_points.items():
        matched = gdf[gdf.geom_equals_exact(point, tolerance=0.01)]
        leave_out_gdf = pd.concat([leave_out_gdf, matched])
    
    # Remove the leave out points from the original gdf and reset the index
    gdf_removed = gdf[~gdf.index.isin(leave_out_gdf.index)].reset_index(drop=True)
    leave_out_gdf = leave_out_gdf.reset_index(drop=True)

    # Map the label of the leave out points to the points in the leave_out_gdf
    leave_out_gdf['Label'] = leave_out_gdf.apply(lambda row: [label for label, point in leave_out_points.items() if row.geometry.equals_exact(point, tolerance=0.01)][0], axis=1)
    return gdf_removed, leave_out_gdf
#%%
if __name__ == "__main__":
    # Read in the GNIP data
    file_date = "2025-07-22"
    gnip_file_path = f"GNIP_Cleaned ({file_date}).csv"
    gnip_gdf = read_gnip_data(gnip_file_path)

    # Remove leave out points from GNIP data
    gnip_gdf_removed, leave_out_gdf = remove_leave_out_points(gnip_gdf, LEAVE_OUT_POINTS)

    # Save the GNIP data with leave out points removed
    gnip_gdf_removed.to_csv(f"GNIP_Data ({file_date}).csv", index=False)

    # Split the remaining GNIP data into training and validation sets
    train_gdf, test_gdf = train_test_split(gnip_gdf_removed, test_size=0.2, random_state=42)
    train_gdf.to_csv(f"GNIP_Train.csv", index=False)
    test_gdf.to_csv(f"GNIP_Test.csv", index=False)

    # Save the leave out points to a csv file
    leave_out_gdf.to_csv(f"../Leave_Out_Points/Leave_Out_Points_GNIP ({file_date}).csv", index=False)
# %%