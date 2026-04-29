# This script grabs all the model results from the models directory and loads them into a single csv

# Import Required Libraries
import os, glob, json
import numpy as np
import pandas as pd
import geopandas as gpd

# Function that sorts modelName features, so I can make sure I have no repeats and 
# ensure consistency in model naming scheme
def sort_features(modelName):
    scheme, feats = modelName.split('_')
    featsSorted = ''.join(sorted(feats))
    return f'{scheme}_{featsSorted}'

def grab_result_paths(allModelPaths):
    # Arguments:
    #   - allModelPaths: List containing the path directories of all model types to be loaded in
    # Returns: Dictionary with each model type as key, and an item that is a dictionary, with two keys 
    #   containg lists of TestResults and LeaveOutPoint paths respectively
    
    # Create an empty dictionary to store model types into and file paths
    models = {}

    # Cycle through all models and grab the different file paths of test results and leave_one_out files
    for m in allModelPaths:
        modelType = os.path.split(m)[-1]
        modelName = sort_features(modelType)

        # Grab all test results and leaveOut points
        testResults = list(filter(lambda x: 'LeaveOut' not in x, glob.glob(os.path.join(m, '*', '*', 'Model_*_TestData.csv'))))
        looResults = glob.glob(os.path.join(m, '*', '*', 'Model_*_LeaveOut_TestData.csv'))

        # Add into the final dictionary
        models[modelName] = {'TestResults': testResults, 'LeaveOutResults': looResults}
    return models

# Function that will read a model path into a dataframe and then add the scheme, features, and run number
def read_results(modelName, path):
    # Arguments:
    #   - modelName: A string contiaing the model name ('Global_B', etc.)
    #   - path: string containg path to csv to be read in
    # Returns: A dataframe read in, with the scheme, features, and run number added in

    # Split name into scheme and features
    scheme, feats = modelName.split('_')

    # Grab the run number from the path
    runNum = int(os.path.split(path)[-1].split('_')[1][3:])

    # Read the csv file from the path
    df = pd.read_csv(path)

    # Add in the extra columns required
    df[['Scheme', 'Features', 'RunNum']] = scheme, feats, runNum

    return df

# Combine all results into either test results, or leave out results
def combine_results(resultType, allModels):
    # Arguments:
    #   - resultType: The type of result that is being read in (i.e. 'TestResults', 'LeaveOutResults')
    #   - allModels: Dicitonary containg the results paths, in the format of the dictionary that is returned from grab_result_paths described above
    # Returns: One dataframe that has columns describing the specific models

    # Make a list of dataframes that are being read in
    dfs = []

    # Cycle through the dictionary reading each csv and adding them to the list of dataframes
    for modelName, results in allModels.items():
        for run in results[resultType]:
            dfs.append(read_results(modelName, run))
    
    return pd.concat(dfs, ignore_index=True)

# I need to get the Date from the Sine of the day of year, so I will need to convert the Sine of the day of year back to the day of year, and then convert that to a date
# JulianDay_Sin = np.sin(2 * np.pi * JulianDay / 365)
# To convert the Sine of the day of year back to the day of year, I will need to use the arcsine function, and then convert that to a date
# JulianDay = (np.arcsin(JulianDay_Sin) * 365) / (2 * np.pi) # This will only give a principal angle that will have two solutions
# To get the correct, compare to data in the test dataset, and get the date that is closest to the date in the test dataset, will have to compare the O18 A and H2 A to get the correct date, 
def get_correct_date(row, ogData):
    # Arguments: 
    #   - row: The row of the dataframe that contains the Sine of the day of year, the O18 A and H2 A values
    #   - ogData: The original test dataset that contains the Date, O18 A and H2 A values
    # Returns:
    #   - The correct date that corresponds to the Sine of the day of year, O18 A and H2 A values in the row

    # Get the Sine of the day of year and isotope values from the row.
    # The test results use `O18 A` / `H2 A`, while the leave-out results use `O18` / `H2`.
    julian_day_sin = row['JulianDay_Sin']
    lat = row['Lat']
    lon = row['Lon']
    year = int(round(row['Year']))
    o18_a = row.get('O18 A', row.get('O18'))
    h2_a = row.get('H2 A', row.get('H2'))

    # Get the possible Julian Day values from the Sine of the day of year.
    # `arcsin` only returns the principal angle, so we keep both sine-compatible solutions.
    possible_julian_days = [
        int((np.arcsin(julian_day_sin) * 365) / (2 * np.pi)),
        int(((np.pi - np.arcsin(julian_day_sin)) * 365) / (2 * np.pi)),
    ]

    # Look up all rows with the same lat, lon, and year in the original test dataset.
    matching_rows = ogData[(ogData['Lat'] == lat) & (ogData['Lon'] == lon) & (ogData['Year'] == year)]

    date = None

    # If there is only one matching row, return that corresponding date.
    if len(matching_rows) == 1:
        date = matching_rows['Date'].iloc[0]
    # If there are multiple matching rows, check which one has a Julian Day that matches one of the possible Julian Day values
    elif len(matching_rows) > 1:
        # Rounding errors may cause the possible Julian Day values to be off by 1, so we check for a match within a range of +/- 1 day.
        for _, matching_row in matching_rows.iterrows():
            if np.isclose(matching_row['JulianDay'], possible_julian_days, atol=1).any():
                date = matching_row['Date']

    if date is None:
        # If no matching date is found, we will return the date corresponding with the matcthing row with the same O18 A and H2 A
        for _, matching_row in matching_rows.iterrows():
            if matching_row['O18'] == o18_a and matching_row['H2'] == h2_a:
                date = matching_row['Date']

    return date

def add_leaveOutPoint_names(results, pointsGDF):
    # Arguments:
    #   - results: A dataframe containing the leave out results, with columns for lat and long coordinates
    #   - pointsGDF: A geodataframe containing the point names and their corresponding lat and long coordinates, with a geometry column
    # Returns: The results dataframe with an extra column added in for the point names

    # Convert the results dataframe to a geodataframe, using the lat and long columns to create a geometry column
    resultsGDF = gpd.GeoDataFrame(results, geometry=gpd.points_from_xy(results.Lon, results.Lat), crs='EPSG:4326')

    # Due to the lats and lons not being exactly the same, use the  geom_equals_exact method to add label names
    grab_label = lambda row: [label for label, point in pointsGDF.geometry.items() if row.geometry.equals_exact(point, tolerance=0.1)]
    
    # Apply the function to each row of the results geodataframe to grab the corresponding label for each point
    resultsGDF['Label'] = resultsGDF.apply(grab_label, axis=1)

    # Add the label column as a string instead of a list, since there should only be one label for each point
    resultsGDF['Label'] = resultsGDF['Label'].apply(lambda x: x[0] if len(x) > 0 else None)

    # Convert back to a regular dataframe by dropping the geometry column
    resultsDF = resultsGDF.drop(columns='geometry')
    return resultsDF 

if __name__ == '__main__':
    # Load in all test files and leave_one_out result files as strings
    allModelTypes = glob.glob(os.path.join('..', 'Models', '*'))

    # Grab dictionary contain paths of all model results and directories
    models = grab_result_paths(allModelTypes)

    # Combine all the results into one dataframe for test results and one dataframe for leave out results
    testResults = combine_results('TestResults', models)
    leaveOutResults = combine_results('LeaveOutResults', models)

    # The leave out results will need an extra column added in to specify the point that was left out, so I will add that in here
    # Load the json file that contains the point names and their corresponding lat and long coordinates
    with open(os.path.join('..', 'Data', 'Leave_Out_Points', 'leave_out_points.json')) as f:
        leaveOutPointsDF = pd.DataFrame.from_dict(json.load(f), orient='index')
        # Convert the leave out points dataframe to a geodataframe, using the lat and long columns to create a geometry column
        leaveOutPointsGDF = gpd.GeoDataFrame(leaveOutPointsDF, geometry=gpd.points_from_xy(leaveOutPointsDF.LON, leaveOutPointsDF.LAT), crs='EPSG:4326')
    
    # Add in the point names to the leave out results dataframe
    leaveOutResults = add_leaveOutPoint_names(leaveOutResults, leaveOutPointsGDF)

    # Add in the correct date to the test results dataframe
    # Load in the original test dataset to be used for comparison
    testDataset = pd.read_csv(os.path.join('..', 'Data', 'DataTest.csv'))

    # Convert the Date column to a datetime object, and add in a Julian Day column and a Year column
    testDataset['Date'] = pd.to_datetime(testDataset['Date'])
    testDataset['JulianDay'] = testDataset['Date'].dt.dayofyear
    testDataset['Year'] = testDataset['Date'].dt.year

    # Apply the function to each row of the test results dataframe to get the correct date
    testResults['Date'] = testResults.apply(lambda row: get_correct_date(row, testDataset), axis=1)
    leaveOutResults['Date'] = leaveOutResults.apply(lambda row: get_correct_date(row, testDataset), axis=1)

    # Save the combined results as csv files
    testResults.to_csv('Combined_Test_Results.csv', index=False)
    leaveOutResults.to_csv('Combined_Leave_Out_Results.csv', index=False)