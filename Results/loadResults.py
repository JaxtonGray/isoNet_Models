# This script grabs all the model results from the models directory and loads them into a single csv

# Import Required Libraries
#%%
import os, sys, glob
from itertools import chain
import pandas as pd
#%%
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
    for modelName, results in allModels:
        for run in results[resultType]:
            dfs.append(read_results(modelName, run))
    
    return pd.concat(dfs, ignore_index=True)

# %%
if __name__ == '__main__':
    # Load in all test files and leave_one_out result files as strings
    allModelTypes = glob.glob(os.path.join('..', 'Models', '*'))

    # Grab dictionary contain paths of all model results and directories
    models = grab_result_paths(allModelTypes)
# %%
