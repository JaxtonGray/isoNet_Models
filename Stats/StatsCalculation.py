####################################################################################################
#  This will read in a model number and output the stats for that specific model.
# It will calculate the KGE (Kling-Gupta Efficiency), RMSE, and Residuals for the model.
# However, given that some of these models have different regional models that make up the main model,
# it will need to be separated into different regions. But only for the models that do have regional models.
# The stats will be outputted to a csv file. To be used in the future for analysis.
####################################################################################################

import json
from glob import glob
import re
import os
import numpy as np
import pandas as pd
import geopandas as gpd

# Caclulate the RMSE for a predicted and observed dataset
# Assume that the observed and predicted datasets are pandas dataframes with the same length
def rmse(observed, predicted):
    return np.sqrt(np.mean((observed - predicted) ** 2))

# Calculate mean residuals for a predicted and observed dataset
# Assume that the observed and predicted datasets are pandas dataframes with the same length
def meanResidual(observed, predicted):
    return np.mean(predicted) - np.mean(observed)

# Calculate the KGE for a predicted and observed dataset
# Assume that the observed and predicted datasets are pandas dataframes with the same length
def kge(observed, predicted):
    # Calculate mean and standard deviation of observed and predicted datasets
    obs_mean = observed.mean()
    sim_mean = predicted.mean()
    obs_std = observed.std()
    sim_std = predicted.std()

    # Calculate the correlation coefficient between the observed and predicted datasets
    r = np.corrcoef(observed, predicted)[0, 1] # [0, 1] is the correlation between the two datasets in the matrix

    # Calculate the bias and variability between the observed and predicted datasets
    beta = sim_mean / obs_mean
    alpha = sim_std / obs_std

    # Calculate the Kling-Gupta Efficiency
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

# Create a function that will load in the model data and return a geopandas dataframe with the data and
# the model architecture
def readModelData(modelNum):
    # Load in the model data
    modelData = pd.read_csv(f"../Models/Model_{modelNum}/Model_{modelNum}_TestData.csv")
    modelData = gpd.GeoDataFrame(modelData, geometry=gpd.points_from_xy(modelData.Lon, modelData.Lat), crs="EPSG:4326")

    # Change the headers of the columns to be more usable with the code
    # Start by extracting the old headers
    oldHeaders = modelData.columns

    # Now go through and change the headers to be more usable using regex to remove anything within brackets
    # Also if there is something like O18 Actual, it will be changed to O18A and O18P for the predicted
    newHeaders = []
    newHeaders = [re.sub(r"\(.*\)", "", header).strip() for header in oldHeaders]
    newHeaders = [re.sub(r" Actual", "A", header).strip() for header in newHeaders]
    newHeaders = [re.sub(r" Predicted", "P", header).strip() for header in newHeaders]

    # Now change the headers of the dataframe
    modelData.columns = newHeaders
    

    return modelData, list(oldHeaders)

def readModelArch(modelArch_Name):
    # Load in the model architecture which will only need to be done if not "Global"
    if modelArch_Name != "Global":
        modelArch = pd.read_csv(f"../Data/ModelSplit_Arch/{modelArch_Name}.csv")
        modelArch = gpd.GeoDataFrame(modelArch, geometry=gpd.GeoSeries.from_wkt(modelArch.geometry), crs="EPSG:4326")
    else:
        modelArch = None
    
    return modelArch

# Calculate the summary statistics for the model and return a dataframe with the results for each region
# and if there are no regions, return the stats for the entire model.
def summaryStats(modelData, modelArch_name):
    # First check if the model has regional models
    if modelArch_name != "Global":
        # Load in the model architecture
        modelArch = readModelArch(modelArch_name)

        # Cycle through the regions and assign the data from each to dictionary
        regionalData = {}
        key = modelArch.columns[0]
        for null, region in modelArch.iterrows():
            regionalData[region[key]] = modelData[modelData.within(region.geometry)]
        
        # Calculate the stats for each region and store in a dictionary
        regionalStats = pd.DataFrame(columns=["Region", "O18 KGE", "O18 RMSE","O18 Mean Residual", "H2 KGE", "H2 RMSE", "H2 Mean Residual"])
        for region, data in regionalData.items():
            stats = {"Region": region,
                     "O18 KGE": kge(data['O18 A'], data['O18 P']),
                     "O18 RMSE": rmse(data['O18 A'], data['O18 P']),
                     "O18 Mean Residual": meanResidual(data['O18 A'], data['O18 P']),
                     "H2 KGE": kge(data['H2 A'], data['H2 P']),
                     "H2 RMSE": rmse(data['H2 A'], data['H2 P']),
                     "H2 Mean Residual": meanResidual(data['H2 A'], data['H2 P'])}
            statsDF = pd.DataFrame([stats], index=[0])
            regionalStats = pd.concat([regionalStats, statsDF], ignore_index=True)
        
        return regionalStats
    else:
        # Calculate the stats for the entire model
        stats = {"O18 KGE": kge(modelData['O18 A'], modelData['O18 P']),
                 "O18 RMSE": rmse(modelData['O18 A'], modelData['O18 P']),
                 "O18 Mean Residual": meanResidual(modelData['O18 A'], modelData['O18 P']),
                 "H2 KGE": kge(modelData['H2 A'], modelData['H2 P']),
                 "H2 RMSE": rmse(modelData['H2 A'], modelData['H2 P']),
                 "H2 Mean Residual": meanResidual(modelData['H2 A'], modelData['H2 P'])}
        
        return pd.DataFrame([stats])

# This function will cylce through all model architectures and calculate the stats for each model and return
# a dictionary with the results: {modelArch_Name: stats, modelArch_Name2: stats2, ...}
def allArchStats(modelData, mainArch):
    # Use glob to get all the model architecture names
    allArchs = glob("../Data/ModelSplit_Arch/*.csv")
    allArchs = [arch.split("/")[-1].split(".")[0] for arch in allArchs]
    allArchs = [re.split(r"\\|//", arch)[1] for arch in allArchs]
    allArchs.append("Global")
    # Create a dictionary to store the stats for each model architecture
    allArchsStats = {}

    # Cycle through each model architecture and calculate the stats for each model
    for arch in allArchs:
        if arch == mainArch:
            allArchsStats[f"{arch} (main)"] = summaryStats(modelData, arch)
        else:
            allArchsStats[arch] = summaryStats(modelData, arch)

    return allArchsStats

# Export the stats to an excel file where each sheet is a different model architecture
def exportStats(allsStats, modelNum):
    if not os.path.exists("SummaryStats"):
        os.mkdir("SummaryStats")
    with pd.ExcelWriter(f"SummaryStats//Model_{modelNum}_Stats.xlsx") as writer:
        for arch, stats in allsStats.items():
            stats.to_excel(writer, sheet_name=arch, index=False)

# Convert Julian Day Sin to Julian Day
def undoJulianDaySin(values):
    return np.ceil((np.arcsin(values) / np.pi + 0.5) * 365).astype(int)

# Caclulate the residuals and export them to a csv file
def calculateResiduals(modelData, modelNum, oldHeaders):
    # Calculate the residuals for the model
    modelData['O18 Residuals'] = modelData['O18 A'] - modelData['O18 P']
    modelData['H2 Residuals'] = modelData['H2 A'] - modelData['H2 P']

    # Create a dictionary that will rename the columns back to the original names
    renameDict = dict(zip(modelData.columns, oldHeaders))
    modelData.rename(columns=renameDict, inplace=True)
    modelData.drop(columns=['geometry'], inplace=True)

    # Convert the Julian Day sine transform back to the original Julian Date
    modelData['JulianDay'] = undoJulianDaySin(modelData['JulianDaySin'])

    # Combine the Year and Julian Day columns to create a date column
    modelData['Date'] = modelData['Year'].astype(int).astype(str) + "-" + modelData['JulianDay'].astype(str)
    modelData['Date'] = pd.to_datetime(modelData['Date'], format="%Y-%j")
    modelData.drop(columns=['Year', 'JulianDay', 'JulianDaySin'], inplace=True)

    # Move the Date column to the front of the dataframe
    cols = modelData.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    modelData = modelData[cols]
    

    # Export the residuals to a csv file
    if not os.path.exists("Residuals"):
        os.mkdir("Residuals")
    modelData.to_csv(f"Residuals//Model_{modelNum}_Residuals.csv", index=False)

def main():
    # Load in the model directory
    with open(r'../Models/modelDirectory.json', 'r') as file:
        data = json.load(file)
    
    # Cycle through the models and calculate the stats for each model
    for modelValues in data.values():
        modelNum = modelValues['num']
        mainArch = modelValues['arch']

        # Load in the model data
        modelData, oldHeaders = readModelData(modelNum)

        # Calculate the stats for the model and export to an excel file
        statsAll = allArchStats(modelData, mainArch)
        exportStats(statsAll, modelNum)
        calculateResiduals(modelData, modelNum, oldHeaders)


main()


