### This script will open a Summary Stat file and the stats, then export the stats to a geojson file.
### It must be noted that to save space on github the stats files are not included in the repo.
### To get them just run the StatsCalc.py script in this same folder. This script will create the stats files.

### Additionally this script will be needed to run each time for a different device
### The reason is to save storage space on github, and thus the geojson files are ignored in the repo.

# Import libraries
import os
import re
import json
from glob import glob
import pandas as pd
import geopandas as gpd

# This function will load in a specified summary stat file (.xlsx), it will do this by loading in each model schemeitecture
# and using that as its basis for the sheet names to load in the data.
# returns a dictionary with the keys being the model schemeitecture and the values being the dataframes

def loadStats(modelNum):
    # Read in the names of the types of Model Schemeitectures
    schemes = glob('..//Data//ModelSplit_Schemes//*')
    schemes = [re.split(r'\\|//', x)[-1] for x in schemes]
    schemes = [x.split('.')[0] for x in schemes]
    # Since Global is not a model schemeitecture file we will add it to the list
    schemes.append('Global')

    # Load in the modelDirectory which will be used to load information
    # about the model
    with open('..//Models//modelDirectory.json', 'r') as file:
        modelDirectory = json.load(file)
        mainScheme = modelDirectory[f'model_{modelNum}']['scheme']

    # Load in the data
    stats = {}
    for scheme in schemes:
        # To load in the data I will need to check if the scheme is the main one
        # If it is I will need to attach (main) to the end of the sheet name
        if scheme == mainScheme:
            stats[scheme] = pd.read_excel(f'..//Stats//SummaryStats//Model_{modelNum}_Stats.xlsx', sheet_name=f'{scheme} (main)')
            
        else:
            stats[scheme] = pd.read_excel(f'..//Stats//SummaryStats//Model_{modelNum}_Stats.xlsx', sheet_name=scheme)

    return stats

# This function will load a model schemeitecture and return the data in a geodataframe
def loadScheme(schemeName):
    # Check to make sure the schemeName is not Global
    if schemeName == 'Global':
        return None

    # Load the data
    scheme = pd.read_csv(f'..//Data//ModelSplit_Schemes//{schemeName}.csv')
    # Convert the data to a geodataframe
    scheme = gpd.GeoDataFrame(scheme, geometry=gpd.GeoSeries.from_wkt(scheme['geometry']))

    return scheme

loadStats(1)['PrevailingWinds_6Split']

# This function will take a stat dataframe and a model schemeitecture dataframe and join them
# together, it will then return the data as a geodataframe
def joinStats(statData, scheme):
    # Check to make sure the scheme is not None
    if scheme is None:
        # Merge the data with a bounding box enclosing the entire world
        statData['Region'] = 'Global'
        statData['geometry'] = 'POLYGON ((-180 -90, 180 -90, 180 90, -180 90, -180 -90))'
        data = gpd.GeoDataFrame(statData, geometry=gpd.GeoSeries.from_wkt(statData['geometry']), crs='EPSG:4326')
    else:
        # Merge the data on the Region column
        data = pd.merge(scheme, statData, on='Region', how='left')
        data = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')

    return data

def openModelDirectory():
    # Load in the modelDirectory which will be used to load information
    # about the model
    with open('..//Models//modelDirectory.json', 'r') as file:
        modelDirectory = json.load(file)

    return modelDirectory

def main():
    # Load in the model directory
    modelDirectory = openModelDirectory()

    # Cycle through each model and export the data
    for model in modelDirectory.values():
        # Load in the stats
        stats = loadStats(model['num'])
        
        # Cycle through each model schemeitecture and export the data
        for schemeName in stats.keys():
            # Load in the model schemeitecture
            scheme = loadScheme(schemeName)
            # Join the data
            data = joinStats(stats[schemeName], scheme)
            # Export the data
            if os.path.exists('Mapping') == False:
                os.mkdir('Mapping')
            
            data.to_file(f'Mapping//Model_{model["num"]}_{schemeName}.geojson', driver='GeoJSON')


main()