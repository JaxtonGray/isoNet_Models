# Script to generate the Model Directory
#%%
import os
import json
import pandas as pd
from itertools import product, combinations

# Open the CSV file
modelGuide = pd.read_csv('modelGuide.csv')

# Separate features and spatial schemes
features = modelGuide[modelGuide['Feature?']]['Abbreviation'].to_list()
spatialSchemes = modelGuide[modelGuide['Scheme?']]['Abbreviation'].to_list()

# Define function that takes in the feature and spatial scheme lists and outputs
# a list of all possible model combinations of varying amounts of features
def generate_model_combinations(features, spatialSchemes):
    # First generate all possible feature combinations of varying amounts of features.
    # This will not include the base set ('B') as it will be added to each after
    other_features = [item for item in features if item != 'B']
    allFeatures = []

    # Generate combinations of other features (0 to all other features)
    for r in range(0, len(other_features) + 1):
        for combo in combinations(other_features, r):
            # Add 'B' to each combination
            allFeatures.append(('B',) + combo)

    # Finally combine all feature combinations with all spatial schemes
    return list(product(allFeatures, spatialSchemes))

# Function that will take a list of features abbreviations and return the full names of the list
def get_full_feature_names(feature_abbr_list):
    return ', '.join(modelGuide[modelGuide['Abbreviation'].isin(feature_abbr_list)]['Name'].to_list())

# Make model name using list features and scheme name
def model_name_gen(featureList, scheme):
    # Model name will look like this:
    # Desert_BTPI(ENA)

    # If the feature list contains I(E)\I(N)\I(A) combine paraentheses into one (See above example)
    tele_features = [x[2] for x in featureList if x.startswith('I(')]

    # Join all features together that aren't the teleconnections
    other_features = [x for x in featureList if not x.startswith('I(')]

    # Make a string of the features but only if len(tele_features ) != 0
    featureString = ''.join(other_features)
    if tele_features:
        featureString += f'I({"".join(tele_features)})'
    
    return f"{scheme}_{featureString}"

# Function that takes in all the model generation and creates a DataFrame of the scheme, and features
# It will look at the model guide to create a list of all the features, in their full name form
def create_model_dir(allModels):
    # Generate the dataframe from the model guide
    model_dir = pd.DataFrame(allModels, columns=['Features (Abbreviation)', 'Spatial Scheme'])

    # Add the full list of features to the DataFrame (not list object)
    model_dir['Features'] = model_dir['Features (Abbreviation)'].apply(get_full_feature_names)

    # Create a column for the model names
    model_dir['Model Name'] = model_dir.apply(lambda row: model_name_gen(row['Features (Abbreviation)'], row['Spatial Scheme']), axis=1)

    return model_dir

#%%
def save_model_dir(model_dir):
    # Save the model directory DataFrame to a json file which will be structured in the following manner
    # { 'model_name': {
    #     "Spatial Scheme": "...",
    #     "Features": "...",
    #   }
    # }
    model_dict = {
        row['Model Name']: {
            "Spatial Scheme": row['Spatial Scheme'],
            "Features": row['Features']
        }
        for index, row in model_dir.iterrows()
    }

    with open('model_directory.json', 'w') as json_file:
        json.dump(model_dict, json_file, indent=4)


#%%
if __name__ == "__main__":
    # Generate all model combinations
    allModels = generate_model_combinations(features, spatialSchemes)

    # Create the model directory DataFrame
    model_directory = create_model_dir(allModels)

    # Save the model directory to a JSON file
    save_model_dir(model_directory)
# %%
