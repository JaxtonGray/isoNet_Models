# Importing the common libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import re
import os

# TensorFlow and Scikit Learn libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

from sklearn.preprocessing import MinMaxScaler

# Create a function that imports the data and returns 6 dataframes for each of the 6 models
def importTrainData(scheme):
    # Importing the data and convert to a GeoDF
    dfTrain = pd.read_csv(r'../../Data/DataTrain.csv')
    dfTrain['Date'] = pd.to_datetime(dfTrain['Date'], utc=True)
    dfTrain = gpd.GeoDataFrame(dfTrain, geometry=gpd.points_from_xy(dfTrain.Lon, dfTrain.Lat, dfTrain.Alt)).set_crs('EPSG:4326')
    cols = dfTrain.columns

    # Remove units from columns for easier processing
    codeCols = list(map(lambda x: re.sub(r'\(([^()]*)\)', '', x).strip(), cols))
    dfTrain.columns = codeCols

    # Load in the PrevailingWinds file for which the model will be split on
    modelLocations = pd.read_csv(f'../../Data/ModelSplit_Schemes/{scheme}.csv')
    modelLocations = gpd.GeoDataFrame(modelLocations, geometry=gpd.GeoSeries.from_wkt(modelLocations['geometry']), crs="EPSG:4326")
    modelLocations.set_index('Region', inplace=True)

    # Create an empty dictionary that will contain each of the modelDatasets to train with
    modelData = {}
    for modelLoc in modelLocations.index:
        # Create a new dataframe for each of the model locations and store it in the dictionary
        modelData[modelLoc] = dfTrain[dfTrain.within(modelLocations.loc[modelLoc].geometry)].copy()
        modelData[modelLoc] = pd.DataFrame(modelData[modelLoc])
        # Drop the geometry column as it is not needed for the model
        modelData[modelLoc].drop(columns='geometry', inplace=True)

    return dfTrain, codeCols, cols, modelData

# Now we will have a function that will setup the data for the model
def dataSetup(modelData, modelName, featureList):
    # Create the features and target variables
    dataset = modelData[modelName]
    targets = dataset[['O18', 'H2']]
    features = dataset.drop(columns=['O18', 'H2'])

    # Extract the year and julian day from the date, convert to sin transformation for julian day
    features['Date'] = pd.to_datetime(features['Date'], utc=True)
    features['Year'] = features['Date'].dt.year
    features['JulianDay'] = features['Date'].dt.dayofyear

    # Create the sin transformation for the julian day
    features['JulianDay_Sin'] = np.sin(2 * np.pi * features['JulianDay'] / 365)
    features.drop(columns=['Date', 'JulianDay'], inplace=True)
    features = features[featureList]
    trainingCols = features.columns

    # Create the Scaler object and fit the features
    scaler = MinMaxScaler()
    xTrain = scaler.fit_transform(features.values)
    yTrain = targets.values

    # Split the training data into training and validation
    xVal, yVal = xTrain[:int(len(xTrain)*0.2)], yTrain[:int(len(yTrain)*0.2)]
    xTrain, yTrain = xTrain[int(len(xTrain)*0.2):], yTrain[int(len(yTrain)*0.2):]

    return xTrain, xVal, yTrain, yVal, scaler, trainingCols


# Create a function that will create the model
def create_model(neurons, lr, numFeatures):
    model = Sequential()
    model.add(InputLayer(shape=(numFeatures,1)))
    model.add(LSTM(neurons))
    model.add(Dense(neurons))
    model.add(Dense(neurons))
    model.add(Dense(2)) # 2 outputs
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
    return model

# This function will train a model
def modelTrain(model, xTrain, yTrain, xVal, yVal, epochs):
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=150, verbose=1)
    model.fit(xTrain, yTrain, epochs=epochs, validation_data=(xVal, yVal), callbacks=[earlyStop], verbose=0)
    return model

# This function will go through the process of training each of the models
def trainAllModels(modelData, featureList):
    # Cycle through each of the models and create an empty dictionary to store the models
    models = {}
    for modelName in modelData.keys():
        # Setup the data for the model
        xTrain, xVal, yTrain, yVal, scaler, trainingCols = dataSetup(modelData, modelName, featureList)
        numFeatures = xTrain.shape[1]

        # Create the model
        model = create_model(64, 0.001, numFeatures)
        model = modelTrain(model, xTrain, yTrain, xVal, yVal, 1000)

        # Store the model in the dictionary
        models[modelName] = model

    return models, list(trainingCols), scaler

# This function will import the test data and setup the data for the model and scale it
def importTestData(cols, codeCols, scaler, modelData, scheme, featureList):
    testData = pd.read_csv(r'../../Data/DataTest.csv')
    testData['Date'] = pd.to_datetime(testData['Date'], utc=True)
    testData = gpd.GeoDataFrame(testData, geometry=gpd.points_from_xy(testData.Lon, testData.Lat, testData.Alt)).set_crs('EPSG:4326')
    testData.columns = codeCols

    # Convert the Date column to year and julian day sin transformation
    testData['Year'] = testData['Date'].dt.year
    testData['JulianDay'] = testData['Date'].dt.dayofyear
    testData['JulianDay_Sin'] = np.sin(2 * np.pi * testData['JulianDay'] / 365)
    testData.drop(columns=['Date', 'JulianDay'], inplace=True)

    # Load in the PrevailingWinds file for which the model will be split on
    modelLocations = pd.read_csv(f'../../Data/ModelSplit_Schemes/{scheme}.csv')
    modelLocations = gpd.GeoDataFrame(modelLocations, geometry=gpd.GeoSeries.from_wkt(modelLocations['geometry']), crs="EPSG:4326")
    modelLocations.set_index('Region', inplace=True)

    # Create an empty dictionary to store the test data for each of the models regions
    testDict = {}
    for region in modelLocations.index:
        # Extract the test data for the region
        testRegion = testData[testData.within(modelLocations.loc[region].geometry)].copy()
        testRegion.drop(columns='geometry', inplace=True)

        # Scale the test data
        yTest = testRegion[['O18', 'H2']].values
        xTest = testRegion[featureList].values
        xTest = scaler.transform(xTest)

        # Store the test data in the dictionary
        testDict[region] = [xTest, yTest, scaler]

    
    return testDict

# This function will predict the values for the test data based on geography
def predictValues(models, testData, trainingCols):
    # Create an empty dictionary to store the predictions
    predictions = {}
    for modelName in models.keys():
        # Extract the test data
        xTest, yTest, scaler = testData[modelName]

        # Predict the values
        yPred = models[modelName].predict(xTest)

        # Inverse the scaling
        xTest = scaler.inverse_transform(xTest)

        # Store the predictions
        predictions[modelName] = [xTest, yPred, yTest]
    
    # Combine the predictions into a single dataframe
    cols = trainingCols + ['O18 A', 'H2 A','O18 P', 'H2 P']
    predDF = pd.DataFrame(columns=cols)
    for modelName in predictions.keys():
        xTest, yPred, yTest = predictions[modelName]
        tempDF = pd.DataFrame(xTest, columns=trainingCols)
        tempDF[cols[-4]] = yTest[:,0]
        tempDF[cols[-3]] = yTest[:,1]
        tempDF[cols[-2]] = yPred[:,0]
        tempDF[cols[-1]] = yPred[:,1]
        predDF = pd.concat([predDF, tempDF], ignore_index=True)

    return predDF

# Export all information to file
def exportData(models, predDF):
    # Create the directory if it does not exist
    if not os.path.exists('Trained_Models'):
        os.makedirs('Trained_Models')
    # Export the models
    for modelName in models.keys():
        models[modelName].save(f'Trained_Models/{modelName}.keras')
    
    # Export the predictions
    predDF.to_csv('Model_4_TestData.csv', index=False)

# Main function to call all the other functions
def main():
    scheme = 'PrevailingWinds_6Split'
    featureList = ['Lat', 'Lon', 'Alt', 'Precip', 'Temp',
                   'KPN_A', 'KPN_B', 'KPN_C', 'KPN_D', 'KPN_E',
                   'Year', 'JulianDay_Sin']
    dfTrain, codeCols, cols, modelData = importTrainData(scheme)
    models, trainingCols, scaler = trainAllModels(modelData, featureList)
    testData = importTestData(cols, codeCols, scaler, modelData, scheme, featureList)
    predictions = predictValues(models, testData, trainingCols)
    exportData(models, predictions)

main()