# %%
# Importing the common libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import re

# TensorFlow and Scikit Learn libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# %%
# Create a function that imports the data and returns 6 dataframes for each of the 6 models
def importData():
    # Importing the data and convert to a GeoDF
    df = pd.read_csv(r'../../Data/GNIP/GNIP_Cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lon, df.Lat, df.Alt)).set_crs('EPSG:4326')
    cols = df.columns

    # Remove units from columns for easier processing
    codeCols = list(map(lambda x: re.sub(r'\(([^()]*)\)', '', x).strip(), cols))
    df.columns = codeCols

    # Load in the PrevailingWinds file for which the model will be split on
    modelLocations = pd.read_csv(r'../../Data/ModelSplit_Arch/PrevailingWinds_6Split.csv')
    modelLocations = gpd.GeoDataFrame(modelLocations, geometry=gpd.GeoSeries.from_wkt(modelLocations['Geometry']), crs="EPSG:4326")
    modelLocations.drop(columns="Geometry", inplace=True)
    modelLocations.set_index('Prevailing Wind', inplace=True)

    # Create an empty dictionary that will contain each of the modelDatasets to train with
    modelData = {}
    for modelLoc in modelLocations.index:
        # Create a new dataframe for each of the model locations and store it in the dictionary
        modelData[modelLoc] = df[df.within(modelLocations.loc[modelLoc].geometry)].copy()
        modelData[modelLoc] = pd.DataFrame(modelData[modelLoc])
        # Drop the geometry column as it is not needed for the model
        modelData[modelLoc].drop(columns='geometry', inplace=True)

    return df, cols, modelData

# %%
# Now we will have a function that will setup the data for the model
def dataSetup(modelData, modelName):
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

    # Create the Scaler object and fit the features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features.values)
    y = targets.values

    # Split the data into training and testing
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the training data into training and validation
    xVal, yVal = xTrain[:int(len(xTrain)*0.2)], yTrain[:int(len(yTrain)*0.2)]
    xTrain, yTrain = xTrain[int(len(xTrain)*0.2):], yTrain[int(len(yTrain)*0.2):]

    return xTrain, xVal, xTest, yTrain, yVal, yTest, scaler


# %%
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

# %%
# This function will train a model
def modelTrain(model, xTrain, yTrain, xVal, yVal, epochs):
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    model.fit(xTrain, yTrain, epochs=epochs, validation_data=(xVal, yVal), callbacks=[earlyStop], verbose=0)
    return model

# %%
# This function will go through the process of training each of the models
def trainAllModels(modelData):
    # Cycle through each of the models and create an empty dictionary to store the models
    models = {}
    testData = {}
    for modelName in modelData.keys():
        # Setup the data for the model
        xTrain, xVal, xTest, yTrain, yVal, yTest, scaler = dataSetup(modelData, modelName)
        numFeatures = xTrain.shape[1]

        # Create the model
        model = create_model(64, 0.001, numFeatures)
        model = modelTrain(model, xTrain, yTrain, xVal, yVal, 500)

        # Store the model in the dictionary
        models[modelName] = model

        # Store the test data in the dictionary
        testData[modelName] = [xTest, yTest, scaler]

    return models, testData

# %%
# This function will predict the values for the test data based on geography
def predictValues(models, testData, cols):
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
    predDF = pd.DataFrame()
    for modelName in predictions.keys():
        xTest, yPred, yTest = predictions[modelName]

        # Create a dataframe for the predictions
    return predictions

# %%
# Main function to call all the other functions
def main():
    # Importing the data
    df, cols, modelData = importData()
    models, testData = trainAllModels(modelData)
    predictions = predictValues(models, testData, cols)
    return df

# %%
main()


