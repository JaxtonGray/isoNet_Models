# Importing the correct libraries
import numpy as np
import pandas as pd
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

# Importing the data
def importData():
    dataTrain = pd.read_csv('../../Data/DataTrain.csv')
    cols = dataTrain.columns
    # Remove units from columns for easier processing
    codeCols = list(map(lambda x: re.sub(r'\(([^()]*)\)', '', x).strip(), cols))
    dataTrain.columns = codeCols
    dataTrain['Date'] = pd.to_datetime(dataTrain['Date'], utc=True)
    dataTest = pd.read_csv('../../Data/DataTest.csv')
    dataTest.columns = codeCols
    dataTest['Date'] = pd.to_datetime(dataTest['Date'], utc=True)
    return dataTrain, dataTest, codeCols, cols

def transformDate(data):
    data['Year'] = data['Date'].dt.year
    data['Julian'] = data['Date'].dt.dayofyear
### Sine transformation of the Julian day to account for the cyclical nature of the year
    data['JulianDay_Sin'] = np.sin(2 * np.pi * data['Julian'] / 365)
    data = data.drop(['Date', 'Julian'], axis=1)
    return data

# Separate into targets and features
def separateTrainData(dataTrain, featureList):
    targetsTrain = dataTrain[['O18', 'H2']]
    featuresTrain = dataTrain[featureList]
    return targetsTrain, featuresTrain
    
# Create arrays from the features and targets
def createArrays(features, targets):
    scaler = MinMaxScaler()
    featureArray = features.values
    xTrain = scaler.fit_transform(featureArray)
    yTrain = targets.values
    xVal, yVal = xTrain[:int(len(xTrain)*0.2)], yTrain[:int(len(yTrain)*0.2)]
    xTrain, yTrain = xTrain[int(len(xTrain)*0.2):], yTrain[int(len(yTrain)*0.2):]
    return xTrain, yTrain, xVal, yVal, scaler

# Define the model
def create_model(neurons, lr, featureLength):
    model = Sequential()
    model.add(InputLayer(shape=(featureLength,1)))
    model.add(LSTM(neurons))
    model.add(Dense(neurons))
    model.add(Dense(neurons))
    model.add(Dense(2)) # 2 outputs
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
    return model

# Create and train the model
def trainModel(model, xTrain, yTrain, xVal, yVal):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True) # Early stopping callback
    model.fit(xTrain, yTrain, epochs=1000, batch_size=32, validation_data=(xVal, yVal), callbacks=[es], verbose=1)
    return model

# Predict the test data and export it
def predictTestData(model, dataTest, featureList, scaler, cols):
    xTest = transformDate(dataTest)
    xTest = dataTest[featureList]
    xTest = scaler.transform(xTest)
    yPred = model.predict(xTest)
    dataTest.columns = cols
    dataTest['O18 (‰)'] = yPred[:,0]
    dataTest['H2 (‰)'] = yPred[:,1]
    dataTest.to_csv('Model_1_Test.csv', index=False)

# Save the model
def saveModel(model):
    model.save('model.keras')

# Main function
def main():
    dataTrain, dataTest, codeCols, cols = importData()
    featureList = [
        'Lat', 'Lon', 'Alt', 'Temp', 'Precip',
        'KPN_A', 'KPN_B', 'KPN_C', 'KPN_D', 'KPN_E', 
        'Year', 'JulianDay_Sin']
    dataTrain = transformDate(dataTrain)
    targetsTrain, featuresTrain = separateTrainData(dataTrain, featureList)
    xTrain, yTrain, xVal, yVal, scaler = createArrays(featuresTrain, targetsTrain)
    model = create_model(64, 0.001, len(featureList))
    model = trainModel(model, xTrain, yTrain, xVal, yVal)
    predictTestData(model, dataTest, featureList, scaler, cols)
    saveModel(model)

main()
