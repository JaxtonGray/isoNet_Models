# Importing the correct libraries
import numpy as np
import pandas as pd

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
dataTrain = pd.read_csv('../../Data/DataTrain.csv')
dataTest = pd.read_csv('../../Data/DataTest.csv')
featureList = ['Lat', 'Lon', 'Date', 'Alt', 'Precip (mm)', 'Temp (°C)']

# Separate into targets and features
targetsTrain = dataTrain[['O18 (‰)', 'H2 (‰)']]
featuresTrain = dataTrain[featureList]

# Change the features date column to year and julian day (sine transformed)
featuresTrain['Date'] = pd.to_datetime(featuresTrain['Date'], utc=True)
featuresTrain['Year'] = featuresTrain['Date'].dt.year
featuresTrain['Julian'] = featuresTrain['Date'].dt.dayofyear

# Sine transformation of the Julian day to account for the cyclical nature of the year
featuresTrain['JulianDaySin'] = np.sin(2 * np.pi * featuresTrain['Julian'] / 365)
featuresTrain = featuresTrain.drop(['Date', 'Julian'], axis=1)

# Prep the min-max scaler
scaler = MinMaxScaler()

# Create arrays for the features and targets
featureArray = featuresTrain.values
xTrain = scaler.fit_transform(featureArray)
yTrain = targetsTrain.values

# Split the training data into training and validation
xVal, yVal = xTrain[:int(len(xTrain)*0.2)], yTrain[:int(len(yTrain)*0.2)]
xTrain, yTrain = xTrain[int(len(xTrain)*0.2):], yTrain[int(len(yTrain)*0.2):]

# Define the model
def create_model(neurons, lr):
    model = Sequential()
    model.add(InputLayer(input_shape=(xTrain.shape[1],1)))
    model.add(LSTM(neurons))
    model.add(Dense(neurons))
    model.add(Dense(neurons))
    model.add(Dense(2)) # 2 outputs
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
    return model

# Create and train the model
model = create_model(64, 0.001)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True) # Early stopping callback
model.fit(xTrain, yTrain, epochs=1000, batch_size=32, validation_data=(xVal, yVal), callbacks=[es], verbose=0)


# Export model testing data
xTest = dataTest[featureList]
xTest['Date'] = pd.to_datetime(xTest['Date'], utc=True)
xTest['Year'] = xTest['Date'].dt.year
xTest['Julian'] = xTest['Date'].dt.dayofyear
xTest['JulianDaySin'] = np.sin(2 * np.pi * xTest['Julian'] / 365)
xTest = xTest.drop(['Date', 'Julian'], axis=1)
xTest = scaler.transform(xTest)

# Predict the test data
yPred = model.predict(xTest)
dataTest['O18 (‰) Predicted'] = yPred[:,0]
dataTest['H2 (‰) Predicted'] = yPred[:,1]
dataTest.to_csv('Model_1_TestData.csv', index=False)

# Save the model
model.save('model.keras')