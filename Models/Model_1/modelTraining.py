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
dataset = pd.read_csv('Data/GNIP/GNIP_Cleaned.csv')

# Separate into targets and features
targets = dataset[['O18 (‰)', 'H2 (‰)']]
features = dataset.drop(['O18 (‰)', 'H2 (‰)'], axis=1)

# Change the features date column to year and julian day (sine transformed)
features['Date'] = pd.to_datetime(features['Date'], utc=True)
features['Year'] = features['Date'].dt.year
features['Julian'] = features['Date'].dt.dayofyear

# Sine transformation of the Julian day to account for the cyclical nature of the year
features['JulianDaySin'] = np.sin(2 * np.pi * features['Julian'] / 365)
features = features.drop(['Date', 'Julian'], axis=1)

# Prep the min-max scaler
scaler = MinMaxScaler()

# Create arrays for the features and targets
featureArray = features.values
x = scaler.fit_transform(featureArray)
y = targets.values

# Split the data into training and testing
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

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
testData = pd.DataFrame(xTest, columns=features.columns)
testData['O18 (‰) Actual'] = yTest[:,0]
testData['H2 (‰) Actual'] = yTest[:,1]
testData['O18 (‰) Predicted'] = model.predict(xTest)[:,0]
testData['H2 (‰) Predicted'] = model.predict(xTest)[:,1]
testData.to_csv('Model_1_TestData.csv', index=False)

# Save the model
model.save('model.keras')