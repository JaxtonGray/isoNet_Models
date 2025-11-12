# Import libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv(r'Data/DataTest.csv')

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['JulianDay'] = pd.to_datetime(data['Date']).dt.dayofyear
data['JulianDay_Sin'] = np.sin(2 * np.pi * data['JulianDay'] / 365)

# Set the input features apart from the target variable
features = data[['Lat', 'Lon', 'Alt', 'Year', 'JulianDay_Sin']]

# Load pre-trained model
model = keras.models.load_model(r'Models/Global_B/Model_Run1/Model_Run1.keras')

# Make predictions
predictions = model.predict(features)

# Save predictions to CSV
output = pd.DataFrame({
    'Date': data['Date'],
    'Lat': data['Lat'],
    'Lon': data['Lon'],
    'Alt': data['Alt'],
    'Predicted_Value': predictions.flatten()
})
output.to_csv(r'Predictions.csv', index=False)