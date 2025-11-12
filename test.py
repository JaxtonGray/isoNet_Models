# Import libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_input(data):
    # Preprocess data
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
    data['Year'] = data['Date'].dt.year
    data['JulianDay'] = pd.to_datetime(data['Date']).dt.dayofyear
    data['JulianDay_Sin'] = np.sin(2 * np.pi * data['JulianDay'] / 365)

    # Set the input features apart from the target variable
    features = data[['Lat', 'Lon', 'Alt', 'Year', 'JulianDay_Sin']]

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    outputScaler = StandardScaler()
    outputScaler = outputScaler.fit(data[['O18', 'H2']])
    return features, outputScaler


# Load pre-trained model and predict model against features
def model_prediction(model, features, outputScaler):
    # Make predictions
    predictions = model.predict(features)

    # Inverse transform predictions if necessary
    # Assuming the target variable was also scaled during training
    predictions = outputScaler.inverse_transform(predictions)
    return predictions

if __name__ == "__main__":
    # Load test data and leave-one-out data
    data_test = pd.read_csv(r'Data/DataTest.csv')
    data_loo = pd.read_csv(r'Data/Leave_Out_Points/Leave_Out_Points_GNIP (2025-07-22).csv')

    # Process input data
    features_test, outputScaler = process_input(data_test)
    features_loo, _ = process_input(data_loo)

    # Cycle through the list of models
    # Store predictions with O18_Model_Run{i} and H2_Model_Run{i} columns

    for i in range(1, 11):
        # Load the pre-trained model
        model = keras.models.load_model(f'Models/Global_B/Model_Run{i}/Model_Run{i}.keras')

        # Make predictions for test data and leave-one-out data
        predictions_test = model_prediction(model, features_test, outputScaler)
        predictions_loo = model_prediction(model, features_loo, outputScaler)

        # Store predictions in the respective dataframes
        data_test[f'O18_Model_Run{i}'] = predictions_test[:, 0]
        data_test[f'H2_Model_Run{i}'] = predictions_test[:, 1]
        data_loo[f'O18_Model_Run{i}'] = predictions_loo[:, 0]
        data_loo[f'H2_Model_Run{i}'] = predictions_loo[:, 1]

    # Save the results to new CSV files
    data_test.to_csv(r'Test_Preds.csv', index=False)
    data_loo.to_csv(r'Leave_Out_Preds_GNIP.csv', index=False)




