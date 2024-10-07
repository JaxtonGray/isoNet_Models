### Model 1 ####
SCHEME = 'Global'
FEATURES = ['Lat', 'Lon', 'Alt', 'Temp', 'Precip', 'Year', 'JulianDay_Sin']

### Import Libraries
# Base Libraries
import numpy as np
import pandas as pd
import re
# Tensorflow, scikit, kerasTuner
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt

# Function to import a dataset and transform headers for easier coding and convert Date column
# Pseudocode:
# 1. Import Dataset Save old Headers for later
# 2. Transform headernames to rid of units
# 3. Convert Date into Year and Julian Day
# 4. Peform a Sine Transformation on JulianDay
# 5. Return the dataset, and old headers
def importData(fileName):
    # Read in the correct file
    dataset = pd.read_csv(f'../../Data/{fileName}.csv')
    oldCols = list(dataset.columns)

    # Remove any units (anything in parentheses)
    codeCols = list(map(lambda x: re.sub(r'\(([^()]*)\)', '', x).strip(), oldCols))
    dataset.columns = codeCols

    # Transform Date into Year and JulianDay_Sin
    dataset['Date'] = pd.to_datetime(dataset['Date'], utc=True)
    dataset['Year'] = dataset['Date'].dt.year
    dataset['JulianDay'] = dataset['Date'].dt.dayofyear
    # Sine transformation to account for cyclical nature of Julian Day
    dataset['JulianDay_Sin'] = np.sin(2*np.pi*dataset['JulianDay']/365) 
    dataset.drop(columns=['Date', 'JulianDay'], inplace=True)
    
    #Add year and JulianDay_Sin to oldCols 
    oldCols += ['Year', 'JulianDay_Sin']
    
    return dataset, oldCols

# Function that will sort dataset into scaled X and Y
# Pseudocode:
# 1. Separate dataset into Features and Target
# 2. Scale the Features array using MinMaxScaler
# 3. Return the scaled X, Y, and the scaler used
def scaleSplitData(dataset):
    # Separate Features and Target
    features = dataset[FEATURES]
    target = dataset[['O18', 'H2']]

    # Scale the Features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features.values)
    Y = target.values

    return X, Y, scaler

# Model Builder Function 
# Pseudocode:
# 1. Create a Sequential Model
# 2. Add an Input Layer
# 3. Prep the Search Space for Hyperparameter Tuning
# 4. Add a LSTM Layer with Hyperparameters
# 5. Add two Dense Layers with Hyperparameters 
# 6. Add a Dense Output Layer
# 7. Compile the Model with Hyperparameters
# 8. Return the Model
def modelBuilder(hp):
    # Create a Sequential Model
    model = Sequential()
    # Add an Input Layer
    model.add(InputLayer(shape=(len(FEATURES), 1)))

    # Prep the Search Space for Hyperparameter Tuning
    hp_numNeurons1 = hp.Int('numNeurons_LSTM', min_value=8, max_value=512, step=8)
    hp_numNeurons2 = hp.Int('numNeurons_Dense1', min_value=8, max_value=512, step=8)
    hp_numNeurons3 = hp.Int('numNeurons_Dense2', min_value=8, max_value=512, step=8)
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Add the hidden layers with Hyperparameters
    model.add(LSTM(units=hp_numNeurons1))
    model.add(Dense(units=hp_numNeurons2, activation='relu'))
    model.add(Dense(units=hp_numNeurons3, activation='relu'))

    # Add the Output Layer
    model.add(Dense(2))

    # Compile the Model with Hyperparameters
    model.compile(optimizer=Adam(learning_rate=hp_lr), loss='mse', metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model

# Hyperparameter tuning process
# Pseudocode:
# 1. Create the Hyperband Tuner
# 2. Create a callback to stop training early
# 3. Perform the search
# 4. Get the best model hyperparameters
# 5. Return the best hyperparameters
def hyperParameterTuning(xTrain, yTrain):
    # Create the Hyperband Tuner
    tuner = kt.Hyperband(modelBuilder, 
                        objective='val_loss', 
                        max_epochs=10, factor=3, 
                        directory='Hyperparameter_Tuner', project_name='G1')

    # Create a callback to stop training early
    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    # Perform the search
    tuner.search(xTrain, yTrain, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

    # Get the best model hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print('Finsihed Tuning')

    return best_hps

# Training Process
# Pseudocode:
# 1. Using the best hyperparameters, build the model
# 2. Early Stopping
# 3. Train the model
# 4. Return the trained model
def trainModel(xTrain, yTrain, best_hps):
    # Using the best hyperparameters, build the model
    model = modelBuilder(best_hps)

    # Early Stopping
    stop_early = EarlyStopping(monitor='val_loss', patience = 100)

    # Train the model
    model.fit(xTrain, yTrain, epochs=1000, validation_split=0.2, callbacks=[stop_early], verbose=1)

    print('Finished Training')
    
    return model

# Test data prediction using the test data and the trained model
# Pseudocode:
# 1. Scale the test data using the scaler
# 2. Predict the test data using the trained model
# 3. Combine the test data and the predictions with original headers
# 4. Save the results to a CSV
def predictTestData(xTest, yTest, model, scaler, oldCols):
    # Scale the test data using the scaler
    x = scaler.transform(xTest.values)
    
    # Predict the test data using the trained model
    yPreds = model.predict(x)

    # Combine the test data and the predictions with original headers
    testResults = pd.DataFrame(np.concatenate((xTest, yTest, yPreds), axis=1), columns=oldCols + ['O18 P (‰)', 'H2 P (‰)'])

    # Save the results to a CSV
    testResults.to_csv(f'Model_1_TestData.csv', index=False)



# Main Function
def main():
    # Import train data and original headers
    trainData, oldCols = importData('DataTrain')

    # Scale and Split the train data
    xTrain, yTrain, scaler = scaleSplitData(trainData)

    # Hyperparameter Tuning
    best_hps = hyperParameterTuning(xTrain, yTrain)

    # Train the Model
    model = trainModel(xTrain, yTrain, best_hps)

    # Import test data and original headers
    testData, _ = importData('DataTest')

    # Predict the test data using the trained model
    predictTestData(testData, testData[['O18', 'H2']], model, scaler, oldCols)

    # Save the model
    model.save('Model_1.keras')

main()
