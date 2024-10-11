### Global Variables ####
MODELNUM = 3
SCHEME = 'Global'
FEATURES = ['Lat', 'Lon', 'Alt', 'Temp', 'Precip', 'KPN_A', 'KPN_B', 'KPN_C', 'KPN_D', 'KPN_E', 'Year', 'JulianDay_Sin']

### Import Libraries
# Base Libraries
import numpy as np
import pandas as pd
import re
import json
import sys
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
def modelBuilder(numNeurons1, numNeurons2, numNeurons3, lr):
    # Create a Sequential Model
    model = Sequential()
    # Add an Input Layer
    model.add(InputLayer(shape=(len(FEATURES), 1)))

    # Add the hidden layers with Hyperparameters
    model.add(LSTM(numNeurons1))
    model.add(Dense(numNeurons2, activation='relu'))
    model.add(Dense(numNeurons3, activation='relu'))

    # Add the Output Layer
    model.add(Dense(2))

    # Compile the Model with Hyperparameters
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model

# Hyperparameter Search Space
# Pseudocode:
# 1. Define the search space for the hyperparameters
# 2. Create a Hyperband Tuner Model from the search space
# 3. Create a callback to stop training early
# 4. Return the model
def hyperParameterSearchSpace(hp):
    # Prep the Search Space for Hyperparameter Tuning
    hp_numNeurons1 = hp.Choice('numNeurons_LSTM', values=[2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10])
    hp_numNeurons2 = hp.Choice('numNeurons_Dense1', values=[2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10])
    hp_numNeurons3 = hp.Choice('numNeurons_Dense2', values=[2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10])
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = modelBuilder(hp_numNeurons1, hp_numNeurons2, hp_numNeurons3, hp_lr)

    return model

# Hyperparameter tuning process
# Pseudocode:
# 1. Create the Hyperband Tuner
# 2. Create a callback to stop training early
# 3. Perform the search
# 4. Get the best model hyperparameters
# 5. Return the best hyperparameters
def hyperParameterTuning(xTrain, yTrain):
    print('Start Tuning')
    # Create the Hyperband Tuner
    tuner = kt.Hyperband(hyperParameterSearchSpace, 
                        objective='val_loss', 
                        max_epochs=10, factor=3,
                        directory='Hyperparameter_Tuning', project_name='Model_1',
                        overwrite=True)

    # Create a callback to stop training early
    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    # Perform the search
    tuner.search(xTrain, yTrain, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=0)

    # Get the best model hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Save the best hyperparameters to a JSON file
    with open('Best_Hyperparameters.json', 'w') as f:
        f.write(json.dumps(best_hps.values))

    print('Finsihed Tuning')

# Training Process
# Pseudocode:
# 1. Load the best hyperparameters
# 2. Using the best hyperparameters, build the model
# 3. Early Stopping
# 4. Train the model
# 5. Return the trained model
def trainModel(xTrain, yTrain):
    print('Start Training')

    # Load the best hyperparameters
    with open('Best_Hyperparameters.json', 'r') as f:
        hyperparams = json.load(f)

    # Using the best hyperparameters, build the model
    model = modelBuilder(hyperparams['numNeurons_LSTM'], hyperparams['numNeurons_Dense1'], hyperparams['numNeurons_Dense2'], hyperparams['learning_rate'])

    # Early Stopping
    stop_early = EarlyStopping(monitor='val_loss', patience = 100, restore_best_weights=True)

    # Train the model
    model.fit(xTrain, yTrain, epochs=1000, validation_split=0.2, callbacks=[stop_early], verbose=0)

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
    testResults = pd.DataFrame(np.concatenate((xTest, yTest, yPreds), axis=1), columns=FEATURES + ['O18 A', 'H2 A', 'O18 P', 'H2 P'])

    # Save the results to a CSV
    testResults.to_csv(f'Model_{MODELNUM}_TestData.csv', index=False)



# Main Function
def main():
    # Based on what script was run (train or train & tune), the script will decide if it needs to retune the model
    shouldTune = bool(sys.argv[1])

    # Import train data and original headers
    trainData, oldCols = importData('DataTrain')

    # Scale and Split the train data
    xTrain, yTrain, scaler = scaleSplitData(trainData)

    if shouldTune:
        # Hyperparameter Tuning
        hyperParameterTuning(xTrain, yTrain)

    # Train the Model
    model = trainModel(xTrain, yTrain)

    # Import test data and original headers
    testData, _ = importData('DataTest')

    # Predict the test data using the trained model
    predictTestData(testData[FEATURES], testData[['O18', 'H2']], model, scaler, oldCols)

    # Save the model
    model.save('Model_1.keras')

main()
