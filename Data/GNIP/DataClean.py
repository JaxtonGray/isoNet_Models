## This file will be a combination of functions that both clean that GNIP data and combine it with other data sources
## depending on the features we want to include in the model

# Importing the necessary libraries
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split
import glob

# Function to clean the GNIP data
def cleanData():
    # Load in the uncleaned GNIP data
    dataUnclean = pd.read_csv('GNIP_Uncleaned.csv')

    # Changing the measurement symbol/unit and amount into separate columns
    data = dataUnclean.copy()
    data['Precip (mm)'] = data['Amount'].where(data['Symbol'] == 'Precipitation', np.nan)
    data['Temp (\u00B0C)'] = data['Amount'].where(data['Symbol'] == 'TempAir', np.nan)
    data['O18 (\u2030)'] = data['Amount'].where(data['Symbol'] == 'O18', np.nan)
    data['H2 (\u2030)'] = data['Amount'].where(data['Symbol'] == 'H2', np.nan)
    data = data.drop(['Amount', 'Symbol', 'Units', 'SampleType'], axis=1)

    # Changing the date to a datetime object
    data['Date'] = pd.to_datetime(data['Date'], utc=True)

    # Combine the rows with the same date, lat, and lon into one row
    dataAgg = data.groupby(['Lat', 'Lon', 'Date', 'Alt']).agg({
        'Precip (mm)': 'first',  # Replace 'first' with your preferred aggregation for non-NaN values
        'Temp (\u00B0C)': 'first',
        'O18 (\u2030)': 'first',
        'H2 (\u2030)': 'first'
    }).reset_index()

    # Remove rows with NaN values in the O18 and H2 columns as they are the target variables
    dataDrop = dataAgg.dropna(subset=['O18 (\u2030)', 'H2 (\u2030)'])

    return dataDrop

# In order to combine the GNIP data with the HydroGFD data, we need to load in the HydroGFD data, to make matters easier later on I have created a function that will load
# the HydroGFD data and return a dictionary with the years as keys and the file names as values
def loadHydroGFD():
    allHydroGFD = glob.glob("../HydroGFD/datasets/*.nc")
    dictHydroGFD = {}
    for file in allHydroGFD:
        dates = file.split('_')
        dates = dates[-1].split('-')
        
        dateTuple = (int(dates[0][:4]), int(dates[1][:4]))

        if dateTuple in dictHydroGFD:
            dictHydroGFD[dateTuple].append(file)
        else:
            dictHydroGFD[dateTuple] = [file]
        
    return dictHydroGFD

# This function will locate missing precipitation data using lat, lon, and date and return the precipitation value
def precipFinding(date, lat, lon, dictHydroGFD):
    # Find the year of the date
    year = date.year
    date = date.tz_localize(None)
    
    # Find the HydroGFD file that contains the date
    for key in dictHydroGFD:
        if year >= key[0] and year <= key[1]:
            file = dictHydroGFD[key][0]
            break

    # Load in the HydroGFD data
    xrid = xr.open_dataset(file)

    # Find the precipitation value
    precipValue = xrid['prAdjust'].sel(lat=lat, lon=lon, time=date, method='nearest').item()
    precipValue *= 86400  # Convert from kg/m^2/s to mm/day

    return precipValue

# This function will locate missing temperature data using lat, lon, and date and return the temperature value
def tempFinding(date, lat, lon, dictHydroGFD):
    # Find the year of the date
    year = date.year
    date = date.tz_localize(None)
    
    # Find the HydroGFD file that contains the date
    for key in dictHydroGFD:
        if year >= key[0] and year <= key[1]:
            file = dictHydroGFD[key][0]
            break

    # Load in the HydroGFD data
    xrid = xr.open_dataset(file)

    # Find the temperature value
    tempValue = xrid['tasAdjust'].sel(lat=lat, lon=lon, time=date, method='nearest').item()

    return tempValue

# Fill in the missing precipitation and temperature data in the GNIP data
def missingData(data, dictHydroGFD):
    # Find the missing precipitation and temperature data
    data[data['Precip (mm)'].isnull()] = data.apply(lambda x: precipFinding(x['Date'], x['Lat'], x['Lon'], dictHydroGFD) if pd.isnull(x['Precip (mm)']) else x['Precip (mm)'], axis=1)
    data[data['Temp (\u00B0C)'].isnull()] = data.apply(lambda x: tempFinding(x['Date'], x['Lat'], x['Lon'], dictHydroGFD) if pd.isnull(x['Temp (\u00B0C)']) else x['Temp (\u00B0C)'], axis=1)

    # Drop the rows with missing precipitation and temperature data
    data = data.dropna(subset=['Precip (mm)', 'Temp (\u00B0C)'])

    return data

# Split the data into training and testing sets (80% training, 20% testing) saved as GNIP_Train.csv and GNIP_Test.csv
def splitData(data):
    train, test = train_test_split(data, test_size=0.2)
    train.to_csv('GNIP_Train.csv', index=False)
    test.to_csv('GNIP_Test.csv', index=False)


def main():
    df = cleanData()
    dictHydroGFD = loadHydroGFD()
    data = missingData(df, dictHydroGFD)
    data = data.to_csv('GNIP_CleanedTEST.csv', index=False)
    splitData(data)

main()
