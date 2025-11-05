#%%
# Script to take in the unclean data and perform cleaning operations
import pandas as pd

# Function to read in the unclean data
def read_data(file_path):
    # Read the uncleaned data from a CSV file
    
    return pd.read_csv(file_path, low_memory=False)

# Remove unnecessary columns and clean up column names
def column_cleanup(df):
    # Set the columns to keep
    columns_to_keep = [
        'Latitude',
        'Longitude',
        'Altitude',
        'Sample Date',
        'Measurand Symbol',
        'Measurand Amount'
    ]
    # Select only the necessary columns
    df = data[columns_to_keep]

    # Rename columns for consistency and clarity
    df = df.rename(columns={
    'Latitude': 'Lat',
    'Longitude': 'Lon',
    'Altitude': 'Alt',
    'Sample Date': 'Date',
    'Measurand Symbol': 'Symbol',
    'Measurand Amount': 'Amount'
    }) 
    # Remove specified unnecessary columns from the DataFrame
    return df

# Change the symbol and amount columns to separate columns
def pivot_data(df):
    # Pivot the table to be long format
    df = df.pivot_table(index=['Lat', 'Lon', 'Alt', 'Date'],
                        columns='Symbol', 
                        values='Amount', 
                        aggfunc='first').reset_index()
    return pd.DataFrame(df)

# Remove rows with missing values in critical columns
def remove_missing_values(df, critical_columns):
    # Drop rows with missing values in the specified critical columns
    df = df.dropna(subset=critical_columns)
    return df

#%%
if __name__ == "__main__":
    # File path to the unclean data
    file_ver = "2025-07-22"
    data = read_data(f"GNIP_Uncleaned ({file_ver}).csv")

    # Remove unnecessary columns and clean up column names
    cleanedCols_data = column_cleanup(data)

    # Pivot the data to have separate columns for each measurand symbol
    pivoted_data = pivot_data(cleanedCols_data)

    # Remove rows with missing values in critical columns
    noMissing_data = remove_missing_values(pivoted_data, ['O18', 'H2'])

    # Save the cleaned data to a new CSV file
    noMissing_data.to_csv(f"GNIP_Cleaned ({file_ver}).csv", index=False)
# %%
