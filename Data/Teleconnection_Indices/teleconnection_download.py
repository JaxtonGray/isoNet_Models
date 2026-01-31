# Script meant to download the ENSO teleconnection (SOI), North Atlantic Oscillation (NAO), 
# and the Antarctic Oscillation (AAO).
#%%
# Import required libraries
import io, datetime
import pandas as pd
import requests

# Function to retrieve a index and return as long-form dataframe
def retrieve_index(index, url, header, end):
    # Retrieve data and remove header
    r = requests.get(url) 

    if end != -1:
        end += 1
    data = r.text.split('\n')[header+1:end]
    headers = ['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    # Convert to DataFrame
    df = pd.read_csv(io.StringIO('\n'.join(data)), delimiter=r'\s+', header=None, names=headers)

    # Melt the dataframe to have the columns 'Year', 'Month', index
    df_melted = df.melt(id_vars=['YEAR'], var_name='MONTH', value_name=index)

    # Rename first 2 columns to title case
    df_melted.rename(columns={'YEAR': 'Year', 'MONTH': 'Month'}, inplace=True)

    # Change the month column to numeric from Short code JAN --> 1, FEB --> 2, etc.
    df_melted['Month'] = df_melted['Month'].apply(lambda x: datetime.datetime.strptime(x.title(), '%b').month)

    return df_melted

if __name__ == "__main__":
    teleIndices = {
        'ENSO': {'url': r'https://www.cpc.ncep.noaa.gov/data/indices/soi', 'header': 3, 'end':78},
        'NAO': {'url': r'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table', 'header': 0, 'end': -1},
    }

    # Loop through teleconnection indices and retrieve data and append it to the dictionary
    for index, params in teleIndices.items():
        df = retrieve_index(index, params['url'], params['header'], params['end'])
        teleIndices[index]['data'] = df

    # Merge each dataframe on 'Year' and 'Month'
    combined_df = teleIndices['ENSO']['data']
    for index, params in teleIndices.items():
        if index != 'ENSO':
            combined_df = combined_df.merge(params['data'], on=['Year', 'Month'], how='left', suffixes=('', f'_{index}'))

    # Save combined dataframe to CSV
    combined_df.to_csv('Data/Teleconnection_Indices/teleconnection_indices.csv', index=False)