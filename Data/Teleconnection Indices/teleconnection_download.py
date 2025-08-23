# Script meant to download the ENSO teleconnection (SOI), North Atlantic Oscillation (NAO), 
# and the Antarctic Oscillation (AAO).
#%%
# Import required libraries
import io
import pandas as pd
import requests








# Function to retrieve a index and return as long-form dataframe
def retrieve_index(url):
    # Retrieve data and remove header
    r = requests.get(url) 
    data = r.text.split('\n')[3:]

    # Convert to DataFrame
    df = pd.read_csv(io.StringIO('\n'.join(data)), delimiter=r'\s+', header=0)

    # Melt the dataframe to have the columns 'Year', 'Month', 'ENSO'
    df_melted = df.melt(id_vars=['YEAR'], var_name='MONTH', value_name='ENSO')

    # Rename first 2 columns to title case
    df_melted.rename(columns={'YEAR': 'Year', 'MONTH': 'Month'}, inplace=True)
#%%
# Check for a header in a returned string
def check_header(inputStr):
    # Cycle through the lines and check to see if they change format or remain consist
    testLines = [line.strip() for line in inputStr.splitlines()]
    for i, line in enumerate(testLines):
        print(i, len(line))
#%%
if __name__ == "__main__":
    teleIndices = {
        'ENSO': r'https://www.cpc.ncep.noaa.gov/data/indices/soi',
        'NAO': r'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table',
        'AAO': r'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii.table'
    }

    r = requests.get(teleIndices['ENSO'])
    check_header(r.text)