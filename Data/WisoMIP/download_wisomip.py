# Download the data from the WisoMIP website

import requests as req
import xarray as xr
import pandas as pd
import numpy as np

# Function to download the data from the WisoMIP website
def download_wisomip_data(url):
    # Arguments:
    # url: The URL of the data file to download
    # Returns:
    # - The xarray dataset containing the data

    # Check to see if url works, if so grab the data
    if req.head(url).status_code == 200:
        r = req.get(url)
        return xr.Dataset.from_netcdf(r.content, decode_times=False)
    else:
        raise ValueError(f"URL {url} is not accessible. Status code: {req.head(url).status_code}")

# Function for decoding the time variable in the dataset
def decode_time_variable(ds, time_var_name='time'):
    # Arguments:
    # ds: The xarray dataset containing the data
    # time_var_name: The name of the time variable in the dataset (default is 'time')
    # Returns:
    # - The xarray dataset with the time variable decoded

    time_var, timeunits = ds[time_var_name].values, ds[time_var_name].attrs['units']

    base_time = pd.to_datetime(timeunits.split('since')[-1].strip())
    decoded_time = np.vectorize(lambda m: base_time + pd.DateOffset(months=int(m)))(time_var)
    ds[time_var_name] = decoded_time
    return ds

if __name__ == "__main__":
    # URL of the data file to download
    monthly_data_url = r"https://portal.nccs.nasa.gov/datashare/giss-publish/pub/paleoclimate/wisomip/1.ENSEMBLE/Total.ENSEMBLE_Monthly.nc"

    # Download the data
    ds = download_wisomip_data(monthly_data_url)

    # Decode the time variable
    ds = decode_time_variable(ds)

    # Save the dataset to a local NetCDF file
    ds.to_netcdf("Total_ENSEMBLE_Monthly_decoded.nc")