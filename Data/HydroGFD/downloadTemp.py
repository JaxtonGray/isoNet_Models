# Import required libraries
import os
import cdsapi
import zipfile

# Retrieve precipitation data from Copernicus Climate Data Store
client = cdsapi.Client()

dataset = "sis-ecv-cmip5-bias-corrected"
request = {
    "variable": "mean_2m_temperature",
    "model": "ec_earth",
    "experiment": "rcp_4_5",
    "period": [
        "19500101_19741231",
        "19750101_20051231",
        "20060101_20301231"
    ]
}

output_file = "mean_2m_temperature_ec_earth_rcp45.zip"

client.retrieve(dataset, request, output_file)

# Unzip the downloaded file
with zipfile.ZipFile(output_file, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(output_file))
    