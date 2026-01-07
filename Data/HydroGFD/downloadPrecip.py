# Import required libraries
import os
import cdsapi
import zipfile

# Retrieve precipitation data from Copernicus Climate Data Store
client = cdsapi.Client()

dataset = "sis-ecv-cmip5-bias-corrected"
request = {
    "variable": "precipitation_flux",
    "model": "ec_earth",
    "experiment": "rcp_4_5",
    "period": [
        "19500101_19741231",
        "19750101_20051231",
        "20060101_20301231"
    ]
}

output_file = "precipitation_flux_ec_earth_rcp45.zip"

client.retrieve(dataset, request, output_file)

# Unzip the downloaded file
with zipfile.ZipFile(output_file, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(output_file))

# Move extracted files to a separate files folder
os.makedirs("data_files", exist_ok=True)
for file in os.listdir(os.getcwd()):
    if file.endswith(".nc"):
        os.rename(file, os.path.join("data_files", file))

# Delete the original zip file to save space
os.remove(output_file)