###############################################################################
# This script is used to download the KPN dataset from https://www.gloh2o.org/koppen/
# This dataset consists of multiple years and future projections of the Koppen-Geiger 
# climate classification which is available in different resolutions (0.01, 0.1, 0.5, 1 degrees)
# The script downloads the data and saves it to a local directory
################################################################################

# Importing required libraries
import requests as req
import zipfile as zf
import os


# Check current working directory and navigate to the Data/KPN folder if not already there
cwd = os.getcwd()
if not cwd.endswith("Data/KPN"):
    os.chdir(os.path.join(cwd, "Data", "KPN"))

# For V2 data this is the link to the data, check the link in the header of this for more information
response = req.get("https://figshare.com/ndownloader/files/45057352")
with open("koppen-geiger.zip", "wb") as f:
    f.write(response.content)
    response.close()

# Extract the contents of the zip file
with zf.ZipFile("koppen-geiger.zip", "r") as zip_ref:
    zip_ref.extractall()

# Remove the zip file after extraction to save space
os.remove("koppen-geiger.zip")