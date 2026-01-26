#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=Add_Features
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=1:00:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Setup the environment
module load python/3.12 proj
virtualenv --no-download env
source env/bin/activate

# Load required libraries
pip install --no-index rasterio xarray geopandas 
pip install --no-index pandas

# Run the script
python featureModularization.py