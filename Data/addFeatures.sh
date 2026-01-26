#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=Add_Features
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=00:30:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Setup the environment
module load python/3.12 openmpi mpi4py
module load proj
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Load required libraries
pip install --no-index rasterio xarray geopandas h5netcdf netcdf4 
pip install --no-index pandas

# Run the script
python featureModularization.py