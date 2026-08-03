#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=Antarctica_Download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=20:00:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Setup the environment
module load python/3.12
module load openmpi mpi4py netcdf hdf5 proj 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Load required libraries
pip install --no-index xarray dask geopandas requests aiohttp h5netcdf netcdf4 mpi4py h5py fsspec zarr
pip install --no-index pandas

# Run the script
python antarctica_ERA5.py ../../Runs/Global_Modelling/grid_points.geojson
