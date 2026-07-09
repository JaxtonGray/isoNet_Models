#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=Test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=00:05:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=../SLURM_Output/test_run_%j.out

# !/bin/bash
# This section will grab the model name to run
# Set up the environment
module purge
module load StdEnv/2023
module load python/3.12
module load proj
module load hdf5

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index dask pandas numpy geopandas scipy rasterio xarray h5netcdf h5py

# Run the training script
python test.py
