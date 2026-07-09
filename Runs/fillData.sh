#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=FillData
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=00:05:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=../SLURM_Output/fill_data_%j.out

# !/bin/bash
# This section will grab the model name to run
# Set up the environment
module --force purge
module load StdEnv/2023
module load python/3.12
module load hdf5
module load netcdf
module load proj

module list

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index pandas geopandas numpy scipy rasterio xarray dask netcdf4

pip list | grep h5

# Run the training script
python fill_data_monthly.py LeaveOut/DataLeaveOut.csv O18 H2
