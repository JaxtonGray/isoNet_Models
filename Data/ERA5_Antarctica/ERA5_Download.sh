#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=Precip_Download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=20:00:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Setup the environment
module load python/3.11
module load openmpi mpi4py netcdf hdf5 proj 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index xarray geopandas h5netcdf netcdf4 mpi4py h5py
pip install --no-index pandas

python antarctica_ERA5.py