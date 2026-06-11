#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=FillData
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=05:00:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=SLURM_Output/fill_data_%j.out

# !/bin/bash
# This section will grab the model name to run
# Set up the environment
module load python/3.12
module load proj

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index pandas numpy geopandas scipy rasterio xarray

# Run the training script
python fillData.py LeaveOut\DataLeaveOut.csv O18 H2