#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=Examine_Grib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000MB
#SBATCH --time=1:00:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Setup the environment
module load python/3.11 eccodes
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Load in the required libraries
pip install --no-index --upgrade pip
pip install -r ../requirements.txt

# Run the script
python testGrib.py