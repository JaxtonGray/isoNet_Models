#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=Wind_Download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=2:00:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL


#!/bin/bash

# Setup the environment
module load python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Load in the required libraries
pip install --no-index --upgrade pip
pip install --no-index cdsapi
python downloadWindERA5.py

# Give file permissions
chmod 777 *