#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=pythonTest1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000MB
#SBATCH --time=1:00:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
python modelTraining.py


