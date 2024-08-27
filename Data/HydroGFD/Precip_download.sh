#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=Precip_Download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=2:00:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Setup the environment
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r req_download.txt

# Run the program
python downloadPrecip.py

# Extract the zip file within the datasets folder
unzip -o datasets/precipDownload.zip -d datasets/

# Remove the zip file
rm datasets/precipDownload.zip