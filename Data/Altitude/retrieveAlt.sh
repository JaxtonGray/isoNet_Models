#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=Altitude_Retrieval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=4:00:00
#SBATCH --array=1-3
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

#!/bin/bash
# This section will grab the CSV file to process
csvFile=$(sed -n ${SLURM_ARRAY_TASK_ID}p csvFileList.txt)

# Declare the CSV file being processed
echo "Processing file: $csvFile"


# Setup the environment
module load python/3.12
module load proj
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index xarray pandas geopandas dask zarr fsspec aiohttp requests

# Run the altitude retrieval script for the specified CSV file
python downloadAltitude.py "$csvFile"