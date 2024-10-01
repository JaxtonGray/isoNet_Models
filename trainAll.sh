#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=model_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=2:15:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=1-2

# !/bin/bash
# This section will grab the model name to run
modelName=$(sed -n ${SLURM_ARRAY_TASK_ID}p modelList.txt)
echo $modelName

cd Models/$modelName

# Set up the environment
module load python/3.11.5
module load proj

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install --no-index tensorflow

# Run the training script
python modelTraining.py