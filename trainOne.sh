#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=model_training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=2:30:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# !/bin/bash
# This section will grab the model name to run
modelName=$(sed -n 1p modelList.txt)
echo $modelName

# Set up the environment
module load python/3.11
module load proj

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index tensorflow pandas geopandas numpy scikit-learn keras-tuner

cd Models/$modelName

# Run the training script
python ../modelTraining.py "$modelName"
