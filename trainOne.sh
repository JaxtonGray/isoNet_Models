#!/bin/bash
#SBATCH --account=def-stadnykt-ab
#SBATCH --job-name=model_training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000MB
#SBATCH --time=02:30:00
#SBATCH --mail-user=jaxton.gray@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=SLURM_Output/model_training_%j.out

# !/bin/bash
# This section will grab the model name to run
modelInfo=$(sed -n 1p modelList.txt)

# Split the modelInfo into modelNum and modelName
IFS=' ' read -ra arr <<< "$modelInfo"
modelNum=${arr[0]}
modelName=${arr[1]}


# Set up the environment
module load python/3.11
module load proj

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index tensorflow pandas geopandas numpy scikit-learn keras-tuner

cd $modelName

# Run the training script
python Model_Training/modelTraining.py "$modelNum" "$modelName"
