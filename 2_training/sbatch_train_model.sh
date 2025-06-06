#!/bin/bash

#SBATCH --job-name=pupilsense2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.kuroda@ucl.ac.uk

source /etc/profile.d/modules.sh
echo "SLURM job info:"
scontrol show job $SLURM_JOB_ID

module load cuda/12.4
module load miniconda
source activate pupilsense

python -u ../pupilsense/train_model.py