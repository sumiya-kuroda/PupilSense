#!/bin/bash

#SBATCH --job-name=pupilsense
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.kuroda@ucl.ac.uk

source /etc/profile.d/modules.sh
echo "SLURM job info:"
scontrol show job $SLURM_JOB_ID

module load cuda/12.4
module load miniconda
conda activate pupilsense

python ../pupilsense/extract_frames.py {my_eye_camera.mp4}