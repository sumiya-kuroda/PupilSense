#!/bin/bash

#SBATCH --job-name=pupilsense2
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
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

python -u ../pupilsense/infer_pupil.py /ceph/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/TAA0000066/ses-018_date-20250403_protocol-t12/behav/Camera2_2025-04-03T13_13_27.mp4