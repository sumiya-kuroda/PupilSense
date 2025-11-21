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
conda activate pupilsense2

export PUPILSENSE_PATH="/nfs/nhome/live/skuroda/Workstation2025/PupilSense"
export INPUT_LIST="eyevideo_dmdmhuman.txt"
export DEST_CEPH="/ceph/mrsic_flogel/public/projects/SuKuMiLo_20220107_DMDM_Human/DMDMHumanData/PupilSenseTraining/train"
export LOG_DIR="/ceph/mrsic_flogel/public/projects/SuKuMiLo_20220107_DMDM_Human/DMDMHumanData/PupilSenseLog"
export PYTHONUNBUFFERED=1

snakemake --cores 4 --executor slurm --jobs 10 --slurm-keep-successful-logs --use-conda --jobname "{rule}.{jobid}" --profile swchpc/
echo "Done! Good job! End time:" $(date '+%Y-%m-%d %H:%M:%S')