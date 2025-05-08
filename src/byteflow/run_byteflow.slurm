#!/bin/bash
#SBATCH --job-name=byt5-training
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --account=becs-delta-gpu
#SBATCH --output=run_byt5_en.out
#SBATCH --error=run_byt5_en.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=b.korojo@gmail.com


# Activate your virtual environment
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Print job information for debugging
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"

cd baseline
python byt5_small_en_baseline.py

echo "Job ended at $(date)"
