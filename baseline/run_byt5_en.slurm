#!/bin/bash
#SBATCH --job-name=byt5-training
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --account=becs-delta-gpu
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=b.korojo@gmail.com


# Activate your virtual environment
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Print job information for debugging
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Command to run your Python script with torchrun for distributed training
srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --node_rank=0 \
  -m src.scripts.train \
  --batch_size=26 \
  --output_dir= ./byt5_checkpoints \
  --wandb_project=byt5-pretraining \
  --save_steps=5000 \
  --batch_size 26 ----tokenized_pcap_dir /projects/becs/data/tokenized_pcaps

echo "Job ended at $(date)"
