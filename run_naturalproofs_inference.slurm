#!/bin/bash
#SBATCH --job-name=naturalproofs
#SBATCH --output=naturalproofs_inference_%j.out
#SBATCH --partition=education_gpu
#SBATCH --time=10:00:00                   # 1 hour max
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kien.lau@yale.edu    # ← replace with your Yale email

# Load modules (if your cluster uses modules)

module load miniconda            # gives you conda + python
module load cuda/12.1
# module load python/3.10

# Activate your virtual env (if you set one up)
source ~/myenv/bin/activate myenv

srun python inference_naturalproofs.py