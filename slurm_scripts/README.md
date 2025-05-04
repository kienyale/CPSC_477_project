# Running Analyses on Yale Grace HPC with SLURM

This directory contains SLURM job scripts for running various analyses on HPC clusters. Each script is configured for Yale's HPC environment but can be adapted for other clusters.

## Available Scripts

1. `run_MATH_train_test.slurm`: Generate training and testing datasets from MATH problems
   ```bash
   sbatch run_MATH_train_test.slurm
   ```

2. `run_train_RL.slurm`: Train the RL-based detector
   ```bash
   sbatch run_train_RL.slurm
   ```


3. `run_naturalproofs_inference.slurm`: Generate LLM answers in NaturalProofs dataset
   ```bash
   sbatch run_naturalproofs_inference.slurm
   ```


4. `run_eval_naturalproofs.slurm`: Run inference for .npz (prediction values) on Naturalproofs
   ```bash
   sbatch run_eval_naturalproofs.slurm
   ```

5. `run_inference.slurm`: Run inference for .npz (prediction values) on MATH test set
   ```bash
   sbatch run_inference.slurm
   ```



## Configuration

Each script is configured with the following default resources:
- 1 GPU
- 16GB RAM
- 4 CPUs
- 15-hour runtime limit

To modify these settings, edit the SBATCH parameters in each script:
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
```

## Environment Setup

The scripts assume you have:
1. CUDA 12.1 available (`module load cuda/12.1`)
2. Python 3.10 (`module load python/3.10`)
3. A virtual environment with required packages

To set up your environment:
```bash
# Create and activate virtual environment
python -m venv myenv
source myenv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Output Files

Each job will create:
- Log file: `[jobname]_%j.out` (where %j is the job ID)
- Model checkpoints in `models/`
- Processed data in `data/processed/`
- Evaluation results in project root

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View job output in real-time:
```bash
tail -f [jobname]_[jobid].out
```

Cancel a job:
```bash
scancel [jobid]
```

## Common Issues

1. If you get CUDA out of memory errors:
   - Reduce batch size in config.yaml
   - Request more GPU memory in SLURM script

2. If job times out:
   - Increase time limit with `#SBATCH --time=HH:MM:SS`
   - Split task into smaller jobs

3. If environment modules are missing:
   - Check module availability: `module avail`
   - Update module load commands in scripts

## Notes

- All paths in scripts are relative to project root
- Make sure data files are in place before running jobs
- Monitor GPU memory usage with `nvidia-smi`
- Use `--mail-type=END,FAIL` to get email notifications 