#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 10
#SBATCH --error=/path/to/merge1.err
#SBATCH --output=/path/to/merge1.txt
#SBATCH --mem=40G
#SBATCH --partition=cpu

SCRIPT_PATH='/path/to/scripts'
IN_DIR='/path/to/input/directory'

srun python $SCRIPT_PATH/merge.py \
    --in_dir $IN_DIR \
    --overwrite 0 \
    --use_pbar 1 \

seff $SLURM_JOBID
