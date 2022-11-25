#!/bin/bash

#SBATCH --job-name=merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 10
#SBATCH --error=/path/to/merge1.err
#SBATCH --output=/path/to/merge1.txt
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=10G
#SBATCH --partition=gpu

SCRIPT_PATH='/path/to/scripts'
IN_DIR='/path/to/input/directory'
RESULT_DIR='/path/to/result/directory'
FNAME_CELLS='/name/of/json/file'


python $SCRIPT_PATH/merge.py \
    --in_dir $IN_DIR \
    --result_dir $RESULT_DIR \
    --fname_cells $FNAME_CELLS \

seff $SLURM_JOBID