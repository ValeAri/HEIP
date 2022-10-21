#!/bin/bash
#SBATCH --job-name=infer_wsi
#SBATCH --ntasks=1
#SBATCH --time=15
#SBATCH --cpus-per-task 40
#SBATCH --error=/path/to/infer_wsi.err
#SBATCH --output=/path/to/infer_wsi1.txt
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=40G
#SBATCH --partition=gpu

# NOTE: MODIFY THE SLURM SETTINGS TO YOUR OWN NEEDS.

CKPT_PATH='/path/to/model.ckpt'
RESULT_PATH='/path/to/results'
DATA_PATH='/path/to/data'
SCRIPT_PATH='/path/to/scripts'

module purge
module load pytorch

srun python $SCRIPT_PATH/infer_wsi.py \
    --in_dir $DATA_PATH \
    --res_dir $RESULT_PATH \
    --ckpt_path $CKPT_PATH \
    --device 'cuda' \
    --n_devices 4 \
    --exp_name 'infer_test1' \
    --exp_version 'try1' \
    --batch_size 40 \
    --padding 120 \
    --stride 80 \
    --patch_size 256 \
    --classes_type 'bg,neoplastic,inflammatory,connective,dead,epithel' \
    --geo_format 'qupath' \
    --offsets 1 \

seff $SLURM_JOBID
