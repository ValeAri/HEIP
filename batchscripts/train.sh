#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task 40
#SBATCH --error=/path/to/train1.err
#SBATCH --output=/path/to/train1.txt
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=40G
#SBATCH --partition=gpu

EXPERIMENT_PATH='/path/to/exp_dor'
DATA_PATH='/path/to/data'
SCRIPT_PATH='/path/to/scripts'
CONFIG_PATH='/path/to/train_configs'

module purge
module load pytorch

srun python $SCRIPT_PATH/train.py \
    --yaml_path $CONFIG_PATH/train1.yaml \
    --exp_dir $EXPERIMENT_PATH/training_experiments \
    --exp_name 'train_test' \
    --exp_version '1' \
    --accelerator 'gpu' \
    --strategy 'ddp' \
    --n_devices 4 \
    --n_epoch 40 \
    --num_workers 10 \
    --batch_size 10 \
    --train_ds $DATA_PATH/exp1/train_cin_320x320s80_simple.h5 \
    --valid_ds $DATA_PATH/exp1/valid_cin_256x256s256_simple.h5 \
    --test_ds $DATA_PATH/exp1/valid_cin_256x256s256_simple.h5 \
    --inst_tr 'cellpose' \
    --img_tr 'blur,hue_sat' \
    --norm 'minmax' \
    --ret_binary 0 \
    --ret_inst 0 \
    --ret_type 1 \
    --ret_sem 1 \
    --use_wandb 1 \
    --seed 422 \
    --classes_type 'bg,neoplastic,inflammatory,connective,dead,glandular_epithel,squamous_epithel' \
    --move_metrics_to_cpu 0 \
    --class_metrics 1 \
    --run_test 0 \

seff $SLURM_JOBID
