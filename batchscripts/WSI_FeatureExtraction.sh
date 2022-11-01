#!/bin/bash
# To run it, ues:
# my_folder/*  | xargs -L 10 -0 bash WSI_patching.sh
# In this way, all the files in my_folder will be processed with WSI_Feat_Ext.sh code, 10 by 10

OUTPUT_PATH='/path/to/output'
SCRIPT_PATH='/path/to/scripts'
INPUT_PATH='/path/to/input'

for input_file in "$@"; do
	srun python $SCRIPT_PATH/Patching.py \
	--input_dir $INPUT_PATH \
	--sampl_name $DATA_PATH \
	--output_dir $OUTPUT_PATH \
	&
done
wait
