#!/bin/bash
# To run it, ues:
# find my_folder/ -name '*.mrxs' -print0  | xargs -L 10 -0 bash WSI_patching.sh
# In this way, the .mrxs files in my_folder will be processed with WSI_patching.sh code, 10 by 10

OUTPUT_PATH='/path/to/output'
SCRIPT_PATH='/path/to/scripts'

for input_file in "$@"; do
	srun python $SCRIPT_PATH/Patching.py \
	--sampl_name $DATA_PATH \
	--output_dir $OUTPUT_PATH \
	--width 1250 \
	--overlap 0 \
	--max_background 0.92 \
	--image_format 'png' \
	--quality 100 \
	&
done
wait