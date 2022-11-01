#!/bin/bash
# To run it, ues:
# my_folder/* | xargs -L 10 -0 bash WSI_preprocessing.sh
# In this way, all the files in my_folder will be processed with WSI_preprocessing.sh code, 10 by 10

OUTPUT_PATH='/path/to/output'
SCRIPT_PATH='/path/to/scripts'
INPUT_PATH='/path/to/input'
for input_file in "$@"; do
	srun python $SCRIPT_PATH/Preprocessing.py \
	--input_dir $INPUT_PATH \
	--output_dir $OUTPUT_PATH \
	--width 1250 \
	--downsample 64 \
	--kernel_size 30 \
	--hue_q01 144 \
	--black_pixels 0.2 \
	--brightness_q01 45 \
	--saturation_q05 30 \
	--sharpness_max 5
	&
done
wait