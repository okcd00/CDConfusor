#!/bin/bash

set -v
set -e

#TASK=sighan15
#MODEL_DIR=dcn_models/wsw5_train_model_192/
TASK=rw
MODEL_DIR=dcn_models/findoc_finetuned_230313/

INPUT_FILE=../data/cn/$TASK/${TASK}_test.sighan.txt
TAB_INDEX=1

DCN_FILE=../data/cn/$TASK/${TASK}_test.dcn.txt
OUTPUT_FILE=$MODEL_DIR/output_${TASK}.txt
MAX_LENGTH=192
BATCH_SIZE=4

CUDA_VISIBLE_DEVICES=1 python predict_DCN.py \
    --model $MODEL_DIR \
    --input_file $INPUT_FILE \
    --text_tab_index 1 \
    --output_file $OUTPUT_FILE \
    --batch_size $BATCH_SIZE \
	--max_len $MAX_LENGTH 


python evaluate.py $DCN_FILE $OUTPUT_FILE 
