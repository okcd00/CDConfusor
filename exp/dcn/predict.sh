#!/bin/bash

set -v
set -e

TASK=cctc
MODEL_DIR=dcn_models/wsw_train_model_192/

INPUT_FILE=../data/cn/$TASK/${TASK}_test.sighan.txt
DCN_FILE=../data/cn/$TASK/${TASK}_test.dcn.txt
OUTPUT_FILE=$MODEL_DIR/output_${TASK}.txt
MAX_LENGTH=192
BATCH_SIZE=4

CUDA_VISIBLE_DEVICES=7 python predict_DCN.py \
    --model $MODEL_DIR \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --batch_size $BATCH_SIZE \
	--max_len $MAX_LENGTH 


python evaluate.py $DCN_FILE $OUTPUT_FILE 
