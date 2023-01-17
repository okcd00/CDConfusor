#!/bin/bash

set -v
set -e


INPUT_FILE=../data/cn/sighan15/TestInput.txt
#INPUT_FILE=../data/cn/cctc/cctc_test.dcn.txt 
#INPUT_FILE=../data/cn/findoc/findoc_test.dcn.txt 

OUTPUT_FILE=output.txt
MODEL_DIR=dcn_models/
MAX_LENGTH=192
BATCH_SIZE=4

CUDA_VISIBLE_DEVICES=7 python predict_DCN.py \
    --model $MODEL_DIR \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --batch_size $BATCH_SIZE \
	--max_len $MAX_LENGTH 