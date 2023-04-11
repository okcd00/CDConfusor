set -v
set -e

# --------------
# | data files |
# --------------

# TRAIN_FILE=../data/cn/Wang271k/dcn_train.mp.dcn.txt  # W271k: 281381 | mp544192
# TRAIN_FILE=../data/tmp/findoc_train.230329.dcn.txt  # W271k+cctc+rfd: 279386 | mp558772
# TRAIN_FILE=../data/tmp/findoc_train.230329.augw1.dcn.txt  # W271k+cctc+rfd: 394822
# TRAIN_FILE=../data/tmp/findoc_train.230329.augw2.dcn.txt  # W271k+cctc+rfd: 508229
# TRAIN_FILE=../data/cn/cctc/cctc_train.dcn.txt  # CCTC_train: 646 
# TRAIN_FILE=../data/cn/rw/rw_test.dcn.txt  # RW = CCTC_test + RFD_test: 1089
# TRAIN_FILE=../data/cn/findoc/findoc_test.v1.dcn.txt  # RFD_test: 368 
# TRAIN_FILE=../data/cn/findoc/findoc_test.v2.dcn.txt  # RFD_train: 6569
TRAIN_FILE=../data/fin/findoc_train.230406.dcn.txt  # findoc-corpus: 7680964
DATASET_LINES=7680964

# TEST_FILE=../data/cn/findoc/findoc_test.v1.dcn.txt  # RFD_test: 368
# TEST_FILE=../data/cn/cctc/cctc_test.dcn.txt  # CCTC_test: 721
TEST_FILE=../data/cn/rw/rw_test.dcn.txt
SIGHAN_TEST_FILE=../data/cn/sighan15/sighan15_test.dcn.txt

# --model_name_or_path $WARMUP_DIR (modified roberta vocab)
BERT_MODEL=../pretrained_models/chinese-roberta-wwm-ext/
# WARMUP_DIR=cd_models/findoc_finetuned_w271k+cctc+rfd/  # nomp
WARMUP_DIR=cd_models/findoc_finetuned_230410_multigpu/checkpoint-153616/
OUTPUT_DIR=cd_models/findoc_finetuned_230410_multigpu/

mkdir -p $OUTPUT_DIR
cp ./train_temp.sh $OUTPUT_DIR/train_temp.sh

# --------------
# | configs    |
# --------------

# model hyper-parameters
SEED=1038
LR=5e-5
MIN_LR=2e-6
SAVE_TOTAL_LIMIT=10
NUM_EPOCHS=10

# 128 for sighan, 192 for dcn-train
MAX_LENGTH=192  
# 4 for 192-text-len on 12G GPU 
# 12 for 192-text-len on 24G GPU / 10 for 192-text-len on multi-GPU
BATCH_SIZE=10

# custom settings for training
GPUS=1,2,3,4,5,6,7,8
GPU_COUNT=$(echo $GPUS | tr -cd "[0-9]" | wc -c)
LOG_STEPS=$(echo "$DATASET_LINES/$GPU_COUNT/$BATCH_SIZE+1" | bc)
SAVE_STEPS=$(echo "$LOG_STEPS/10" | bc)
WARMUP_STEPS=$(echo "$SAVE_STEPS/10" | bc)
echo "In this run, LOG_STEPS is $LOG_STEPS, SAVE_STEPS is $SAVE_STEPS"
TOKENIZER_PARALLELISM=true

# --------------
# | RUN !!!    |
# --------------

# run the command
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$GPU_COUNT train_DCN.py \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LR  \
    --min_lr $MIN_LR \
    --warmup_steps $WARMUP_STEPS \
    --model_type=bert \
    --model_name_or_path $WARMUP_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
	--logging_steps $LOG_STEPS \
    --logging_first_step \
	--block_size $MAX_LENGTH \
    --train_data_file $TRAIN_FILE \
    --eval_data_file $SIGHAN_TEST_FILE \
    --test_data_file $TEST_FILE \
    --seed $SEED \
    --do_train \
    --do_eval \
    --do_predict \
	--evaluate_during_training \
    --evaluate_during_mlm \
    --mlm --mlm_probability 0.15 \
    --overwrite_output_dir


