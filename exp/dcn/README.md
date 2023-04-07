## Important Notes and Citation

> + effective scripts and model codes in this dir are forked from @destwang/[DCN](https://github.com/destwang/DCN).
> + updated and modified by @okcd00 for further research.
> + origin readme file [here](readme.origin.md)


## Usage 
+ Step I: prepare all your datasets
+ Step II: run the command `train_temp.sh`


```bash
set -v
set -e

# --------------
# | data files |
# --------------

TRAIN_FILE=../data/fin/findoc_train.230406.dcn.txt  # findoc-corpus: 7680964
TEST_FILE=../data/cn/rw/rw_test.dcn.txt
SIGHAN_TEST_FILE=../data/cn/sighan15/sighan15_test.dcn.txt

# --model_name_or_path $WARMUP_DIR (modified roberta vocab)
BERT_MODEL=../pretrained_models/chinese-roberta-wwm-ext/
WARMUP_DIR=cd_models/findoc_finetuned_w271k/
OUTPUT_DIR=cd_models/findoc_finetuned_230406/

mkdir -p $OUTPUT_DIR
cp ./train_temp.sh $OUTPUT_DIR/train_temp.sh

# --------------
# | configs    |
# --------------

# model hyper-parameters
SEED=1038
LR=5e-5
MIN_LR=2e-6
SAVE_TOTAL_LIMIT=5
MAX_LENGTH=192  # 128 for sighan, 192 for dcn-train
BATCH_SIZE=4  # 4 for 192-text-len on 12G GPU, 12 for 192-text-len on 24G GPU
NUM_EPOCHS=10
WARMUP_STEPS=0

# custom settings for training
GPUS=0,1,2,3
DATASET_LINES=7680964
GPU_COUNT=$(echo $GPUS | tr -cd "[0-9]" | wc -c)
SAVE_STEPS=$(echo "$DATASET_LINES/$GPU_COUNT/$BATCH_SIZE+1" | bc)
echo "In this run, SAVE_STEPS is $SAVE_STEPS"

# --------------
# | RUN !!!    |
# --------------

# run the command
WANDB_PROJECT=findoc_csc_finetuning CUDA_VISIBLE_DEVICES=$GPUS python train_DCN.py \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LR  \
    --min_lr $MIN_LR \
    --warmup_steps $WARMUP_STEPS \
    --model_type=bert \
    --model_name_or_path $BERT_MODEL \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
	--logging_steps $SAVE_STEPS \
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
```