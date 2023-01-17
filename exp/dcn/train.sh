set -v
set -e


TRAIN_FILE=../data/cn/Wang271k_augc/dcn_train.augc.dcn.txt
TEST_FILE=../data/cn/rw/rw_test.dcn.txt

BERT_MODEL=../pretrained_models/chinese-roberta-wwm-ext/
OUTPUT_DIR=dcn_models/wsc_train_model_192/

# for DCN_augc, 17007 steps/epoch
# for DCN_train, when batch_size=8, 8794 steps/epoch; or bs=4, 17587 steps/epoch

SAVE_STEPS=17007  
SEED=1038
LR=5e-5
SAVE_TOTAL_LIMIT=5
MAX_LENGTH=192  # 128 for sighan, 192 for dcn-train
BATCH_SIZE=4  # 8 will OOM for 192-text-len on 12G GPU
NUM_EPOCHS=5


CUDA_VISIBLE_DEVICES=1,2,3,4 python train_DCN.py \
    --output_dir $OUTPUT_DIR \
	--learning_rate $LR  \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --model_type=bert \
    --model_name_or_path=$BERT_MODEL \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
	--logging_steps $SAVE_STEPS \
	--save_total_limit $SAVE_TOTAL_LIMIT \
	--block_size $MAX_LENGTH \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --do_train \
    --do_eval \
    --do_predict \
	--evaluate_during_training \
    --seed $SEED \
    --mlm \
	--mlm_probability 0.15

