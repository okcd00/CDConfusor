set -v
set -e

# --------------
# | data files |
# --------------

TRAIN_FILE=../data/cn/Wang271k/dcn_train.mp.dcn.txt  # W271k: 281381 | mp544192
# TRAIN_FILE=../data/tmp/findoc_train.230329.dcn.txt  # W271k+cctc+fd2: 279386 | mp558772
# TRAIN_FILE=../data/cn/findoc/findoc_test.v1.dcn.txt  # fd1: 368 
# TRAIN_FILE=../data/cn/findoc/findoc_test.v2.dcn.txt  # fd2: 6569
# TRAIN_FILE=../data/cn/cctc/cctc_train.dcn.txt  # cctc: 646 
# TRAIN_FILE=../data/cn/rw/rw_test.dcn.txt  # rw=cctc+fd1: 1089

# TEST_FILE=../data/cn/findoc/findoc_test.v1.dcn.txt  # fd1: 368
# TEST_FILE=../data/cn/findoc/findoc_test.v2.dcn.txt  # fd2: 6569
# TEST_FILE=../data/cn/cctc/cctc_test.dcn.txt  # cctc: 721
TEST_FILE=../data/cn/rw/rw_test.dcn.txt
SIGHAN_TEST_FILE=../data/cn/sighan15/sighan15_test.dcn.txt

# --model_name_or_path $WARMUP_DIR (modified roberta vocab)
BERT_MODEL=../pretrained_models/chinese-roberta-wwm-ext/
WARMUP_DIR=cd_models/findoc_finetuned_w271k.mp/checkpoint-272096ep2/
OUTPUT_DIR=cd_models/findoc_finetuned_w271k.mp/

# for DCN_augc, 17007 steps/epoch, for 6GPU DCN_augw, 11338 steps/epoch
# for DCN_train, when batch_size=8, 8794 steps/epoch; or bs=4, 17587 steps/epoch

# fd_230324: 17813 steps/epoch on 4GPU (285013 samples)
# fd_230328: 25328 steps/epoch on 4GPU
# W271k: 17587 steps/epoch on 4GPU
# rw_v1: 273 steps/epoch
# fd_v2: 1688 steps/epoch

SAVE_STEPS=136048
WARMUP_STEPS=0  # $SAVE_STEPS
SEED=1038
LR=5e-5
SAVE_TOTAL_LIMIT=5
MAX_LENGTH=192  # 128 for sighan, 192 for dcn-train
BATCH_SIZE=4  # 8 will OOM for 192-text-len on 12G GPU
NUM_EPOCHS=3


mkdir -p $OUTPUT_DIR
cp ./train_temp.sh $OUTPUT_DIR/train_temp.sh


CUDA_VISIBLE_DEVICES=5  python train_DCN.py \
    --output_dir $OUTPUT_DIR \
	--learning_rate $LR  \
    --warmup_steps $WARMUP_STEPS \
    --min_lr 1e-6 \
    --per_device_train_batch_size $BATCH_SIZE \
    --model_type=bert \
    --model_name_or_path $WARMUP_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
	--logging_steps $SAVE_STEPS \
	--save_total_limit $SAVE_TOTAL_LIMIT \
	--block_size $MAX_LENGTH \
    --train_data_file $TRAIN_FILE \
    --eval_data_file $SIGHAN_TEST_FILE \
    --test_data_file $TEST_FILE \
    --do_train \
    --do_eval \
    --do_predict \
	--evaluate_during_training \
    --seed $SEED \
    --mlm --mlm_probability 0.15 \
    --overwrite_output_dir


