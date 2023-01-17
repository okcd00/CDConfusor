# TRAIN_PATH=../data/cn/Wang271k/dcn_train.tsv

TRAIN_PATH=../data/cn/Wang271k_augc/dcn_train.augc.tsv
TEST_PATH=../data/cn/rw/rw_test.tsv

MODEL_PATH=./t5_models/wsc_train_model_192
MODEL_NAME=Langboat/mengzi-t5-base

# batch_size = 8 will OOM for 192-text-len on 12G GPU

CUDA_VISIBLE_DEVICES=6 python train.py \
  --model_name_or_path $MODEL_NAME \
  --train_path $TRAIN_PATH \
  --test_path $TEST_PATH \
  --logging_steps 500 \
  --warmup_steps 500 \
  --eval_steps 500 \
  --epochs 5 \
  --batch_size 4 \
  --max_len 192 \
  --save_dir $MODEL_PATH \
  --do_train \
  --do_eval
