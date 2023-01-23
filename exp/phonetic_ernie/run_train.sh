task=wsw5_dcn_train_192

# TRAIN_DIR=../data/cn/Wang271k_augc
# TEST_PATH=../data/cn/rw/rw_test.dcn.txt
# --extra_test_ds_file $TEST_PATH

TRAIN_DIR=../data/cn/Wang271k_augw

CUDA_VISIBLE_DEVICES=5 python train.py \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --max_seq_length 192 \
  --model_name_or_path ernie-1.0 \
  --logging_steps 500 \
  --save_steps 50000 \
  --output_dir ./checkpoints/$task/ \
  --extra_train_ds_dir $TRAIN_DIR