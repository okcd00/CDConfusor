TEST_PATH=../data/cn/sighan15/sighan15_test.tsv
MODEL_PATH=./output/ws_train_model

CUDA_VISIBLE_DEVICES=1,2 python infer.py \
  --save_dir $MODEL_PATH \
  --test_path $TEST_PATH \
  --max_len 128
