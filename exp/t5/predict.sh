TEST_PATH=../data/cn/sighan15/sighan15_test.tsv
MODEL_PATH=./t5_models/ws_train_model_192/

CUDA_VISIBLE_DEVICES=1 python infer.py \
  --save_dir $MODEL_PATH \
  --test_path $TEST_PATH \
  --max_len 192
