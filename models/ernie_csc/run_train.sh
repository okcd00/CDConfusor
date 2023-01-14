export CUDA_VISIBLE_DEVICES=7
task=ws_dcn_train_192

python train.py \
  --epochs 5 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --max_seq_length 192 \
  --model_name_or_path ernie-1.0 \
  --logging_steps 100 \
  --save_steps 10000 \
  --output_dir ./checkpoints/$task/ \
  --extra_train_ds_dir ../data/cn/Wang271k 