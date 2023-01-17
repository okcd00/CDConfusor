task=ws_dcn_train_192

python export_model.py \
  --params_path checkpoints/$task/best_model.pdparams \
  --output_path infer_model/$task/static_graph_params
