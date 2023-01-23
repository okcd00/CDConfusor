task=wsw5_dcn_train_192

sighan_test=../data/cn/sighan15/sighan15_test.tsv
cctc_test=../data/cn/cctc/cctc_test.tsv
findoc_test=../data/cn/findoc/findoc_test.tsv

CUDA_VISIBLE_DEVICES=5 python predict.py \
  --model_file ./infer_model/$task/static_graph_params.pdmodel \
  --params_file ./infer_model/$task/static_graph_params.pdiparams \
  --test_file $sighan_test

CUDA_VISIBLE_DEVICES=5 python predict.py \
  --model_file ./infer_model/$task/static_graph_params.pdmodel \
  --params_file ./infer_model/$task/static_graph_params.pdiparams \
  --test_file $cctc_test

CUDA_VISIBLE_DEVICES=5 python predict.py \
  --model_file ./infer_model/$task/static_graph_params.pdmodel \
  --params_file ./infer_model/$task/static_graph_params.pdiparams \
  --test_file $findoc_test