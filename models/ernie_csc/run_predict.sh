task=ws_dcn_train

cctc_test=../data/cn/CCTC/cctc_test.txt
sighan_test=../SIGHAN15_test/sighan15_test.txt
findoc_test=../data/cn/FinDoc_test/findoc_collect.txt

CUDA_VISIBLE_DEVICES=7 python predict.py \
  --model_file ./infer_model/$task/static_graph_params.pdmodel \
  --params_file ./infer_model/$task/static_graph_params.pdiparams \
  --test_file $findoc_test