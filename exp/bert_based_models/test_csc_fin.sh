export PYTHONPATH=.
python tools/train_csc.py \
    --config_file "fin/cdmac_findoc_pret.yml"
    --opts mode=['test']
