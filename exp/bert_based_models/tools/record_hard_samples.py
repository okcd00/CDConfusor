
from tqdm import tqdm
from tools.inference import load_model_directly
from bbcm.data.datasets.csc import CscDataset
from bbcm.data.datasets.sqlite_db import SQLiteDB


def record_hard_samples(samples, record_db_path):
    dir_path = '/data/chendian/bbcm_checkpoints/cdmac_autodoc_cor_3rd_220922/'
    ckpt_file = dir_path + '/epoch=00_train_loss_epoch=0.1338_train_det_f1_epoch=0.9004_train_cor_f1_epoch=0.1794.ckpt'
    config_file = dir_path + '/config.yml'

    record_samples = []
    record_db = SQLiteDB(record_db_path, load_now=True)

    model = load_model_directly(
        ckpt_file=ckpt_file, 
        config_file=config_file)

    err, hit = 0, 0
    start_position = 0
    sid_offset = len(record_db) // block_size
    block_size = 32
    for i in tqdm(range(start_position, len(samples) // block_size + 1)):
        _samples = [samples[_i] for _i in range(block_size*i, block_size*(i+1))]
        src = [items[0] for items in _samples if items is not None]
        dest = [items[1] for items in _samples if items is not None]
        res = model.predict(src)
        for p, t in zip(res, dest):
            if p != t:
                record_samples.append(f"{p}\t{t}")
                err += 1
            else:
                hit += 1 

        if len(record_samples) >= 100:
            record_db.write(record_samples, sid_offset=sid_offset)
            sid_offset += len(record_samples)
            record_samples = []
    else:
        if record_samples:
            record_db.write(record_samples, sid_offset=sid_offset)
            sid_offset += len(record_samples)
        record_samples = []

    print(err, hit, err/(hit+err))


if __name__ == "__main__":
    findoc_train_db = "/data/chendian/cleaned_findoc_samples/findoc_samples_cand301.220803.fixed2.db"
    findoc_test_db = '/data/chendian/cleaned_findoc_samples/autodoc_test.220424.db'
    dataset = CscDataset(fp=findoc_train_db)
    record_hard_samples(
        dataset, record_db_path='/data/chendian/cleaned_findoc_samples/autodoc_train_failed.221010.db')
