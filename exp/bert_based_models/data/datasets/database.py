# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : Database.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-04-12
#   desc     :
#   refer    : from utie.data.training_dbs import Database
# ==========================================================================
from bbcm.utils.custom_encodings import *
import logging


def write_data(stream, text, encoding='unicode'):
    # once write **text** into a file, need to know
    # the basestring for py2 and py3 are different
    if encoding in ['unicode', 'u']:
        stream.write(convert_to_unicode(text))
    elif encoding in ['bytes', 'utf-8', 'b']:
        stream.write(convert_to_bytes(text))
    else:  # others
        stream.write(text)


def write_one_sample_per_file(answers, folder, compress=False):
    register = ['{}'.format(s['info']['sid']) for i, s in enumerate(answers)]
    if not os.path.exists(folder):
        os.mkdir(folder)
    save_register(register=register, folder=folder)
    for s in answers:
        file_path = path_join(folder, s['info']['sid'])
        json_dump(s, path=file_path, encrypt=compress)


def append_write_one_sample_per_file(answers, folder):
    assert os.path.isdir(folder), 'folder should exist if you want to append to existing dataset'
    sids = load_register(folder)
    conflict_sids = set(sids).intersection([s['info']['sid'] for s in answers])
    assert not conflict_sids, 'some sids already exist: {}'.format(list(conflict_sids)[:10])
    new_register = ['{}'.format(s['info']['sid']) for i, s in enumerate(answers)]
    save_register(register=new_register, folder=folder, append=True)

    f = 0
    for s in answers:
        try:
            sid_str = convert_to_unicode(s['info']['sid'])
            json_dump(obj_=s, path=path_join(folder, sid_str), encrypt=False)
        except OverflowError:
            logging.warning('{} save error'.format(s['info']['sid']))
            f += 1
            if f > 30:
                break


def save_register(register, folder, append=False):
    # saving bytes is faster, but here is 'append' without 'b'
    # remain storing as source text
    if append:
        with open(path_join(folder, 'register'), 'a') as fw:
            # 'w' for py2 and py3 is different
            write_data(fw, '\n')
            write_data(fw, '\n'.join(register))
    with open(path_join(folder, 'register'), 'w') as fw:
        # 'w' for py2 and py3 is different
        write_data(fw, '\n'.join(register))


def load_register(folder, n_samples=None):
    sids = []
    # loading bytes is faster (append with 'ab+', loading with 'rb')
    with open(path_join(folder, 'register'), 'r') as fr:
        if n_samples is None:
            # faster list-construction
            sids = [line.strip().split(',')[-1] for line in fr]
        else:  # custom n_samples is usually small,
            for line in fr:  # list-appending will be faster.
                sid = line.strip().split(',')[-1]
                sids.append(sid)
                if n_samples is not None:
                    if len(sids) >= n_samples:
                        break
    return sids


class FolderDB(object):
    """
    一个sample写到一个文件里，一个DB就是一个文件夹，只能按照文件名进行索引
    NEW: 也可以按下标遍历
    """

    def __init__(self, folder, n_samples=None, load_now=False):
        self.folder = folder
        self.compress = False
        self.n_samples = n_samples
        self.sids = None
        if load_now:
            self.load_register()

    def write(self, samples):
        # samples: a list of sample-items, with 'sid' key
        write_one_sample_per_file(samples, self.folder)

    def append(self, samples):
        # samples: a list of sample-items, with 'sid' key
        append_write_one_sample_per_file(samples, self.folder)

    def get_by_sid(self, sid):
        file_path = path_join(self.folder, sid)
        sample = json.load(open(file_path))
        return sample

    def check_sids(self):
        return len(self.sids) == len(set(self.sids))

    def load_register(self):
        if self.sids is not None:
            return
        sids = load_register(self.folder)
        if self.n_samples:
            sids = sids[: self.n_samples]
        self.sids = sids
        assert self.check_sids(), 'exist duplicated sids'

    def remove_sids(self, sids, remove_file=False):
        for _, sid in enumerate(sids):
            self.sids.remove(sid)
            if remove_file:
                file_path = path_join(self.folder, sid)
                if os.path.exists(file_path):
                    os.remove(file_path)
        self.save_register()

    def save_register(self):
        assert self.check_sids(), 'exist duplicated sids'
        save_register(register=self.sids, folder=self.folder)

    def next(self):
        if self.n == self.__len__():
            raise StopIteration
        n = self.n
        self.n += 1
        return self[n]

    def __next__(self):
        return self.next()

    def __len__(self):
        self.load_register()
        return len(self.sids)

    def __iter__(self):
        self.n = 0
        return self

    def __getitem__(self, index):
        self.load_register()
        sid = self.sids[index]
        return self.get_by_sid(sid)

    @property
    def all_samples(self):
        """return all samples in this dataset"""
        return [self[i] for i in range(len(self))]


class CFolderDB(FolderDB):
    """A json-encrypted FolderDB"""
    def write(self, samples):
        write_one_sample_per_file(samples, self.folder, compress=True)

    def get_by_sid(self, sid):
        file_path = path_join(self.folder, sid)
        sample = json_load(path=file_path, mode='r', decrypt=True)
        # sample = json.loads(zlib.decompress(open(file_path, 'rb').read()).decode('utf-8'))
        return sample


class Database(object):
    """
    A unified wrapper for OneFileDB, FolderDB
    Now it is only used for FolderDB
    """

    def __init__(self, path, n_samples=None, load_now=False):
        db = FolderDB(path, n_samples=n_samples, load_now=load_now)
        self.db = db
        self.sids = db.sids

    def write(self, samples):
        return self.db.write(samples)

    def get_by_sid(self, sid):
        return self.db.get_by_sid(sid)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.sl(item)
        return self.db[item]

    def sl(self, key):
        start, stop, step = key.indices(len(self))
        for i in range(start, stop, step):
            yield self.db[i]

    def __len__(self):
        return self.db.__len__()

    def __iter__(self):
        return self.db.__iter__()

    def next(self):
        return self.db.next()

    @property
    def all_samples(self):
        return self.db.all_samples


if __name__ == "__main__":
    pass
