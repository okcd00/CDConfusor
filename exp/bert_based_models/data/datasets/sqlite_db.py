# coding=utf8

from __future__ import unicode_literals
from six import iteritems

import os
import time
import sqlite3
from copy import deepcopy
from tqdm import tqdm
from queue import Queue
from threading import Thread


import sys
PY2 = int(sys.version[0]) == 2

if PY2:
    text_type = unicode  # noqa
    binary_type = str
    string_types = (str, unicode)  # noqa
    unicode = unicode  # noqa
    basestring = basestring  # noqa
else:
    text_type = str
    binary_type = bytes
    string_types = (str,)
    unicode = str
    basestring = (str, bytes)


import json
import sqlite3
import numpy as np


class TrainDBBase(object):
    """
    
    An immutable dataset once write.

    @staticmethod
    def random_ints(n):
        # return n random ints that are distinct
        assert n < 10**9, 'Too many distinct numbers asked.'
        row_randoms = np.random.randint(0, np.iinfo(np.int64).max, 2*n)
        uniques = np.unique(row_randoms)
        while len(uniques) < n:
            r = np.random.randint(0, np.iinfo(np.int64).max, 2*n)
            uniques = np.unique(np.stack([uniques, r]))
        return uniques[:n]

    def add_sindex(self, labels):
        # deprecated
        indexes = random_ints(len(labels))
        for i, l in enumerate(labels):
            l['info']['sindex'] = indexes[i]
        self.sindex_to_sid_dict = {
            s['info']['sindex']: s['info']['sid'] for s in labels}
        del indexes
        return labels
    """

    def write(self, samples):
        """save samples"""
        raise NotImplementedError()

    def get_by_sid(self, sid):
        """get sample by sid"""
        raise NotImplementedError()

    def sindex_to_sid(self, sindex):
        """ return sid given sindex"""
        raise NotImplementedError()

    def __getitem__(self, item):
        """ get sample by index in dataset"""
        raise NotImplementedError()

    def __len__(self):
        """return the number of samples in this dataset"""
        raise NotImplementedError()

    def __iter__(self):
        self.n = 0
        return self

    def next(self):
        if self.n == self.__len__():
            raise StopIteration
        n = self.n
        self.n += 1
        return self[n]

    def __next__(self):
        return self.next()

    @property
    def all_samples(self):
        """return all samples in this dataset"""
        return [self[i] for i in range(len(self))]


class SQLiteDBExample(TrainDBBase):

    def __init__(self, db_path, n_samples=None, read_only=True, load_now=False):
        self.samples = None
        self.n_samples = n_samples
        self.sids = None
        self.sid_to_sample = None
        self.db_path = db_path
        self.sindexes = None
        self.sindex_to_sid_dict =None
        self.sid_to_sindex_dict =None
        self.conn = None
        self.cursor = None
        self.saved_length = None
        self.pure_text_samples = True  # True for CSC tasks.
        if load_now:
            self.get_cursor()
            self.load_sid_sindex()
            self.cursor.close()
            self.conn = None
            self.cursor = None

    def get_cursor(self):
        if self.cursor is not None:
            return

        conn = sqlite3.connect(  # WAL mode for multi-processing
            self.db_path, 
            isolation_level=None,  # https://www.cnblogs.com/Gaimo/p/16098045.html
            check_same_thread=False,  # https://codeantenna.com/a/VNKPkxjiFx
            timeout=3)

        conn.row_factory = sqlite3.Row
        self.conn = conn
        self.cursor = conn.cursor()
        # WAL mode for multi-processing
        self.cursor.execute('PRAGMA journal_mode=wal')  # https://www.coder.work/article/2441365
        self.cursor.execute('PRAGMA synchronous=OFF')  # 

    def remove_file(self):
        import os
        os.remove(self.db_path)

    def write(self, samples):
        self.get_cursor()
        # if os.path.exists(self.db_path):
        #     logging.warn('removing the existing dataset')
        #     os.remove(self.db_path)

        # create table
        self.cursor.execute(
            'CREATE TABLE samples (sid TEXT PRIMARY KEY NOT NULL, data TEXT, sindex INT)')
        self.conn.commit()

        # execute
        if self.pure_text_samples:
            for i, s in tqdm(enumerate(samples)):
                sid = unicode(f'{i}')
                s = unicode(s.strip().replace("'", "''"))
                try:
                    self.cursor.execute(
                        "insert into samples(sid, data, sindex) values ('{}', '{}', {})".format(sid, s, i))
                    # error:
                    # sqlite3.DatabaseError: database disk image is malformed
                    # https://blog.csdn.net/The_Time_Runner/article/details/106590571
                except Exception as e:
                    print(e)
                    print(sid)
                    print(s)
                    print(i)
        else:
            # pre-processing
            for s in tqdm(samples):
                s['info']['sid'] = unicode(s['info']['sid'])
                sample_dict = {s['info']['sid']: json.dumps(s) for s in samples}

            i = 0
            for sid, s in tqdm(iteritems(sample_dict)):
                self.cursor.execute(
                    "insert into samples(sid, data, sindex) values ('{}', '{}', {})".format(sid, s, i))
                i += 1

        self.conn.commit()

    def get_by_sid(self, sid):
        self.load_sid_sindex()
        sql = "select data from samples where sid = '{}' ".format(sid)
        try:
            ret = self.cursor.execute(sql).fetchone()[0]
            # ret = self.cursor.execute(sql).fetchall()[0][0]
        except Exception as e:
            print(f"{e}\nError at:", sql)
            raise ValueError()
        if self.pure_text_samples:
            sample = ret
        else:
            sample = json.loads(ret)
            sample['info']['sindex'] = self.sid_to_sindex_dict[sid]
        # time.sleep(0.05)
        return sample

    def load_sid_sindex(self):
        if self.sids is not None:
            return
        self.get_cursor()
        sid_sindex = self.cursor.execute(
            "select sid, sindex from samples").fetchall()
        if self.n_samples:
            sid_sindex = sid_sindex[: self.n_samples]
        self.sids, self.sindexes = zip(*sid_sindex)
        assert len(set(self.sids)) == len(self.sids)
        assert len(set(self.sindexes)) == len(self.sindexes)
        # logging.warn(json.dumps(self.sindexes))
        # logging.warn(json.dumps(self.sids))

        self.sid_to_sindex_dict = {sid: sindex for sid, sindex in sid_sindex}
        self.sindex_to_sid_dict = {sindex: sid for sid, sindex in sid_sindex}
        # logging.warning(f"loaded {len(self.sids)} samples.")
        self.saved_length = len(self.sids)
    
    def sindex_to_sid(self, sindex):
        self.get_cursor()
        self.load_sid_sindex()
        return self.sindex_to_sid_dict[sindex]

    def __getitem__(self, item):
        self.get_cursor()
        self.load_sid_sindex()

        sid = self.sids[item]
        return self.get_by_sid(sid)

    def __len__(self):
        return self.saved_length


class SQLiteDB(TrainDBBase):

    def __init__(self, db_path, n_samples=None, read_only=True, load_now=False):
        self.db_path = db_path
        self.n_samples = n_samples

        self.sids = None
        self.conn = None
        self.cursor = None
        self.saved_length = None
        self.writer_inited = False

        self.samples = None
        if load_now:
            self.get_cursor()
            self.init_saved_length()
            self.cursor.close()
            self.conn = None
            self.cursor = None

    def get_cursor(self):
        if self.cursor is not None:
            return

        conn = sqlite3.connect(  # WAL mode for multi-processing
            self.db_path, 
            isolation_level=None,  # https://www.cnblogs.com/Gaimo/p/16098045.html
            check_same_thread=False,  # https://codeantenna.com/a/VNKPkxjiFx
            timeout=2.5)

        conn.row_factory = sqlite3.Row
        self.conn = conn
        self.cursor = conn.cursor()
        
        # WAL mode for multi-processing
        self.cursor.execute('PRAGMA journal_mode=wal')  # https://www.coder.work/article/2441365
        self.cursor.execute('PRAGMA synchronous=OFF')  # 

    def remove_file(self):
        import os
        os.remove(self.db_path)

    def init_writer(self):
        if self.writer_inited:
            return

        self.get_cursor()
        # if os.path.exists(self.db_path):
        #     logging.warn('removing the existing dataset')
        #     os.remove(self.db_path)

        # create table
        self.cursor.execute(
            'CREATE TABLE samples (sid TEXT PRIMARY KEY NOT NULL, data TEXT, sindex INT)')
        self.conn.commit()
        self.writer_inited = True

    def write(self, samples, sid_offset=0):
        self.init_writer()

        # execute
        for i, s in tqdm(enumerate(samples)):
            sid = unicode(f'{i + sid_offset}')
            s = unicode(s.strip().replace("'", "''"))
            try:
                self.cursor.execute(
                    "insert into samples(sid, data, sindex) values ('{}', '{}', {})".format(
                        sid, s, i+sid_offset))
                # error:
                # sqlite3.DatabaseError: database disk image is malformed
                # https://blog.csdn.net/The_Time_Runner/article/details/106590571
            except Exception as e:
                print(e)
                print(sid)
                print(s)
                print(i)

        self.conn.commit()

    def get_by_sid(self, sid):
        self.get_cursor()
        self.init_saved_length()

        try:
            if isinstance(sid, list):
                # sid_str = " ".join([f"'_s'" for _s in sid])
                # sql = "SELECT data FROM samples WHERE sid IN ({}) ".format(sid_str)
                samples = [self.get_by_sid(_s) for _s in sid]
                return samples
            else:
                sql = "select data from samples where sid = '{}' ".format(sid)
                sample = self.cursor.execute(sql).fetchone()[0]
            # ret = self.cursor.execute(sql).fetchall()[0][0]
        except Exception as e:
            print(f"{e}\nError at:", sql)
            raise ValueError()
        return deepcopy(sample)

    def init_saved_length(self):
        if self.sids is not None:
            return
        self.get_cursor()
        sid_sindex = self.cursor.execute(
            "select sid, sindex from samples").fetchall()
        if self.n_samples:
            sid_sindex = sid_sindex[: self.n_samples]
        self.sids, _ = zip(*sid_sindex)
        assert len(set(self.sids)) == len(self.sids)
        del sid_sindex

        self.saved_length = len(self.sids)
        # logging.warn(json.dumps(self.sids))

    def __getitem__(self, item):
        if self.cursor is None:
            self.get_cursor()
            self.init_saved_length()
        if item >= self.saved_length:
            return None
        sid = self.sids[item]
        return self.get_by_sid(sid)

    def __len__(self):
        return self.saved_length


def write_existed_samples(txt_path, db_path):
    db = SQLiteDB(db_path, load_now=False)
    db.remove_file()
    samples = open(txt_path, 'r')
    db.write(samples)


def single_thread_load_samples(_id, dataset):
    print(f"init {_id}-th subprocess.")
    total_length = 0
    for i in range(1000):
        res = dataset[i]
        total_length += res.__len__()
    # print("Loaded {} charaters.".format(total_length))

def test_multiprocessing(dataset):
    import multiprocessing
    print('Run the main process (%s).' % (os.getpid()))

    i = 0
    n_cores = 32
    for i in range(n_cores):
        p = multiprocessing.Process(
            target=single_thread_load_samples,
            args=(i, dataset))
        p.start()
    print('Waiting for all subprocesses done ...')


if __name__ == "__main__":
    import time
    start_time = time.time()

    # train_path = '/data/chendian/cleaned_findoc_samples/findoc_samples_cand301.220803.fixed2.txt'
    # train_db_path = '/data/chendian/cleaned_findoc_samples/findoc_samples_cand301.220803.fixed2.db'
    # write_existed_samples(train_path, train_db_path)

    test_path = '/home/user/cleaned_findoc_samples/autodoc_test.220424.txt'
    test_db_path = '/home/user/cleaned_findoc_samples/autodoc_test.220424.db'
    # write_existed_samples(test_path, test_db_path)

    dataset = SQLiteDB(
        test_db_path, 
        load_now=True)
    print("Init SQLite Ends.", time.time() - start_time)
    print("The first sample is:", dataset[0])
    # test_multiprocessing(dataset)
