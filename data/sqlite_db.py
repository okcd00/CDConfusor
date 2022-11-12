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


class SQLiteDB(object):

    def __init__(self, db_path, n_samples=None, read_only=True, load_now=False):
        self.db_path = db_path
        self.n_samples = n_samples

        self.conn = None
        self.cursor = None
        self.saved_length = None
        self.writer_inited = False

        self.words, self.sindex = None, None

        self.samples = None
        if load_now:
            self.get_cursor()
            self.init_saved_length()
            self.cursor.close()
            self.conn = None
            self.cursor = None

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
        try:
            self.cursor.execute(
                'CREATE TABLE samples (word TEXT PRIMARY KEY NOT NULL, confusion TEXT, sindex INT)')
            self.conn.commit()
        except Exception as e:
            print(f"{e}")
        self.writer_inited = True

    def write(self, samples, sid_offset=0):
        self.init_writer()
        self.init_saved_length(force=True)

        if self.saved_length is not None:
            sid_offset = self.saved_length
        # execute
        for i, (word, confusion) in tqdm(enumerate(samples.items())):
            if isinstance(confusion, list):
                confusion = '\x01'.join(confusion)
            try:
                self.cursor.execute(
                    "insert into samples(word, confusion, sindex) values ('{}', '{}', {})".format(
                        word, confusion, i+sid_offset))
                # error:
                # sqlite3.DatabaseError: database disk image is malformed
                # https://blog.csdn.net/The_Time_Runner/article/details/106590571
            except Exception as e:
                if isinstance(e, sqlite3.IntegrityError):
                    print(f"Overwrite {i}-th word {word}.")
                    self.cursor.execute(
                        "UPDATE samples SET confusion='{}' WHERE word='{}'".format(confusion, word))
                else:
                    print(type(e), e)

        self.conn.commit()
        self.init_saved_length(force=True)

    def get_by_word(self, word):
        self.get_cursor()
        self.init_saved_length()

        try:
            if isinstance(word, list):
                # sid_str = " ".join([f"'_s'" for _s in sid])
                # sql = "SELECT data FROM samples WHERE sid IN ({}) ".format(sid_str)
                samples = [self.get_by_word(_s) for _s in word]
                return samples
            else:
                sql = "select confusion from samples where word = '{}' ".format(word)
                sample = self.cursor.execute(sql).fetchone()[0]
            # ret = self.cursor.execute(sql).fetchall()[0][0]
        except Exception as e:
            print(f"{e}\nError at:", sql)
            raise ValueError()
        return deepcopy(sample)

    def init_saved_length(self, force=False):
        if not force and self.sindex is not None:
            return
        self.get_cursor()
        word_index = self.cursor.execute(
            "select word, sindex from samples").fetchall()
        if self.n_samples:
            word_index = word_index[: self.n_samples]
        self.words, self.sindex = zip(*word_index)
        assert len(set(self.words)) == len(self.words)
        del word_index

        self.saved_length = len(self.words)
        # logging.warn(json.dumps(self.sids))

    def __getitem__(self, sindex):
        if self.cursor is None:
            self.get_cursor()
            self.init_saved_length()
        if isinstance(sindex, int):
            word = self.words[sindex]
        else:
            word = sindex
        return self.get_by_word(word)

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
    test_db_path = './tmp/confusionset_sighan.221110.db'

    dataset = SQLiteDB(
        test_db_path, 
        load_now=True)

    dataset.write({}, sid_offset=dataset.saved_length)

    print("Init SQLite Ends.", time.time() - start_time)
    print("The first sample is:", dataset[0])
    # test_multiprocessing(dataset)
