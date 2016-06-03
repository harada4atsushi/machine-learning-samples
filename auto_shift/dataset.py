# -*- coding: utf-8 -
import MeCab
import numpy as np
from tinydb import TinyDB, Query

class Dataset:
    UNIDIC_PATH = '/usr/local/lib/mecab/dic/unidic/'

    def __init__(self, db_path=None):
        if db_path is None:
          return
        #self.day_of_week = np.array([])
        self.features = np.empty((0,2), int)
        self.attend = np.array([])
        self.load_from_tinydb(db_path)

    def load_from_tinydb(self, db_path):
        db = TinyDB(db_path)
        results = db.all()
        for record in results:
            self.features = np.append(self.features, np.array([[record['day_of_week'], record['day_of_week']]]), axis=0)
            self.attend = np.append(self.attend, record['attend'])

        #self.day_of_week = self.day_of_week
        #self.attend = self.attend
