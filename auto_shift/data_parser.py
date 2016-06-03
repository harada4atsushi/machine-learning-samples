# -*- coding: utf-8 -
import MeCab
import sqlite3
from tinydb import TinyDB, Query

class DataParser:
    UNIDIC_PATH = '/usr/local/lib/mecab/dic/unidic/'

    def __init__(self, db=None):
        if db is None:
          return
        self.lines = []
        self.texts = []
        self.category_ids = []
        self.splited_texts = []
        self.labels = []
        self.load_from_tinydb(db)

    def load_from_tinydb(self, tinydb):
        results = tinydb.all()
        for record in results:
            self.lines.append(record)
            self.texts.append(record['text'])
            self.category_ids.append(record['category_id'])
            self.splited_texts.append(self.split(record['text']))
            self.labels.append(record['label'])

    def split(self, text):
        #tagger = MeCab.Tagger("-d " + DataParser.UNIDIC_PATH)
        tagger = MeCab.Tagger("-u dict/custom.dic")
        text = text.encode("utf-8")
        node = tagger.parseToNode(text)
        word_list = []
        while node:
            pos = node.feature.split(",")[0]
            if pos in ["名詞", "動詞", "形容詞", "感動詞", "助動詞"]:
                lemma = node.feature.split(",")[6].decode("utf-8")
                if lemma == u"*":
                    lemma = node.surface.decode("utf-8")
                word_list.append(lemma)
            node = node.next
        return u" ".join(word_list)
