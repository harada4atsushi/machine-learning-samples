# -*- coding: utf-8 -
import MeCab
import sqlite3
from tinydb import TinyDB, Query

class DataParser:
    def __init__(self, db=None):
        self.lines = []
        self.texts = []
        self.splited_texts = []
        self.labels = []
        self.load_from_tinydb(db)

    def load_from_tinydb(self, tinydb):
        results = tinydb.all()
        for record in results:
            self.lines.append(record)
            self.texts.append(record['text'])
            self.splited_texts.append(self.split(record['text']))
            self.labels.append(record['label'])

    def split(self, text):
        tagger = MeCab.Tagger()
        text = text.encode("utf-8")
        node = tagger.parseToNode(text)
        word_list = []
        while node:
            pos = node.feature.split(",")[0]
            if pos in ["名詞", "動詞", "形容詞"]:
                lemma = node.feature.split(",")[6].decode("utf-8")
                if lemma == u"*":
                    lemma = node.surface.decode("utf-8")
                word_list.append(lemma)
            node = node.next
        return u" ".join(word_list)
