# -*- coding: utf-8 -
import MeCab
import sqlite3
from tinydb import TinyDB, Query

class DataParser:
    def __init__(self, file=None, tinydb=None, sqlitedb=None):
        self.lines = []
        self.texts = []
        self.splited_texts = []
        self.labels = []

        if file is not None:
            self.load_from_file(file)
        elif tinydb is not None:
            self.load_from_tinydb(tinydb)
        elif sqlitedb is not None:
            self.load_from_sqlitedb(sqlitedb)

    def load_from_file(self, file):
        for line in open(file, 'r'):
            arr = line.split("\t")
            self.lines.append(arr)
            self.texts.append(arr[0].decode('utf-8'))
            self.splited_texts.append(self.split(arr[0].decode('utf-8')))
            self.labels.append(arr[1])

    def load_from_tinydb(self, tinydb):
        results = tinydb.all()
        for record in results:
            self.lines.append(record)
            self.texts.append(record['text'])
            self.splited_texts.append(self.split(record['text']))
            self.labels.append(record['label'])

    def load_from_sqlitedb(self, sqlitedb):
        sqlitedb.row_factory = sqlite3.Row
        cursor = sqlitedb.cursor()
        cursor.execute("select * from training_sets where label is not null")
        results = cursor.fetchall()

        for record in results:
            self.lines.append([record['text'], record['label']])
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
        return u" ".join(word_list[1:-1])
