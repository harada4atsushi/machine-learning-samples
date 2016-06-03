# -*- coding: utf-8 -
import pdb
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class FeatureBuilder:
    # UNIDIC_PATH = '/usr/local/lib/mecab/dic/unidic/'

    def __init__(self, dataset):
        self.dataset = dataset


    #def get_features_vector(self, fields):
    def get_features(self):
        count_vectorizer = CountVectorizer()
        feature_vectors = count_vectorizer.fit_transform(self.dataset.splited_texts)  # TODO
        self.vocabulary = count_vectorizer.get_feature_names()

        #pdb.set_trace()
        features_array = feature_vectors.toarray()
        features = np.c_[self.dataset.answer_id2s, features_array]
        features = np.c_[self.dataset.answer_id1s, features]
        return features

    # def __init__(self, db=None):
    #     if db is None:
    #       return
    #     self.lines = []
    #     self.texts = []
    #     self.category_ids = []
    #     self.splited_texts = []
    #     self.labels = []
    #     self.load_from_tinydb(db)
    #
    # def load_from_tinydb(self, tinydb):
    #     results = tinydb.all()
    #     for record in results:
    #         self.lines.append(record)
    #         self.texts.append(record['text'])
    #         self.category_ids.append(record['category_id'])
    #         self.splited_texts.append(self.split(record['text']))
    #         self.labels.append(record['label'])
    #
    # def split(self, text):
    #     #tagger = MeCab.Tagger("-d " + DataParser.UNIDIC_PATH)
    #     tagger = MeCab.Tagger("-u dict/custom.dic")
    #     text = text.encode("utf-8")
    #     node = tagger.parseToNode(text)
    #     word_list = []
    #     while node:
    #         pos = node.feature.split(",")[0]
    #         if pos in ["名詞", "動詞", "形容詞", "感動詞", "助動詞"]:
    #             lemma = node.feature.split(",")[6].decode("utf-8")
    #             if lemma == u"*":
    #                 lemma = node.surface.decode("utf-8")
    #             word_list.append(lemma)
    #         node = node.next
    #     return u" ".join(word_list)
