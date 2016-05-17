# -*- coding: utf-8 -
import numpy as np
import pandas as pd
import sys
from tinydb import TinyDB, Query
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn import cross_validation
from data_parser import DataParser
from plotter import Plotter
from predicter import Predicter

# featureを準備する
db = TinyDB('db/yes_no.json')
data_parser = DataParser(db=db)
count_vectorizer = CountVectorizer()
feature_vectors = count_vectorizer.fit_transform(data_parser.splited_texts)
vocabulary = count_vectorizer.get_feature_names()

# 学習する
svm_tuned_parameters = [
    {
        'kernel': ['linear'],
        # 'gamma': [2**n for n in range(-5, 3)],
        # 'C': [2**n for n in range(-5, 8)]
        'gamma': [2**n for n in range(-15, 3)],
        'C': [2**n for n in range(-5, 15)]
    }
]
gscv = GridSearchCV(
    SVC(),
    svm_tuned_parameters,
    cv = 2,
    n_jobs = 1,
    verbose = 3
)

gscv.fit(feature_vectors, data_parser.labels)
svm_model = gscv.best_estimator_

print svm_model  # 高パフォーマンスの学習モデル
print gscv.best_params_  # 高パフォーマンスのパラメータ(gamma,Cの値)

# Xtrain = pd.Series(["はい", "いいえ", "うん", "いや", "ストップ"])
# input_texts = []
# for input_text in Xtrain:
#   input_texts.append(data_parser.split(input_text.decode('utf-8')))
#
# count_vectorizer = CountVectorizer(
#     vocabulary=vocabulary
# )
# feature_vectors = count_vectorizer.fit_transform(input_texts)

results = svm_model.predict(feature_vectors)
#print svm_model.score(data_parser.labels, feature_vectors)

for i in range(len(results)):
    print results[i]
    print data_parser.texts[i]
    print

scores = cross_validation.cross_val_score(svm_model, feature_vectors, data_parser.labels, cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 学習済みモデルをdumpする
#joblib.dump(svm_model, "models/svm_model")
#joblib.dump(vocabulary, 'vocabulary/vocabulary.pkl')

# 学習曲線を描画する
# plotter = Plotter()
# plotter.plot(svm_model, feature_vectors, data_parser.labels)

# 分類させる
# sys.argv.pop(0)
# predicter = Predicter()
# result = predicter.predict(svm_model, data_parser, sys.argv, vocabulary)
# print result
