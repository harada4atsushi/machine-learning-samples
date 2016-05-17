# -*- coding: utf-8 -
import numpy as np
import sqlite3
from tinydb import TinyDB, Query
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from data_parser import DataParser
#from plotter import Plotter

# featureを準備する
#db = TinyDB('../db/development.json')
#data_parser = DataParser(tinydb=db)
sqlitedb = sqlite3.connect("../db/development.sqlite3")
data_parser = DataParser(sqlitedb=sqlitedb)
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
    cv = 2,  # k-fold cross-validation トレーニングセットとクロスバリデーションセットに分割したものをcv回クロスバリデーションする
    n_jobs = 1,
    verbose = 3
)

gscv.fit(feature_vectors, data_parser.labels)
svm_model = gscv.best_estimator_

print svm_model  # 高パフォーマンスの学習モデル
print gscv.best_params_  # 高パフォーマンスのパラメータ(gamma,Cの値)

# 学習済みモデルをdumpする
joblib.dump(svm_model, "models/svm_model")
joblib.dump(vocabulary, 'vocabulary/vocabulary.pkl')

# 学習曲線を描画する
# plotter = Plotter()
# plotter.plot(svm_model, feature_vectors, data_parser.labels)
