# -*- coding: utf-8 -

import MeCab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tinydb import TinyDB, Query
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from data_parser import DataParser
from plotter import Plotter
from predicter import Predicter

# featureを準備する
data_parser = DataParser(file='data.tsv')
count_vectorizer = CountVectorizer()
feature_vectors = count_vectorizer.fit_transform(data_parser.splited_texts)
vocabulary = count_vectorizer.get_feature_names()

# 学習する
svm_tuned_parameters = [
    {
        'kernel': ['linear'],
        'gamma': [2**n for n in range(-5, 3)],
        'C': [2**n for n in range(-5, 8)]
        # 'gamma': [2**n for n in range(-15, 3)],
        # 'C': [2**n for n in range(-5, 15)]
    }
]
gscv = GridSearchCV(
    svm.SVC(),
    svm_tuned_parameters,
    cv = 2,  # k-fold cross-validation トレーニングセットとクロスバリデーションセットに分割したものをcv回クロスバリデーションする
    n_jobs = 1,
    verbose = 3
)

gscv.fit(feature_vectors, data_parser.labels)
svm_model = gscv.best_estimator_
print svm_model  # 高パフォーマンスの学習モデル
print gscv.best_params_  # 高パフォーマンスのパラメータ(gamma,Cの値)

#joblib.dump(svm_model, "models/svm_model")
#svm_model = joblib.load("models/svm_model")

# 学習曲線を描画する
#plotter = Plotter()
#plotter.plot(svm_model, feature_vectors, data_parser.labels)

# 分類させる
sys.argv.pop(0)
predicter = Predicter()
result = predicter.predict(svm_model, data_parser, sys.argv, vocabulary)
print result
