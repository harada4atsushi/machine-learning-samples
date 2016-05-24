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
from predicter import Predicter

# featureを準備する
db = TinyDB('db/dataset_answer.json')
data_parser = DataParser(db=db)
count_vectorizer = CountVectorizer()
feature_vectors = count_vectorizer.fit_transform(data_parser.splited_texts)
vocabulary = count_vectorizer.get_feature_names()

# featureにcategory_idを追加
features_array = feature_vectors.toarray()
category_ids =  np.array(data_parser.category_ids).T
features_array = np.c_[category_ids, features_array]

# 学習する
svm_tuned_parameters = [
    {
        'kernel': ['linear'],
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

gscv.fit(features_array, data_parser.labels)
svm_model = gscv.best_estimator_

print svm_model  # 高パフォーマンスの学習モデル
print gscv.best_params_  # 高パフォーマンスのパラメータ(gamma,Cの値)

# 学習済みモデルをdumpする
joblib.dump(svm_model, "models/answer_svm_model")
joblib.dump(vocabulary, 'vocabulary/answer_vocabulary.pkl')
