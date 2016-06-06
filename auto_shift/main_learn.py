# -*- coding: utf-8 -
#import pdb
import sys; sys.path.append('db')
import numpy as np
from seed import *
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from dataset import Dataset
from feature_builder import FeatureBuilder
# from predicter import Predicter

seed()

dataset = Dataset('db/dataset.json')

# feature_builder = FeatureBuilder(dataset)
# features = feature_builder.get_features()
#
# 学習する
# svm_tuned_parameters = [
#     {
#         'kernel': ['linear'],
#         'gamma': [2**n for n in range(-15, 3)],
#         'C': [2**n for n in range(-5, 15)]
#     }
# ]
# gscv = GridSearchCV(
#     SVC(),
#     svm_tuned_parameters,
#     cv = 2,
#     n_jobs = 1,
#     verbose = 3
# )
#
# #pdb.set_trace()
# gscv.fit(dataset.day_of_week, dataset.attend)
# svm_model = gscv.best_estimator_

# print svm_model  # 高パフォーマンスの学習モデル
# print gscv.best_params_  # 高パフォーマンスのパラメータ(gamma,Cの値)
#
# # 学習済みモデルをdumpする
# joblib.dump(svm_model, "models/svm_model")
# joblib.dump(feature_builder.vocabulary, 'vocabulary/vocabulary.pkl')

# print dataset.features
# print dataset.attend

clf = SVC()
clf.fit(dataset.features, dataset.attend)

print clf.predict([[1, 1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7]])
