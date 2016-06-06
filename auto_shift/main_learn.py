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
svm_tuned_parameters = [
    {
        'kernel': ['rbf'],
        'gamma': [2**n for n in range(-15, 3)],
        'C': [2**n for n in range(-5, 15)]
    }
]
gscv = GridSearchCV(
    SVC(probability=True),
    svm_tuned_parameters,
    cv = 2,
    n_jobs = 1,
    verbose = 3
)

gscv.fit(dataset.features, dataset.attend)
clf = gscv.best_estimator_
#
# print clf  # 高パフォーマンスの学習モデル
# print gscv.best_params_  # 高パフォーマンスのパラメータ(gamma,Cの値)

# clf = SVC(probability=True)
# clf.fit(dataset.features, dataset.attend)

# shift = [
#    [0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0],
# ]
#
# for i in range(0, 7):
#     attend_number = 0
#     for row in shift:
#         attend_number += row[i]
#
#     attend = clf.predict([[i + 1, attend_number]])[0]
#     shift[0][i] = attend
#
# for i in range(0, 7):
#     attend_number = 0
#     for row in shift:
#         attend_number += row[i]
#
#     attend = clf.predict([[i + 1, attend_number]])[0]
#     shift[1][i] = attend
#
# print shift

shift = np.c_[np.zeros(4)]
for i in range(0, 7):
    results = clf.predict_proba([[i+1,1],[i+1,2],[i+1,3],[i+1,4]])
    indexes = np.argsort(results[:,1])[::-1]

    # 平日は2名、休日は3名体制
    threshold = 2
    if i in [5,6]:
        threshold = 3

    col = np.zeros(4)
    for i in range(0, threshold):
        col[indexes[i]] = 1

    shift = np.hstack([shift, np.c_[col]])



    #for j in range(1, 5):


    #for result in results

    #result = np.c_[clf.predict_proba([[i+1,1],[i+1,2],[i+1,3],[i+1,4]])]
    #shift = np.hstack([shift, result])

print shift[:,1:]
#     results = clf.predict([[i,1],[i,2]])


#print clf.predict_proba([[0,0]])
