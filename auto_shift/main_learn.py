# -*- coding: utf-8 -
#import pdb
import sys; sys.path.append('db')
import numpy as np
from seed import *
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from dataset import Dataset
from prettytable import PrettyTable

seed()
dataset = Dataset('db/dataset.json')

# 学習する
svm_tuned_parameters = [
    {
        'kernel': ['rbf'],
        # 'gamma': [2**n for n in range(-15, 3)],
        # 'C': [2**n for n in range(-5, 15)]
        # 毎回GridSearchさせると時間がかかってしまうので開発中は固定しておく
        'gamma': [4],
        'C': [1]
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

print clf  # 高パフォーマンスの学習モデル
print gscv.best_params_  # 高パフォーマンスのパラメータ(gamma,Cの値)

shift = np.c_[np.zeros(4)]
for i in range(0, 7):
    results = clf.predict_proba([[i+1,1],[i+1,2],[i+1,3],[i+1,4]])  # 予測する
    indexes = np.argsort(results[:,1])[::-1]  # 確率の高い順に並び替え

    # 平日は上位2名、休日は上位3名を出勤にする
    threshold = 2
    if i in [5,6]:
        threshold = 3

    col = np.zeros(4)
    for i in range(0, threshold):
        col[indexes[i]] = 1
    shift = np.hstack([shift, np.c_[col]])

shift = shift[:,1:]  # 先頭の0埋め列を削除する

# 表形式に表示する
header = ['名前', '月', '火', '水', '木', '金', '土', '日']
employees = ['山田', '田中', '佐藤', '高橋']
table = PrettyTable(header)
table.padding_width = 2  # 左右の空白は 2 つ

for i, row in enumerate(shift):
    row = row.tolist()
    row.insert(0, employees[i])
    table.add_row(row)

print table
