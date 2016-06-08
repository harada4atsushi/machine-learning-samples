# -*- coding: utf-8 -
#import pdb
import sys; sys.path.append('db')
import numpy as np
import calendar
from seed import *
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from dataset import Dataset
from prettytable import PrettyTable
from datetime import date

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

# 6月分のシフト表を生成する
# 意図したとおりに動作しなかったので一旦コメントアウト
# holidays = [
#     [1,7,8, 14,15,21,22,28,29],  # 山田の休み希望
#     [2,3,9, 10,16,17,23,24,31],  # 田中の休み希望
#     [3,4,10,11,17,18,24,25,],  # 佐藤の休み希望
#     [5,6,12,13,19,20,26,27,3],  # 高橋の休み希望
# ]

month = 6
days = calendar.monthrange(2016, month)[1]
header = ['名前']
day_of_week_label = ["月","火","水","木","金","土","日"]
for day in range(1, days + 1):
    day_of_week = date(2016, month, day).weekday()
    results = clf.predict_proba([[day_of_week,1],[day_of_week,2],[day_of_week,3],[day_of_week,4]])  # 予測する
    indexes = np.argsort(results[:,1])[::-1]  # 確率の高い順に並び替え
    header.append(str(day) + " " + day_of_week_label[day_of_week])

    # 平日は上位2名、休日は上位3名を出勤にする
    threshold = 2
    if day_of_week in [5,6]:
        threshold = 3

    col = np.zeros(4)
    for i in range(0, 4):
        # if day in holidays[i]:  # 休み希望が入っている日程の場合は入れない
        #     #print day
        #     continue
        col[indexes[i]] = 1
        if sum(col) >= threshold:
            break

    shift = np.hstack([shift, np.c_[col]])

shift = shift[:,1:]  # 先頭の0埋め列を削除する

# 表形式に表示する
employees = ['山田', '田中', '佐藤', '高橋']
table = PrettyTable(header)
table.padding_width = 1  # 左右の空白

for i, row in enumerate(shift):
    row = row.tolist()
    row.insert(0, employees[i])
    table.add_row(row)

print table
