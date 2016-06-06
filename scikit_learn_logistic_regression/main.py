# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]  # フィーチャーを2つに絞る
Y = iris.target

# 学習させる
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
print logreg.predict([6.5, 3])
