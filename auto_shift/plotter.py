# -*- coding: utf-8 -
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve

class Plotter:
    #def __init__(self):

    def plot(self, estimator, X, y):
        train_sizes = np.arange(40, int(len(y) * 0.6), 10)  # 等間隔数値の配列
        cv = cross_validation.StratifiedKFold(y, n_folds=3, shuffle=True)
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, train_sizes=train_sizes, cv=cv)

        plt.figure()
        plt.title('learning curve')
        #plt.ylim([0.7, 1.01])  # yの表示範囲
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        train_scores_mean = np.mean(train_scores, axis=1)
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        #plt.show()
        plt.pause(5)
