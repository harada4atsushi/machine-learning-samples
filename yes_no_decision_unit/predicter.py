# -*- coding: utf-8 -
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.learning_curve import learning_curve
from data_parser import DataParser

class Predicter:

    def predict(self, estimator, data_parser, X, vocabulary):
        #print X
        Xtrain = pd.Series(X)
        input_texts = []
        for input_text in Xtrain:
          input_texts.append(data_parser.split(input_text.decode('utf-8')))

        count_vectorizer = CountVectorizer(
            vocabulary=vocabulary # 学習時の vocabulary を指定する
        )
        feature_vectors = count_vectorizer.fit_transform(input_texts)
        return estimator.predict(feature_vectors)
