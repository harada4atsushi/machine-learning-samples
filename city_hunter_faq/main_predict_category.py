# -*- coding: utf-8 -
from sklearn.externals import joblib
from predicter import Predicter

X = ['あらすじについて知りたい', 'ストーリー教えて', '登場人物', 'キャラ']

category_svm_model = joblib.load("models/category_svm_model")
category_vocabulary = joblib.load("vocabulary/category_vocabulary.pkl")

predicter = Predicter(category_svm_model, category_vocabulary)
result = predicter.predict(X)

print result
