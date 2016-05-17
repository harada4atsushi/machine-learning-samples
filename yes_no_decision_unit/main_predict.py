# -*- coding: utf-8 -

import MeCab
import pandas as pd
import numpy as np
import sqlite3
import sys
from tinydb import TinyDB, Query
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from data_parser import DataParser
from predicter import Predicter

#db = TinyDB('development.json')
try:
    sqlitedb = sqlite3.connect("../db/development.sqlite3")
except:
    sqlitedb = sqlite3.connect("db/development.sqlite3")

data_parser = DataParser(sqlitedb=sqlitedb)
#count_vectorizer = CountVectorizer()
#feature_vectors = count_vectorizer.fit_transform(data_parser.splited_texts)
#vocabulary = count_vectorizer.get_feature_names()

try:
    svm_model = joblib.load("models/svm_model")
    vocabulary = joblib.load("vocabulary/vocabulary.pkl")
except:
    svm_model = joblib.load("learning/models/svm_model")
    vocabulary = joblib.load("learning/vocabulary/vocabulary.pkl")

# 分類させる
sys.argv.pop(0)
print sys.argv[0]
predicter = Predicter()
result = predicter.predict(svm_model, data_parser, sys.argv, vocabulary)
print result
print result[0]
sys.exit(0)
