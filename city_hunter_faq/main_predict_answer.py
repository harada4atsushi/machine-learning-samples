# -*- coding: utf-8 -
from sklearn.externals import joblib
from predicter_answer import PredicterAnswer

X = [
        [1, '主人公について'],
        [1, '投げナイフ使う女'],
        [1, '猫が苦手'],
        [1, '隣で探偵やってる'],
        [2, '指の間を撃ちぬくやつ'],
        [2, 'ギャンブラー']
    ]

svm_model = joblib.load("models/answer_svm_model")
vocabulary = joblib.load("vocabulary/answer_vocabulary.pkl")

predicter = PredicterAnswer(svm_model, vocabulary)
result = predicter.predict(X)

# TODO 回答文言に変換する
print result
