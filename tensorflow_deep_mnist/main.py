# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnistの手書き文字データを取得して読み込む
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# tensorflowのセッションを取得
sess = tf.InteractiveSession()

# トレーニングセットのfeatureを入れる入れものを定義
# 画像の28*28ドットのfeatureが入る
x = tf.placeholder(tf.float32, shape=[None, 784])

# ラベルを入れるための入れ物を定義。0〜9までの数字が入るので10で初期化している
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# weightを0で初期化。学習結果で更新される値
W = tf.Variable(tf.zeros([784,10]))

# biasを0で初期化。学習結果で更新される値
b = tf.Variable(tf.zeros([10]))

# 変数を初期化する(詳しくはわからないけど)
sess.run(tf.initialize_all_variables())

# w,bが0のままでyを予測させる
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 予測結果と実際の平均誤差を取得？
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
