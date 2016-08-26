import sys
import json
import tensorflow as tf
import numpy as np
import random

def load_file():
  return json.loads(open("/Users/parkererway/Desktop/ms_data.json").read())

def matrix_for_datapoint(dp):
  # Embedding matrix format:
  # [Rep] [Site... (one-hot)] [Reasons... (many-hot)]

  sites_embedding = [0] * 309 # Site count. Ugly.

  try:
    sites_embedding[dp["site"]] = 1
  except:
    print("Site out of range")

  reasons_embedding = [0] * 99 # Reason count. Also ugly.

  try:
    for r in dp['reasons']:
      reasons_embedding[r] = 1
  except:
    print("Reason out of range")

  return [dp['id']] + [dp['user_reputation']] + sites_embedding + reasons_embedding

def correct_prediction_for_datapoint(dp):
  return dp['result']

f = load_file()
embedding_matrices = np.array([matrix_for_datapoint(d) for d in f])
correct_answers = np.array([correct_prediction_for_datapoint(d) for d in f])

test_embedding_matrices = embedding_matrices[:1000]
test_correct_answers = correct_answers[:1000]

embedding_matrices = embedding_matrices[1000:]
correct_answers = correct_answers[1000:]

print(test_embedding_matrices)

x = tf.placeholder(tf.float32, [None, 410])
W = tf.Variable(tf.zeros([410, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(30):
  batch_xs, batch_ys = embedding_matrices, correct_answers
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(y, feed_dict={x: test_embedding_matrices, y_: test_correct_answers}))
