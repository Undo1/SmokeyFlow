# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import sys
import json

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

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
embedding_matrices = np.array([matrix_for_datapoint(d) for d in f]).tolist()
correct_answers = np.array([correct_prediction_for_datapoint(d) for d in f]).tolist()

test_embedding_matrices = embedding_matrices[:1000]
test_correct_answers = correct_answers[:1000]

embedding_matrices = embedding_matrices[1000:]
correct_answers = correct_answers[1000:]

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 410])
W = tf.Variable(tf.zeros([410, 2]))
b = tf.Variable(tf.zeros([2]))
intermediate = tf.matmul(x, W)
y = tf.nn.softmax(intermediate + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
for i in range(2):
  batch_xs, batch_ys = embedding_matrices, correct_answers
  train_step.run({x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(intermediate.eval({x: test_embedding_matrices, y_: test_correct_answers}))
