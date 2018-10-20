# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:43:40 2017

@author: ggc
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.reset_default_graph()
tf.set_random_seed(42)
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer = tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(input_tensor, regularizer,reuse):
    with tf.variable_scope('layer1',reuse=reuse):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope('layer2',reuse=reuse):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer = tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1,weights)+biases)
    return layer2

x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
REGULARAZTION_RATE = 0.0001
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
y = inference(x,regularizer,None)
y = inference(x,regularizer,reuse=True)

