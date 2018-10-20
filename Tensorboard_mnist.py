# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:24:18 2017
 
@author: ggc
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
File_address = 'D://dogs'
BATCH_SIZE = 100
TRAIN_STEPS = 30000
def variable_summary(var,name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/'+name,mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.scalar_summary('stddev/'+name,stddev)
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name.scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim,output_dim], stddev=0.1))
            variable_summaries(weights, layer_name+'/weights')
        with tf.name_scope('biases'):
            
            
                
                
        
    
