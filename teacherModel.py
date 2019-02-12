import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K


class Teacher(object):

    def __init__(self, num_channels, trainable=True):
        self.trainable = trainable
        self.parameters = []
        self.num_channels = num_channels

    def basic_block(self, nInputPlane, nOutputPlane):

        print("wide_basic")

        with tf.name_scope('block_conv1') as scope:
            batchNorm = BatchNormalization(axis = -1, name= 'block_conv1_BatchNormal')(nInputPlane)
            print(batchNorm)
            relu = tf.nn.relu(batchNorm, name='block_conv1_relu')
            kernel = tf.Variable(tf.truncated_normal([3, 3, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=seed), trainable=self.trainable, name='block_conv1_kernel')
            conv = tf.nn.conv2d(relu, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[nOutputPlane], dtype=tf.float32),
                                 trainable=self.trainable, name='block_conv1_biases')
            out = tf.nn.bias_add(conv, biases, name = scope)

        #with tf.name_scope('block_conv2') as scope:
        #    batchNorm = BatchNormalization(axis = -1, name= 'block_conv1_BatchNormal')(block_conv1_out)

    def build_teacher_model(self, rgb, num_classes, seed):

        k = 2
        nStages = [16, 16 * k, 32 * k, 64 * k]

        with tf.name_scope('teacher_conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, self.num_channels, 16], dtype=tf.float32, stddev=1e-2, seed=seed), trainable=self.trainable, name='teacher_conv1_kernel')
            conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=self.trainable, name='teacher_conv1_biases')
            conv1_out = tf.nn.bias_add(conv, biases, name = scope)
            print(conv1_out)




