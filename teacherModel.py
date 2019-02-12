import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K


class Teacher(object):

    def __init__(self, num_channels, seed, trainable=True):
        self.trainable = trainable
        self.parameters = []
        self.num_channels = num_channels
        self.seed = seed

    def Convolution(self, imgInput, nInputPlane, nOutputPlane):
        with tf.name_scope('Convolution') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='kernel')
            conv = tf.nn.conv2d(imgInput, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[nOutputPlane], dtype=tf.float32), trainable=self.trainable, name='biases')
            imgOutput = tf.nn.bias_add(conv, biases, name=scope)
        return imgOutput

    def basic_block(self, imgInput, nInputPlane, nOutputPlane):

        print("basic_block")

        with tf.name_scope('block_conv1') as scope:
            batchNorm = BatchNormalization(axis = -1, name= 'BatchNormal')(imgInput)
            relu = tf.nn.relu(batchNorm, name='relu')
            out1 = self.Convolution(relu, nInputPlane, nOutputPlane)
            print(out1)

        with tf.name_scope('block_conv2') as scope:
            batchNorm = BatchNormalization(axis = -1, name= 'block_conv1_BatchNormal')(out1)
            relu = tf.nn.relu(batchNorm, name='relu')
            dropout = tf.nn.dropout(relu, 0.5, seed=self.seed)
            out2 = self.Convolution(dropout, nOutputPlane, nOutputPlane)
            print(out2)


    def build_teacher_model(self, rgb, num_classes):

        k = 10
        nStages = [16, 16 * k, 32 * k, 64 * k]

        #self.basic_block(nInputPlane, nOutputPlane)
        conv1 = self.Convolution(rgb, self.num_channels, nStages[0])
        print(conv1)
        self.basic_block(conv1, nStages[0], nStages[1])




