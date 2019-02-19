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
        self.fc = None
        self.softmax = None

    def Convolution(self, imgInput, nInputPlane, nOutputPlane, stride=1, padding=0):
        with tf.name_scope('Convolution') as scope:
            x = tf.pad(imgInput, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            kernel = tf.Variable(tf.truncated_normal([3, 3, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='kernel')
            conv = tf.nn.conv2d(imgInput, filter=kernel, strides=[1, stride, stride, 1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[nOutputPlane], dtype=tf.float32), trainable=self.trainable, name='biases')
            imgOutput = tf.nn.bias_add(conv, biases, name=scope)
        return imgOutput

    def FullyConnect(self, imgInput, nOutputPlane):
        with tf.name_scope('FullyConnect') as scope:
            shape = int(np.prod(imgInput.get_shape()[1:]))
            fcw = tf.Variable(tf.truncated_normal([shape, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='weights')
            fcb = tf.Variable(tf.constant(0.0, shape=[nOutputPlane], dtype=tf.float32), trainable=self.trainable, name='biases')
            flat = tf.reshape(imgInput, [-1, shape])
            imgOutput = tf.nn.bias_add(tf.matmul(flat, fcw), fcb)
        return imgOutput

    def group(self, input, nInputPlane, nOutputPlane, stride, n):
        o = input
        for i in range(0, n):
            x = o
            o = self.Convolution(x, nInputPlane, nOutputPlane)
            o = tf.nn.relu(o)
            o = self.Convolution(o, nOutputPlane, nOutputPlane, stride=i == 0 and stride or 1, padding=1)
            o = tf.nn.relu(o)
            o = self.Convolution(o, nOutputPlane, nOutputPlane)
            if i == 0:
                o += self.Convolution(x, nOutputPlane, nOutputPlane, stride=stride)
            else:
                o += x
            o = tf.nn.relu(o)
        return o

    def build_teacher_model(self, rgb, num_classes, k, n):

        nStages = [16, 16 * k, 32 * k, 64 * k]
        o = self.Convolution(rgb, self.num_channels, nStages[0], 2, 3)
        print(o)
        o = tf.nn.relu(o)
        o = tf.pad(o, [[0, 0], [1, 1], [1, 1], [0, 0]])
        o = tf.nn.max_pool(o, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        print(o)
        o_g0 = self.group(o, nStages[0], nStages[0], 1, 1)
        print(o)
        o_g1 = self.group(o_g0, nStages[0], nStages[1], 2, 1)
        print(o)
        o_g2 = self.group(o_g1, nStages[1], nStages[2], 2, 1)
        print(o)
        o_g3 = self.group(o_g2, nStages[2], nStages[3], 2, 1)
        print(o)
        o = tf.nn.avg_pool(o_g3, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
        self.fc = self.FullyConnect(o, 2048)
        print(o)
        self.softmax = tf.nn.softmax(self.fc)
        print(o)
        return self

    def loss(self, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc, name='crossEntropy')
        return tf.reduce_mean(cross_entropy, name='entropyMean')

    def training(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op












