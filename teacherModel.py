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

    def Convolution(self, imgInput, nInputPlane, nOutputPlane, stride):
        with tf.name_scope('Convolution') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='kernel')
            conv = tf.nn.conv2d(imgInput, filter=kernel, strides=[1, stride, stride, 1], padding='SAME')
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

    def basic_block(self, imgInput, nInputPlane, nOutputPlane, stride):

        print("basic_block")

        with tf.name_scope('block_conv1') as scope:
            out = self.Convolution(imgInput, nInputPlane, nOutputPlane, stride)
            relu = tf.nn.relu(out, name='relu')
            batchNorm = BatchNormalization(axis = -1, name= 'BatchNormal')(relu)
            print(batchNorm)

        with tf.name_scope('block_conv2') as scope:
            dropout = tf.nn.dropout(batchNorm, 0.3, seed=self.seed)
            out = self.Convolution(dropout, nOutputPlane, nOutputPlane, stride)
            relu = tf.nn.relu(out, name='relu')
            batchNorm = BatchNormalization(axis = -1, name= 'BatchNormal')(relu)
            print(batchNorm)
        return batchNorm

    def layer(self, imgInput, nInputPlane, nOutputPlane, n, stride):

        print("group")
        with tf.name_scope('group1') as scope:
            block = self.basic_block(imgInput, nInputPlane, nOutputPlane, stride)
            for i in range(n-1):
                x = block
                block = self.basic_block(x, nOutputPlane, nOutputPlane, 1)
        return block

    def build_teacher_model(self, rgb, num_classes, k, n):

        K.set_learning_phase(True)

        nStages = [16, 16 * k, 32 * k, 64 * k]

        conv1 = self.Convolution(rgb, self.num_channels, nStages[0], 1)
        print(conv1)
        #group1 = self.layer(conv1, nStages[0], nStages[1], n, 1)
        #group2 = self.layer(group1, nStages[1], nStages[2], n, 2)
        #group3 = self.layer(group2, nStages[2], nStages[3], n, 2)

        group1 = self.basic_block(conv1, nStages[0], nStages[1], 1)
        group2 = self.basic_block(group1, nStages[1], nStages[2], 2)
        group3 = self.basic_block(group2, nStages[2], nStages[3], 2)

        batchNorm = BatchNormalization(axis=-1, name='BatchNormal')(group3)
        relu = tf.nn.relu(batchNorm, name='relu')
        averagePool = tf.nn.avg_pool(relu, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='SAME', name='averagePool')
        print(averagePool)
        self.fc = self.FullyConnect(averagePool, num_classes)
        print(self.fc)
        self.softmax = tf.nn.softmax(self.fc)
        print(self.softmax)
        return self

    def build_vgg_conv1fc1(self, rgb, num_classes):

        print("build_vgg_conv1fc1")
        K.set_learning_phase(True)
        conv = self.Convolution(rgb, self.num_channels, 16, 1)
        relu = tf.nn.relu(conv, name='relu')
        #batchNorm = BatchNormalization(axis=-1, name='BatchNormal')(relu)
        self.fc = self.FullyConnect(relu, num_classes)
        self.softmax = tf.nn.softmax(self.fc)
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












