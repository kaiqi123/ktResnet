import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K


class Model(object):

    def __init__(self, num_channels, seed, trainable=True):
        self.trainable = trainable
        self.parameters = []
        self.num_channels = num_channels
        self.seed = seed
        self.fc = None
        self.softmax = None

    def Convolution(self, imgInput, nInputPlane, nOutputPlane, stride, padding):
        with tf.name_scope('Convolution') as scope:
            imgInput = tf.pad(imgInput, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            kernel = tf.Variable(tf.truncated_normal([3, 3, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='kernel')
            conv = tf.nn.conv2d(imgInput, filter=kernel, strides=[1, stride, stride, 1], padding='VALID', name="conv")
            biases = tf.Variable(tf.constant(0.0, shape=[nOutputPlane], dtype=tf.float32), trainable=self.trainable, name='biase')
            imgOutput = tf.nn.bias_add(conv, biases, name=scope)
            print(conv)
            print(biases)
        return imgOutput

    def Convolution_dim(self, imgInput, nInputPlane, nOutputPlane, stride, padding):
        with tf.name_scope('Convolution_dim') as scope:
            imgInput = tf.pad(imgInput, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            kernel = tf.Variable(tf.truncated_normal([1, 1, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='kernel')
            conv = tf.nn.conv2d(imgInput, filter=kernel, strides=[1, stride, stride, 1], padding='VALID', name="conv_dim")
            biases = tf.Variable(tf.constant(0.0, shape=[nOutputPlane], dtype=tf.float32), trainable=self.trainable, name='biases_dim')
            imgOutput = tf.nn.bias_add(conv, biases, name=scope)
            print(conv)
            print(biases)
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

            batchNorm = BatchNormalization(axis=-1, name='BatchNormal')(imgInput)
            o1 = tf.nn.relu(batchNorm, name='relu')
            y = self.Convolution(o1, nInputPlane, nOutputPlane, stride=stride, padding=1)
            #print(y)

        with tf.name_scope('block_conv2') as scope:

            batchNorm = BatchNormalization(axis=-1, name='BatchNormal')(y)
            o2 = tf.nn.relu(batchNorm, name='relu')
            # dropout = tf.nn.dropout(relu, 0.3, seed=self.seed)
            z = self.Convolution(o2, nOutputPlane, nOutputPlane, stride=1, padding=1)
            #print(z)

        if nInputPlane != nOutputPlane:
            output = z + self.Convolution_dim(o1, nInputPlane, nOutputPlane, stride=stride, padding=0)
        else:
            output = z + imgInput
        print(output)

        return output

    def group(self, imgInput, nInputPlane, nOutputPlane, n, stride):
        print("group")
        with tf.name_scope('group1') as scope:
            block = self.basic_block(imgInput, nInputPlane, nOutputPlane, stride)
            for i in range(n - 1):
                x = block
                block = self.basic_block(x, nOutputPlane, nOutputPlane, 1)
        return block

    def build_teacher_model(self, rgb, num_classes, k, n):

        K.set_learning_phase(True)

        nStages = [16, 16 * k, 32 * k, 64 * k]

        conv1 = self.Convolution(rgb, self.num_channels, nStages[0], stride=1, padding=1)
        #print(conv1)
        group1 = self.group(conv1, nStages[0], nStages[1], n, 1)
        group2 = self.group(group1, nStages[1], nStages[2], n, 2)
        group3 = self.group(group2, nStages[2], nStages[3], n, 2)

        batchNorm = BatchNormalization(axis=-1, name='BatchNormal')(group3)
        relu = tf.nn.relu(batchNorm, name='relu')
        averagePool = tf.nn.avg_pool(relu, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID', name='averagePool')
        #print(averagePool)
        self.fc = self.FullyConnect(averagePool, num_classes)
        #print(self.fc)
        self.softmax = tf.nn.softmax(self.fc)
        #print(self.softmax)
        return self

    def build_conv1fc1(self, rgb, num_classes):

        print("build_conv1fc1")
        K.set_learning_phase(True)
        conv = self.Convolution(rgb, self.num_channels, 16, stride=1, padding=1)
        relu = tf.nn.relu(conv, name='relu')
        self.fc = self.FullyConnect(relu, num_classes)
        self.softmax = tf.nn.softmax(self.fc)
        return self

    def build_resnet_conv1Block1Fc1(self, rgb, num_classes, k):
        print("build_resnet_conv1Block1Fc1")
        K.set_learning_phase(True)
        nStages = [16, 16 * k, 32 * k, 64 * k]
        conv1 = self.Convolution(rgb, self.num_channels, nStages[0], stride=1, padding=1)
        print(conv1)
        block = self.basic_block(conv1,  nStages[0],  nStages[1], 2)
        batchNorm = BatchNormalization(axis=-1, name='BatchNormal')(block)
        relu = tf.nn.relu(batchNorm, name='relu')
        print(relu)
        averagePool = tf.nn.avg_pool(relu, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='SAME', name='averagePool')
        print(averagePool)
        self.fc = self.FullyConnect(averagePool, num_classes)
        print(self.fc)
        self.softmax = tf.nn.softmax(self.fc)
        print(self.softmax)
        return self

    def loss(self, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc, name='crossEntropy')
        return tf.reduce_mean(cross_entropy, name='EntropyMean')

    def training(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True )
        optimizer = tf.contrib.opt.MomentumWOptimizer(weight_decay=0.0005, learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op












