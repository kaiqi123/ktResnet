import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K


class Model(object):

    def __init__(self, num_channels, seed, trainable=True, phase_train=True):
        self.trainable = trainable
        self.parameters = []
        self.num_channels = num_channels
        self.seed = seed
        self.fc = None
        self.softmax = None
        self.phase_train = phase_train

    def conv2d(self, imgInput, nInputPlane, nOutputPlane, stride, padding):
        with tf.name_scope('Convolution') as scope:
            imgInput = tf.pad(imgInput, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            kernel = tf.Variable(tf.truncated_normal([3, 3, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='conv_kernel')
            conv = tf.nn.conv2d(imgInput, filter=kernel, strides=[1, stride, stride, 1], padding='VALID', name="conv")
            print(kernel)
        return conv

    def conv2d_dim(self, imgInput, nInputPlane, nOutputPlane, stride, padding):
        with tf.name_scope('Convolution_dim') as scope:
            imgInput = tf.pad(imgInput, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            kernel = tf.Variable(tf.truncated_normal([1, 1, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='conv_kernel')
            conv = tf.nn.conv2d(imgInput, filter=kernel, strides=[1, stride, stride, 1], padding='VALID', name="conv_dim")
            print(kernel)
        return conv

    def FullyConnect(self, imgInput, nOutputPlane):
        with tf.name_scope('FullyConnect') as scope:
            shape = int(np.prod(imgInput.get_shape()[1:]))
            fcw = tf.Variable(tf.truncated_normal([shape, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='weights')
            fcb = tf.Variable(tf.constant(0.0, shape=[nOutputPlane], dtype=tf.float32), trainable=self.trainable, name='biases')
            flat = tf.reshape(imgInput, [-1, shape])
            imgOutput = tf.nn.bias_add(tf.matmul(flat, fcw), fcb)
            print(fcw)
            print(fcb)
        return imgOutput

    def batch_norm(self, imgInput, bnN):
        with tf.name_scope('bn') as scope:
            #weight= tf.Variable(tf.random_uniform(shape=[bnN], minval=0.0, maxval=1.0, dtype=tf.float32, seed=self.seed), trainable=self.trainable, name='weight')
            #bias = tf.Variable(tf.constant(0.0, shape=[bnN], dtype=tf.float32), trainable=self.trainable, name='bias')
            #running_mean = tf.Variable(tf.constant(0.0, shape=[bnN], dtype=tf.float32), trainable=False, name='running_mean')
            #running_var = tf.Variable(tf.constant(1.0, shape=[bnN], dtype=tf.float32), trainable=False, name='running_var')

            weight = tf.random_normal_initializer(1.0, 0.0)
            bias = tf.constant_initializer(0.)
            running_mean = tf.constant_initializer(0.)
            running_var = tf.ones_initializer()

            params = {
                'beta': bias,
                'gamma': weight,
                'moving_mean': running_mean,
                'moving_variance': running_var
            }
            batchNorm = tf.contrib.layers.batch_norm(imgInput, center=True, scale=True, param_initializers=params, is_training=self.phase_train, scope=scope)

            print(weight)
            print(bias)
            print(running_mean)
            print(running_var)
        return batchNorm

    def basic_block(self, imgInput, nInputPlane, nOutputPlane, stride):

        print("basic_block")

        with tf.name_scope('block_conv1') as scope:
            o1 = tf.nn.relu(self.batch_norm(imgInput, nInputPlane), name='relu')
            y = self.conv2d(o1, nInputPlane, nOutputPlane, stride=stride, padding=1)
            #print(y)

        with tf.name_scope('block_conv2') as scope:
            o2 = tf.nn.relu(self.batch_norm(y, nOutputPlane), name='relu')
            # dropout = tf.nn.dropout(relu, 0.3, seed=self.seed)
            z = self.conv2d(o2, nOutputPlane, nOutputPlane, stride=1, padding=1)
            #print(z)

        if nInputPlane != nOutputPlane:
            output = z + self.conv2d_dim(o1, nInputPlane, nOutputPlane, stride=stride, padding=0)
        else:
            output = z + imgInput
        # print(output)

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

        x = self.conv2d(rgb, self.num_channels, nStages[0], stride=1, padding=1)
        g0 = self.group(x, nStages[0], nStages[1], n, 1)
        g1 = self.group(g0, nStages[1], nStages[2], n, 2)
        g2 = self.group(g1, nStages[2], nStages[3], n, 2)

        relu = tf.nn.relu(self.batch_norm(g2, nStages[3]), name='relu')
        averagePool = tf.nn.avg_pool(relu, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID', name='averagePool')
        self.fc = self.FullyConnect(averagePool, num_classes)
        self.softmax = tf.nn.softmax(self.fc)
        return self


    def loss(self, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc, name='crossEntropy')
        return tf.reduce_mean(cross_entropy, name='EntropyMean')

    def training(self, loss, learning_rate, global_step):

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            tf.summary.scalar('loss', loss)
            optimizer = tf.contrib.opt.MomentumWOptimizer(weight_decay=0.0005, learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op












