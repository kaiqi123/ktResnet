import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K
import torch

class Model(object):

    def __init__(self, num_channels, seed, trainable=True):
        self.trainable = trainable
        self.parameters = []
        self.num_channels = num_channels
        self.seed = seed
        self.fc = None
        self.softmax = None
        #self.phase_train = phase_train
        self.params = self.read_parameters()

    def read_parameters(self):
        def tr(v):
            if v.ndim == 4:
                return v.transpose(2, 3, 1, 0)
            elif v.ndim == 2:
                return v.transpose()
            return v

        params = {k: v.detach().cpu().numpy() for k, v in torch.load('cifar10_input/model_d28w10.pt7')['params'].items()}
        # for k, v in sorted(params.items()):
        #    print(k, tuple(v.shape))
        #print("---------------------")
        params_new = {}
        for k, v in sorted(params.items()):
            if 'bn' in k:
                # params_new[k] = tf.constant(v.transpose())
                print(k, v)
                # print()
            else:
                params_new[k] = tf.constant(tr(v))
                print(k, tf.shape(params_new[k]))
        # params = {k: tf.constant(tr(v)) for k, v in params.items()}
        #for k, v in sorted(params_new.items()):
        #    print(k, type(v))
        return params

    def batch_norm(self, x, params, base, mode):

        #batchNorm = BatchNormalization(axis=-1, name='BatchNorm', trainable=self.trainable)(imgInput)
        bias = tf.constant_initializer(params[base + '.bias'])
        weight = tf.constant_initializer(params[base + '.weight'])
        moving_mean = tf.constant_initializer(params[base + '.running_mean'])
        moving_variance = tf.constant_initializer(params[base + '.running_var'])
        params_init = {
            'beta': bias,
            'gamma': weight,
            'moving_mean': moving_mean,
            'moving_variance': moving_variance
        }
        batchNorm = tf.contrib.layers.batch_norm(x, center=True, scale=True, param_initializers=params_init, is_training=mode)
        return batchNorm

    def conv2d(self, x, params, stride=1, padding=0):
        x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        z = tf.nn.conv2d(x, params, [1, stride, stride, 1], padding='VALID')
        # print(params)
        return z

    def block(self, x, params, base, mode, stride):
        o1 = tf.nn.relu(self.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = self.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = tf.nn.relu(self.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = self.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + self.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(self, o, params, base, mode, stride, n):
        for i in range(n):
            o = self.block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o


    def build_teacher_model(self, input, n, mode):

        K.set_learning_phase(True)
        params = self.params
        x = self.conv2d(input, params['conv0'], padding=1)
        g0 = self.group(x, params, 'group0', mode, 1, n)
        g1 = self.group(g0, params, 'group1', mode, 2, n)
        g2 = self.group(g1, params, 'group2', mode, 2, n)

        o = tf.nn.avg_pool(g2, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        shape = int(np.prod(o.get_shape()[1:]))
        o = tf.reshape(o, [-1, shape])
        self.fc = tf.matmul(o, params['fc.weight']) + params['fc.bias']
        self.softmax = tf.nn.softmax(self.fc)
        return self


    def loss(self, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc, name='crossEntropy')
        return tf.reduce_mean(cross_entropy, name='EntropyMean')

    def training(self, loss, learning_rate, global_step):

        """
        tf.summary.scalar('loss', loss)
        optimizer = tf.contrib.opt.MomentumWOptimizer(weight_decay=0.0005, learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            tf.summary.scalar('loss', loss)
            optimizer = tf.contrib.opt.MomentumWOptimizer(weight_decay=0.0005, learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
            train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op













