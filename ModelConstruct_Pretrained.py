# from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K
import torch
import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self, num_channels, seed, trainable=True):
        self.trainable = trainable
        self.parameters = []
        self.num_channels = num_channels
        self.seed = seed
        self.fc = None
        self.softmax = None
        #self.phase_train = phase_train
        #self.params = self.read_parameters()

    def read_parameters(self):
        def tr(v):
            if v.ndim == 4:
                return v.transpose(2, 3, 1, 0)
            elif v.ndim == 2:
                return v.transpose()
            return v

        params = {k: v.detach().cpu().numpy() for k, v in torch.load('cifar10_input/model_d28w10.pt7')['params'].items()}

        print("Display the dimesion of the pretrained parameters: ")
        params_new = {}
        for k, v in sorted(params.items()):
            if 'bn' in k:
                params_new[k] = v
                print(k, params_new[k].shape)
            else:
                params_new[k] = tr(v)
                print(k, params_new[k].shape)
        print("---------------------")
        # for k, v in sorted(params.items()):
        #    print(k, tuple(v.shape))
        # params = {k: tf.constant(tr(v)) for k, v in params.items()}
        return params_new

    def batch_norm(self, x, params, base, mode):

        #batchNorm = BatchNormalization(axis=-1, name='BatchNorm', trainable=self.trainable)(imgInput)

        bias = tf.constant_initializer(params[base + '.bias'])
        weight = tf.constant_initializer(params[base + '.weight'])
        moving_mean = tf.constant_initializer(params[base + '.running_mean'])
        moving_variance = tf.constant_initializer(params[base + '.running_var'])
        """
        weight = tf.random_normal_initializer(1.0, 0.0)
        bias = tf.constant_initializer(0.)
        moving_mean = tf.constant_initializer(0.)
        moving_variance = tf.ones_initializer()

        params_init = {
            'beta': bias,
            'gamma': weight,
            'moving_mean': moving_mean,
            'moving_variance': moving_variance
        }
        """
        batchNorm = tf.layers.batch_normalization(x, center=True, scale=True,
                                                  beta_initializer=bias,
                                                  gamma_initializer=weight,
                                                  moving_mean_initializer=moving_mean,
                                                  moving_variance_initializer = moving_variance,
                                                  training=mode)
        return batchNorm

    """
    def conv2d(self, x, params, stride=1, padding=0):
        x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        # kernel = tf.Variable(params, trainable=self.trainable, name='conv_kernel')
        size = params.shape[0]
        nInputPlane = params.shape[2]
        nOutputPlane = params.shape[3]
        kernel = tf.Variable(tf.truncated_normal([size, size, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed),trainable=self.trainable, name='conv_kernel')
        z = tf.nn.conv2d(x, filter=kernel, strides=[1, stride, stride, 1], padding='VALID')
        print(kernel)
        return z
    """

    def block(self, x, params, base, mode, stride):

        """
        o1 = tf.nn.relu(self.batch_norm(x, params, base + '.bn0', mode))
        y = self.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = tf.nn.relu(self.batch_norm(y, params, base + '.bn1', mode))
        z = self.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + self.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x
        """
        o1 = tf.nn.relu(x)
        y = self.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = tf.nn.relu(y)
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

        #o = tf.nn.relu(self.batch_norm(g2, params, 'bn', mode))
        o = tf.nn.relu(g2)
        o = tf.nn.avg_pool(o, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        shape = int(np.prod(o.get_shape()[1:]))
        o = tf.reshape(o, [-1, shape])
        self.fc = tf.matmul(o, params['fc.weight']) + params['fc.bias']
        self.softmax = tf.nn.softmax(self.fc)

        print(x)
        print(g0)
        print(g1)
        print(g2)
        print(self.fc)
        return self

    def conv2d(self, imgInput, nInputPlane, nOutputPlane, stride, padding):
        with tf.name_scope('Convolution') as scope:
            imgInput = tf.pad(imgInput, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            kernel = tf.Variable(tf.truncated_normal([3, 3, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='conv_kernel')
            conv = tf.nn.conv2d(imgInput, filter=kernel, strides=[1, stride, stride, 1], padding='VALID', name="conv")
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

    def test(self, input, n, mode):

        print("test teacher")
        K.set_learning_phase(True)
        # params = self.read_parameters()
        x = self.conv2d(input, self.num_channels, 16, stride=1, padding=1)
        # x = self.conv2d(input, params['conv0'], padding=1)
        o = tf.nn.relu(x)
        self.fc = self.FullyConnect(o, 10)
        self.softmax = tf.nn.softmax(self.fc)
        return self


    def loss(self, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc, name='crossEntropy')
        return tf.reduce_mean(cross_entropy, name='EntropyMean')

    def training(self, loss, learning_rate, global_step):

        """
        optimizer = tf.contrib.opt.MomentumWOptimizer(weight_decay=0.0005, learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
        """

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.contrib.opt.MomentumWOptimizer(weight_decay=0.0005, learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        train_op = optimizer.minimize(loss, global_step=global_step)
        # train_op = tf.group([train_op, update_ops])
        return train_op













