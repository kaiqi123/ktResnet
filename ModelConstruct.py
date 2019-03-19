import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
# from tensorflow.python.keras.layers import BatchNormalization
# from tensorflow.python.keras import backend as K


class Model(object):

    def __init__(self, num_channels, seed, trainable=True):
        self.trainable = trainable
        self.parameters = []
        self.num_channels = num_channels
        self.seed = seed
        self.fc = None
        self.softmax = None
        #self.bn = self.batch_norm()

    def conv2d(self, imgInput, nInputPlane, nOutputPlane, stride, padding):
        with tf.variable_scope('conv'):
            imgInput = tf.pad(imgInput, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            kernel = tf.Variable(tf.truncated_normal([3, 3, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='conv_kernel')
            conv = tf.nn.conv2d(imgInput, filter=kernel, strides=[1, stride, stride, 1], padding='VALID', name="conv")
            print(kernel)
        return conv

    def conv2d_dim(self, imgInput, nInputPlane, nOutputPlane, stride, padding):
        with tf.variable_scope('conv_dim'):
            imgInput = tf.pad(imgInput, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            kernel = tf.Variable(tf.truncated_normal([1, 1, nInputPlane, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='conv_kernel')
            conv = tf.nn.conv2d(imgInput, filter=kernel, strides=[1, stride, stride, 1], padding='VALID', name="conv_dim")
            print(kernel)
        return conv

    def FullyConnect(self, imgInput, nOutputPlane):
        with tf.variable_scope('fc'):
            shape = int(np.prod(imgInput.get_shape()[1:]))
            fcw = tf.Variable(tf.truncated_normal([shape, nOutputPlane], dtype=tf.float32, stddev=1e-2, seed=self.seed), trainable=self.trainable, name='weights')
            fcb = tf.Variable(tf.constant(0.0, shape=[nOutputPlane], dtype=tf.float32), trainable=self.trainable, name='biases')
            flat = tf.reshape(imgInput, [-1, shape])
            imgOutput = tf.nn.bias_add(tf.matmul(flat, fcw), fcb)
            print(fcw)
            print(fcb)
        return imgOutput

    def batch_norm(self, x, phase_train, scope):

        if phase_train:
            reuse_flag = False
        else:
            reuse_flag = True

        weight = tf.random_normal_initializer(mean=1, stddev=0.045)
        bias = tf.constant_initializer(value=0)
        moving_mean = tf.constant_initializer(value=0)
        moving_variance = tf.ones_initializer()

        bn = tf.contrib.layers.batch_norm(
            x,
            decay=0.9,
            center=True,
            scale=True,
            epsilon=1e-5,
            activation_fn=None,
            param_initializers={'beta': bias,
                                'gamma': weight,
                                'moving_mean': moving_mean,
                                'moving_variance': moving_variance},
            updates_collections=tf.GraphKeys.UPDATE_OPS,
            is_training=phase_train,
            reuse=reuse_flag,
            trainable=True,
            fused=True,
            data_format='NHWC',
            zero_debias_moving_mean=False,
            scope=scope)

        print(weight)
        print(bias)
        print(moving_mean)
        print(moving_variance)
        return bn

    def basic_block(self, imgInput, nInputPlane, nOutputPlane, stride, mode, scope):

        print("basic_block")
        with tf.variable_scope(scope):

            with tf.variable_scope('sub_block0'):
                o1 = tf.nn.relu(self.batch_norm(imgInput, mode, 'bn0'), name='relu')
                y = self.conv2d(o1, nInputPlane, nOutputPlane, stride=stride, padding=1)
                print(y)

            with tf.variable_scope('sub_block1'):
                o2 = tf.nn.relu(self.batch_norm(y, mode, 'bn1'), name='relu')
                # dropout = tf.nn.dropout(relu, 0.3, seed=self.seed)
                z = self.conv2d(o2, nOutputPlane, nOutputPlane, stride=1, padding=1)
                #print(z)

            if nInputPlane != nOutputPlane:
                output = z + self.conv2d_dim(o1, nInputPlane, nOutputPlane, stride=stride, padding=0)
            else:
                output = z + imgInput
            # print(output)

        return output

    def group(self, imgInput, nInputPlane, nOutputPlane, n, stride, mode, scope):
        print(scope)
        with tf.variable_scope(scope):
            block = self.basic_block(imgInput, nInputPlane, nOutputPlane, stride, mode, 'block'+str(0))
            for i in range(n - 1):
                x = block
                block = self.basic_block(x, nOutputPlane, nOutputPlane, 1, mode, 'block' + str(i+1))
        return block

    def build_teacher_model(self, rgb, num_classes, k, n, mode):

        # K.clear_session()

        nStages = [16, 16 * k, 32 * k, 64 * k]

        x = self.conv2d(rgb, self.num_channels, nStages[0], stride=1, padding=1)
        g0 = self.group(x, nStages[0], nStages[1], n, 1, mode, scope='group0')
        g1 = self.group(g0, nStages[1], nStages[2], n, 2, mode, scope='group1')
        g2 = self.group(g1, nStages[2], nStages[3], n, 2, mode, scope='group2')

        relu = tf.nn.relu(self.batch_norm(g2, mode, 'afterGroupBn3'), name='relu')
        averagePool = tf.nn.avg_pool(relu, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID', name='averagePool')
        self.fc = self.FullyConnect(averagePool, num_classes)
        self.softmax = tf.nn.softmax(self.fc)
        return self


    def loss(self, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc, name='crossEntropy')
        return tf.reduce_mean(cross_entropy, name='EntropyMean')

    def training(self, loss, learning_rate, global_step):

        print("Define training.....................................................")
        #print(self.bn)
        #print(self.bn.updates)
        optimizer = tf.contrib.opt.MomentumWOptimizer(weight_decay=0.0005, learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        apply_op = optimizer.minimize(loss, global_step=global_step)
        train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([apply_op]):
            train_op = tf.group(*train_ops)
        #train_op = tf.group([train_op, update_ops])

        """
        print('variables: %d' % len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(e)

        print('trainable variables: %d' % len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
        for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(e)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print('n_update_ops: %d' % len(update_ops))
        # print(update_ops)

        #print('n_update_ops(bn): %d' % len(self.bn.updates))
        #print(self.bn.updates)
        # for e in bn.updates:
        #    print(e)
        """

        return train_op












