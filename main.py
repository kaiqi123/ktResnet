import tensorflow as tf
import numpy as np
import random
from DataInput import DataInput
from teacherModel import Teacher
import os
import time
import pdb
import sys
from tensorflow.python import debug as tf_debug
import argparse
from tensorflow.python.client import device_lib

tf.reset_default_graph()
NUM_ITERATIONS = 7820
seed = 1234
dataset_path = "./"

class Resnet(object):

    def define_teacher(self, images_placeholder, labels_placeholder, global_step, sess):

        mentor = Teacher(FLAGS.num_channels)

        mentor.build_teacher_model(images_placeholder, FLAGS.num_classes, seed)
        init = tf.global_variables_initializer()
        sess.run(init)

    def main(self, _):
        with tf.Graph().as_default():
            print("test whether to use gpu")
            print(device_lib.list_local_devices())

            # This line allows the code to use only sufficient memory and does not block entire GPU
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

            # set the seed so that we have same loss values and initializations for every run.
            tf.set_random_seed(seed)

            data_input_train = DataInput(dataset_path, FLAGS.train_dataset, FLAGS.batch_size,
                                         FLAGS.num_training_examples, FLAGS.image_width, FLAGS.image_height,
                                         FLAGS.num_channels, seed, FLAGS.datasetName)

            data_input_test = DataInput(dataset_path, FLAGS.test_dataset, FLAGS.batch_size, FLAGS.num_testing_examples,
                                        FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.datasetName)

            images_placeholder = tf.placeholder(tf.float32,
                                                shape=(FLAGS.batch_size, FLAGS.image_height,
                                                       FLAGS.image_width, FLAGS.num_channels))
            labels_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size))

            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'

            sess = tf.Session(config=config)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            #phase_train = tf.placeholder(tf.bool, name='phase_train')

            print("NUM_ITERATIONS: " + str(NUM_ITERATIONS))
            print("learning_rate: " + str(FLAGS.learning_rate))
            print("batch_size: " + str(FLAGS.batch_size))

            if FLAGS.teacher:
                self.define_teacher(images_placeholder, labels_placeholder, global_step, sess)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--teacher',
        type=bool,
        help='train teacher',
        default=False
    )
    parser.add_argument(
        '--teacher_weights_filename',
        type=str,
        default="./summary-log/teacher_weights_filename_cifar10"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128
    )
    parser.add_argument(
        '--image_height',
        type=int,
        default=32
    )
    parser.add_argument(
        '--image_width',
        type=int,
        default=32
    )
    parser.add_argument(
        '--train_dataset',
        type=str,
        default="cifar10_input/cifar10-train.txt"
    )
    parser.add_argument(
        '--test_dataset',
        type=str,
        default="cifar10_input/cifar10-test.txt"
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10
    )
    parser.add_argument(
        '--NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN',
        type=int,
        default=50000
    )
    parser.add_argument(
        '--num_training_examples',
        type=int,
        default=50000
    )
    parser.add_argument(
        '--num_testing_examples',
        type=int,
        default=10000
    )
    parser.add_argument(
        '--datasetName',
        type=str,
        help='name of the dataset',
        default='cifar10'
    )
    parser.add_argument(
        '--num_channels',
        type=int,
        help='number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
        default='3'
    )
    parser.add_argument(
        '--top_1_accuracy',
        type=bool,
        help='top-1-accuracy',
        default=True
    )

    FLAGS, unparsed = parser.parse_known_args()
    ex = Resnet()
    tf.app.run(main=ex.main, argv=[sys.argv[0]] + unparsed)
