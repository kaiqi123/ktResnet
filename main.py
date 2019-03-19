import tensorflow as tf
import numpy as np
from DataInput import DataInput
from ModelConstruct import Model
# from ModelConstruct_Pretrained import Model
import time
import sys
import argparse
import os
from tensorflow.python.client import device_lib

tf.reset_default_graph()
NUM_ITERATIONS = 78000
Widen_Factor = 10
Depth = 28
TeacherModel_N = (Depth - 4) / 6
SEED = 444
Num_Epoch_Per_Decay = 60
learningRateDecayRatio = 0.2
Pad = 4
Test_accuracy_List = []
Train_accuracy_List = []
DecayedLearningRate_List = []

class Resnet(object):

    def fill_feed_dict(self, data_input, images_pl, labels_pl, sess):
        images_feed, labels_feed = sess.run([data_input.example_batch, data_input.label_batch])
        feed_dict = {images_pl: images_feed, labels_pl: labels_feed}
        return feed_dict, images_feed, labels_feed

    def evaluation(self, logits, labels):
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def do_eval(self, sess, meval, images_placeholder, labels_placeholder, data_input, mode):

        eval_correct = self.evaluation(meval.softmax, labels_placeholder)

        if mode == 'Test':
            steps_per_epoch = FLAGS.num_testing_examples //FLAGS.batch_size
            num_examples = steps_per_epoch * FLAGS.batch_size
        if mode == 'Train':
            steps_per_epoch = FLAGS.num_training_examples //FLAGS.batch_size
            num_examples = steps_per_epoch * FLAGS.batch_size

        true_count = 0
        for step in xrange(steps_per_epoch):
            feed_dict, images_feed, labels_feed = self.fill_feed_dict(data_input, images_placeholder, labels_placeholder, sess)
            count = sess.run(eval_correct, feed_dict=feed_dict)
            true_count = true_count + count
        precision = float(true_count) / num_examples
        print ('Mode is %s, Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
               (mode, num_examples, true_count, precision))
        return precision

    def build_models(self, modelName, images_placeholder):
        with tf.variable_scope(modelName, use_resource=False):
            m = Model(FLAGS.num_channels, SEED)
            m = m.build_teacher_model(images_placeholder, FLAGS.num_classes,
                                                                      Widen_Factor, TeacherModel_N, True)
            self.saver = tf.train.Saver()
        with tf.variable_scope(modelName, reuse=True, use_resource=False):
            meval = Model(FLAGS.num_channels, SEED)
            meval = meval.build_teacher_model(images_placeholder, FLAGS.num_classes,
                                                                    Widen_Factor, TeacherModel_N, False)
        return m, meval

    def define_teacher(self, images_placeholder, labels_placeholder, global_step):

        print("Define Teacher")

        self.m, self.meval = self.build_models("teacherModel", images_placeholder)

        self.loss = self.m.loss(labels_placeholder)

        steps_per_epoch = FLAGS.num_examples_per_epoch_for_train / FLAGS.batch_size
        decay_steps = int(steps_per_epoch * Num_Epoch_Per_Decay)
        self.lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, learningRateDecayRatio, staircase=True)
        print("Steps_per_epoch: "+str(steps_per_epoch))
        print("Decay_steps: " + str(decay_steps))

        self.train_op = self.m.training(self.loss, self.lr, global_step)

    def save_model(self, session, step=None):
        model_save_name = os.path.join(FLAGS.teacher_model_dir, 'model.ckpt')
        if not tf.gfile.IsDirectory(FLAGS.teacher_model_dir):
            tf.gfile.MakeDirs(FLAGS.teacher_model_dir)
        self.saver.save(session, model_save_name, global_step=step)
        print('Saved model')

    def train_model(self, data_input_train, data_input_test, images_placeholder, labels_placeholder, sess):

        try:
            print('Begin to train model...')

            eval_correct_train = self.evaluation(self.m.softmax, labels_placeholder)

            for i in range(NUM_ITERATIONS):
                # print("iteration: "+str(i))
                feed_dict, images_feed, labels_feed = self.fill_feed_dict(data_input_train,
                                                                          images_placeholder, labels_placeholder, sess)

                if FLAGS.teacher or FLAGS.student:
                    # print("train function: independent student or teacher")
                    _, loss_value, train_count_per_batch = sess.run([self.train_op, self.loss, eval_correct_train], feed_dict=feed_dict)
                    train_acc_per_iteration = float(train_count_per_batch) / FLAGS.batch_size

                    if i % 20 == 0:
                        print ('Step %d: loss_value = %.20f, train_acc = %.20f' % (i, loss_value, train_acc_per_iteration))

                if (i) % (FLAGS.num_examples_per_epoch_for_train // FLAGS.batch_size) == 0:

                    #self.save_model(sess, step=i)
                    decayedLearningRate = sess.run(self.lr)
                    DecayedLearningRate_List.append(decayedLearningRate)
                    print ('Decayed learning rate list: ' + str(DecayedLearningRate_List))

                    train_acc = self.do_eval(sess, self.meval, images_placeholder, labels_placeholder, data_input_test, 'Test')
                    test_acc = self.do_eval(sess, self.meval, images_placeholder, labels_placeholder, data_input_train, 'Train')
                    Test_accuracy_List.append(test_acc)
                    Train_accuracy_List.append(train_acc)
                    tf.logging.info('Train Acc List: {}'.format(Train_accuracy_List))
                    tf.logging.info('Test Acc List: {}'.format(Test_accuracy_List))

        except Exception as e:
            print(e)

    def main(self, _):

        start_time = time.time()
        with tf.Graph().as_default():
            print("test whether to use gpu")
            print(device_lib.list_local_devices())

            # This line allows the code to use only sufficient memory and does not block entire GPU
            # config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

            # set the seed so that we have same loss values and initializations for every run.
            tf.set_random_seed(SEED)

            data_input_train = DataInput(FLAGS.train_dataset, FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height,
                      FLAGS.num_channels, SEED, Pad, FLAGS.datasetName)

            data_input_test = DataInput(FLAGS.test_dataset, FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height,
                                        FLAGS.num_channels, SEED, Pad, FLAGS.datasetName)

            images_placeholder = tf.placeholder(tf.float32,
                                                shape=(FLAGS.batch_size, FLAGS.image_height,
                                                       FLAGS.image_width, FLAGS.num_channels))
            labels_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size))

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.allocator_type = 'BFC'

            sess = tf.Session(config=config)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            #phase_train = tf.placeholder(tf.bool, name='phase_train')

            print("NUM_ITERATIONS: " + str(NUM_ITERATIONS))
            print("learning_rate: " + str(FLAGS.learning_rate))
            print("batch_size: " + str(FLAGS.batch_size))
            print("Depth: " + str(Depth))
            print("TeacherModel_N: " + str(TeacherModel_N))
            print("Widen_Factor: " + str(Widen_Factor))

            if FLAGS.teacher:
                self.define_teacher(images_placeholder, labels_placeholder, global_step)

            init = tf.global_variables_initializer()
            sess.run(init)

            self.train_model(data_input_train, data_input_test, images_placeholder, labels_placeholder, sess)

            tf.logging.info('Train Acc List: {}'.format(Train_accuracy_List))
            tf.logging.info('Test Acc List: {}'.format(Test_accuracy_List))
            #writer_tensorboard = tf.summary.FileWriter('tensorboard/', sess.graph)

            coord.request_stop()
            coord.join(threads)

        sess.close()
        #writer_tensorboard.close()

        end_time = time.time()
        runtime = round((end_time - start_time) / (60 * 60), 2)
        print("run time is: " + str(runtime) + " hour")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=bool, help='train teacher', default=False)
    parser.add_argument('--student', type=bool, help='train independent student', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_height', type=int, default=32)
    parser.add_argument('--image_width', type=int, default=32)
    parser.add_argument('--train_dataset', type=str, default="cifar10_input/cifar10-train.txt")
    parser.add_argument('--test_dataset', type=str, default="cifar10_input/cifar10-test.txt")
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_training_examples', type=int, default=50000)
    parser.add_argument('--num_testing_examples',type=int,default=10000)
    parser.add_argument('--datasetName', type=str, help='name of the dataset', default='cifar10')
    parser.add_argument('--num_channels', type=int, default='3')
    parser.add_argument('--num_optimizers', type=int, help='number of mapping layers from teacher', default=1)
    parser.add_argument('--num_examples_per_epoch_for_train', type=int, default=50000)
    #parser.add_argument('--teacher_model_dir', type=str, default="./summary-log/teacher/")
    FLAGS, unparsed = parser.parse_known_args()
    ex = Resnet()
    tf.app.run(main=ex.main, argv=[sys.argv[0]] + unparsed)
