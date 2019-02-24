import tensorflow as tf
import numpy as np
from DataInput import DataInput
from ModelConstruct import Model
import time
import sys
import argparse
from tensorflow.python.client import device_lib

tf.reset_default_graph()
NUM_ITERATIONS = 2
TeacherModel_K = 10
Depth = 28
TeacherModel_N = (Depth - 4) / 6
SEED = 1234
Num_Epoch_Per_Decay = 60
learningRateDecayRatio = 0.2
test_accuracy_list = []
Pad = 4

class Resnet(object):

    def fill_feed_dict(self, data_input, images_pl, labels_pl, sess, mode, phase_train):

        images_feed, labels_feed = sess.run([data_input.example_batch, data_input.label_batch])

        if mode == 'Train':
            feed_dict = {
                images_pl: images_feed,
                labels_pl: labels_feed,
                phase_train: True
            }

        if mode == 'Test':
            feed_dict = {
                images_pl: images_feed,
                labels_pl: labels_feed,
                phase_train: False
            }

        if mode == 'Validation':
            feed_dict = {
                images_pl: images_feed,
                labels_pl: labels_feed,
                phase_train: False
            }
        return feed_dict, images_feed, labels_feed

    def evaluation(self, logits, labels):

        if FLAGS.top_1_accuracy:
            print('evaluation: top 1 accuracy ')
            correct = tf.nn.in_top_k(logits, labels, 1)
        elif FLAGS.top_3_accuracy:
            correct = tf.nn.in_top_k(logits, labels, 3)
        elif FLAGS.top_5_accuracy:
            correct = tf.nn.in_top_k(logits, labels, 5)

        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def do_eval(self, sess, eval_correct, logits, images_placeholder, labels_placeholder, dataset,mode, phase_train):

            if mode == 'Test':
                steps_per_epoch = FLAGS.num_testing_examples //FLAGS.batch_size
                num_examples = steps_per_epoch * FLAGS.batch_size
            if mode == 'Train':
                steps_per_epoch = FLAGS.num_training_examples //FLAGS.batch_size
                num_examples = steps_per_epoch * FLAGS.batch_size
            if mode == 'Validation':
                steps_per_epoch = FLAGS.num_validation_examples //FLAGS.batch_size
                num_examples = steps_per_epoch * FLAGS.batch_size

            true_count = 0
            for step in xrange(steps_per_epoch):
                if FLAGS.datasetName == 'mnist':
                    feed_dict = {images_placeholder: np.reshape(dataset.test.next_batch(FLAGS.batch_size)[0], [FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels]), labels_placeholder: dataset.test.next_batch(FLAGS.batch_size)[1]}
                else:
                    feed_dict, images_feed, labels_feed = self.fill_feed_dict(dataset, images_placeholder,
                                                            labels_placeholder,sess, mode,phase_train)
                count = sess.run(eval_correct, feed_dict=feed_dict)
                true_count = true_count + count

            precision = float(true_count) / num_examples
            print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
                            (num_examples, true_count, precision))

            if mode == 'Test':
                test_accuracy_list.append(precision)

    def define_teacher(self, images_placeholder, labels_placeholder, global_step, sess):

        mentor = Model(FLAGS.num_channels, SEED)
        mentor_data_dict = mentor.build_teacher_model(images_placeholder, FLAGS.num_classes, TeacherModel_K, TeacherModel_N)
        #mentor_data_dict = mentor.build_vgg_conv1fc1(images_placeholder, FLAGS.num_classes)
        self.loss = mentor.loss(labels_placeholder)

        # learning rate decay
        steps_per_epoch = FLAGS.num_examples_per_epoch_for_train / FLAGS.batch_size
        decay_steps = int(steps_per_epoch * Num_Epoch_Per_Decay)
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, learningRateDecayRatio, staircase=True)

        self.train_op = mentor.training(self.loss, lr, global_step)
        self.softmax = mentor_data_dict.softmax

        init = tf.global_variables_initializer()
        sess.run(init)
        self.saver = tf.train.Saver()

    def define_independent_student(self, images_placeholder, labels_placeholder, global_step, sess):

        print("Define Independent student")
        student = Model(FLAGS.num_channels, SEED)
        if FLAGS.num_optimizers == 3:
            mentee_data_dict = student.build_resnet_conv1Block1Fc1(images_placeholder, FLAGS.num_classes,  TeacherModel_K)
        if FLAGS.num_optimizers == 1:
            mentee_data_dict = student.build_conv1fc1(images_placeholder, FLAGS.num_classes)

        self.loss = student.loss(labels_placeholder)

        steps_per_epoch = FLAGS.num_examples_per_epoch_for_train / FLAGS.batch_size
        decay_steps = int(steps_per_epoch * Num_Epoch_Per_Decay)
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, learningRateDecayRatio, staircase=True)

        self.train_op = student.training(self.loss, lr, global_step)
        self.softmax = mentee_data_dict.softmax

        init = tf.global_variables_initializer()
        sess.run(init)
        self.saver = tf.train.Saver()

    def train_model(self, data_input_train, data_input_test, images_placeholder, labels_placeholder, sess, phase_train):

        try:
            print('train model')

            eval_correct = self.evaluation(self.softmax, labels_placeholder)

            for i in range(NUM_ITERATIONS):

                # print("iteration: "+str(i))

                feed_dict, images_feed, labels_feed = self.fill_feed_dict(data_input_train, images_placeholder,
                                                                          labels_placeholder, sess, 'Train',
                                                                          phase_train)

                if FLAGS.teacher or FLAGS.student:
                    # print("train function: independent student or teacher")
                    _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if i % 10 == 0:
                        print ('Step %d: loss_value = %.20f' % (i, loss_value))

                if (i) % (FLAGS.num_examples_per_epoch_for_train // FLAGS.batch_size) == 0 or (i) == NUM_ITERATIONS - 1:

                    if FLAGS.teacher:
                        print("save teacher to: " + str(FLAGS.teacher_weights_filename))
                        self.saver.save(sess, FLAGS.teacher_weights_filename)

                    print ("Training Data Eval:")
                    self.do_eval(sess,
                                 eval_correct,
                                 self.softmax,
                                 images_placeholder,
                                 labels_placeholder,
                                 data_input_train,
                                 'Train', phase_train)

                    print ("Test  Data Eval:")
                    self.do_eval(sess,
                                 eval_correct,
                                 self.softmax,
                                 images_placeholder,
                                 labels_placeholder,
                                 data_input_test,
                                 'Test', phase_train)
                    print ("max test accuracy % f", max(test_accuracy_list))

        except Exception as e:
            print(e)

    def main(self, _):

        start_time = time.time()
        with tf.Graph().as_default():
            print("test whether to use gpu")
            print(device_lib.list_local_devices())

            # This line allows the code to use only sufficient memory and does not block entire GPU
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

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
            config.gpu_options.allocator_type = 'BFC'

            sess = tf.Session(config=config)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            phase_train = tf.placeholder(tf.bool, name='phase_train')

            print("NUM_ITERATIONS: " + str(NUM_ITERATIONS))
            print("learning_rate: " + str(FLAGS.learning_rate))
            print("batch_size: " + str(FLAGS.batch_size))
            print("TeacherModel_N: " + str(TeacherModel_N))

            if FLAGS.teacher:
                self.define_teacher(images_placeholder, labels_placeholder, global_step, sess)

            elif FLAGS.student:
                self.define_independent_student(images_placeholder, labels_placeholder, global_step, sess)

            self.train_model(data_input_train, data_input_test, images_placeholder, labels_placeholder, sess, phase_train)

            print(test_accuracy_list)
            writer_tensorboard = tf.summary.FileWriter('tensorboard/', sess.graph)

            coord.request_stop()
            coord.join(threads)

        sess.close()
        writer_tensorboard.close()

        end_time = time.time()
        runtime = round((end_time - start_time) / (60 * 60), 2)
        print("run time is: " + str(runtime) + " hour")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=bool, help='train teacher', default=False)
    parser.add_argument('--student', type=bool, help='train independent student', default=False)
    parser.add_argument('--teacher_weights_filename', type=str, default="./summary-log/teacher_weights_filename_cifar10")
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
    parser.add_argument('--top_1_accuracy', type=bool, help='top-1-accuracy', default=True)
    parser.add_argument('--num_optimizers', type=int, help='number of mapping layers from teacher', default=1)
    FLAGS, unparsed = parser.parse_known_args()
    ex = Resnet()
    tf.app.run(main=ex.main, argv=[sys.argv[0]] + unparsed)
