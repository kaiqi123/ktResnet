import tensorflow as tf
import numpy as np
import os

test_dataset = "cifar10_input/cifar10-test.txt"
batch_size = 128
num_testing_examples = 10000
image_width = 32
image_height = 32
num_channels = 3
seed = 1234
pad = 4

os.chdir(r'/home/users/kaiqi/ktResnet/')
print(os.getcwd())

filename_queue = tf.train.string_input_producer([test_dataset], num_epochs=None)
reader = tf.TextLineReader()
key_temp, value_temp = reader.read(filename_queue)
print(value_temp)
record_defaults = [[1], ['']]
col1, col2 = tf.decode_csv(value_temp, record_defaults=record_defaults)
print(col1, col2)

file_content = tf.read_file(col2)
# file_content = np.load(col2)
print(file_content)

train_image = tf.image.per_image_standardization(file_content)
"""
train_image = tf.image.resize_image_with_pad(train_image, image_width + pad, image_width + pad)
train_image = tf.image.random_flip_left_right(train_image)
train_image = tf.random_crop(train_image, [image_height, image_width, 3], seed=seed, name="crop")
print(train_image)

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
example_batch, label_batch = tf.train.shuffle_batch (
        [train_image, col1], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, seed=seed)
"""

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.initialize_all_variables())

    for i in range(1):

        """
        images_feed, labels_feed = sess.run([example_batch, label_batch])
        print(images_feed.shape)
        print(labels_feed)
        print(type(images_feed))
        print(type(labels_feed))
        """

        images_feed = sess.run(train_image)
        print(images_feed)

    coord.request_stop()
    coord.join(threads)
