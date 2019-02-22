import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

batch_size = 128
seed = 1234

y = np.load("cifar10/pylearn2_gcn_whitened/train_labels.npy")
print(y.shape)

x = np.load("cifar10/pylearn2_gcn_whitened/train.npy")
x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
print(x.shape)

images_list = []
for i in range(x.shape[0]):
    one = {}
    one["feature"] = x[i]
    one["label"] = y[i]
    images_list.append(one)

images_queue = tf.train.input_producer(images_list)


min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
image_batch = tf.train.shuffle_batch(
    image_queue, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, seed=seed)

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.initialize_all_variables())

    for i in range(1):
        images_one_batch = sess.run(image_batch)
        images_feed = []
        labels_feed = []
        for id_sample in range(len(images_one_batch)):
            image = images_one_batch[id_sample]
            images_feed.append(image["feature"])
            labels_feed.append(image["label"])
        images_feed = np.array(images_feed)
        labels_feed = np.array(labels_feed)
        print(images_feed.shape)
        print(labels_feed.shape)

    coord.request_stop()
    coord.join(threads)

print("1111111")