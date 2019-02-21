import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

"""
a = np.load("temp/temp_origin.npy")
print(a.shape)
plt.imshow(a, interpolation='none')
plt.show()

b = np.load("temp/temp_whiten.npy")
print(b.shape)
plt.imshow(b)
plt.show()
"""

batch_size = 128
seed = 1234

y = np.load("cifar10/pylearn2_gcn_whitened/train_labels.npy")
print(y.shape)

x = np.load("cifar10/pylearn2_gcn_whitened/train.npy")
x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
print(x.shape)

"""
image_x = x[7000]
image_y = y[7000]
print(image_x.shape)
print(image_y)
image_x = tf.image.per_image_standardization(image_x)
tf.train.input_producer()
"""

#feature_list = []
image_list = []
for i in range(x.shape[0]):
    one = {}
    one["feature"] = x[i]
    one["label"] = y[i]
    image_list.append(one)

image_queue = tf.train.input_producer(image_list)
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
        #print(images_feed.shape)
        #print(labels_feed.shape)

    coord.request_stop()
    coord.join(threads)
    print("1111111")