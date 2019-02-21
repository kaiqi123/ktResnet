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

train_x = np.load("cifar10/pylearn2_gcn_whitened/train.npy")
train_x = train_x.reshape((train_x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
print(train_x.shape)
image= train_x[7000]
print(image.shape)
train_image = tf.image.per_image_standardization(image)

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
example_batch= tf.train.shuffle_batch (
        [train_image], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, seed=seed)

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.initialize_all_variables())

    for i in range(1):
        images_feed = sess.run(example_batch)
        print(images_feed.shape)
        #print(labels_feed)

    coord.request_stop()
    coord.join(threads)