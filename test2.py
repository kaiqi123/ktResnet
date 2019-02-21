import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

print("2222")

batch_size = 128
seed = 1234

y = np.load("cifar10/pylearn2_gcn_whitened/train_labels.npy")
print(y.shape)

x = np.load("cifar10/pylearn2_gcn_whitened/train.npy")
x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
print(x.shape)

image_list = []
for i in range(x.shape[0]):
    one = {}
    one["feature"] = x[i]
    one["label"] = y[i]
    image_list.append(one)
print(len(image_list))

image_queue = tf.train.input_producer(image_list)
min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
image_batch = tf.train.shuffle_batch(
    image_queue, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, seed=seed)

print(image_batch)

print("1111111")