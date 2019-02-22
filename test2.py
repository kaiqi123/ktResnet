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

train_image = tf.image.per_image_standardization(image_queue)

print(train_image)

print("1111111")