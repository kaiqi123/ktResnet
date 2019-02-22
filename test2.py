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

#images_list = []
#for i in range(x.shape[0]):
#    one = tf.convert_to_tensor([x[i], y[i]])
#    images_list.append(one)

image = tf.train.slice_input_producer([x[0], y[0]])
print(image)

