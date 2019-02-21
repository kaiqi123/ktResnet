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
train_x = np.load("cifar10/pylearn2_gcn_whitened/train.npy")
train_x = train_x.reshape((train_x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
print(train_x.shape)
image= train_x
print(image.shape)
train_image = tf.image.per_image_standardization(image)