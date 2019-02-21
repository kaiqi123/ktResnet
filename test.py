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
image= train_x[7000]
print(image.shape)
train_image = tf.image.per_image_standardization(image)
print(train_image)

"""
train_image = tf.image.per_image_standardization(train_image)
train_image = tf.image.resize_image_with_pad(train_image, self.image_width + self.pad, self.image_width + self.pad)
train_image = tf.image.random_flip_left_right(train_image)
print(train_image)
self.train_image = tf.random_crop(train_image, [self.image_height, self.image_width, 3], seed=self.seed, name="crop")
print(self.train_image)
"""