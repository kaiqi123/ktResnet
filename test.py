import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


file_npy = np.load("temp/temp_whiten.npy")
print(file_npy.shape)
train_image = tf.image.per_image_standardization(file_npy)
print(train_image.shape)

