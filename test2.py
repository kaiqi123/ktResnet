import numpy as np
import tensorflow as tf
from pylearn2.utils import serial

batch_size = 128
seed = 1234

y = np.load("cifar10/pylearn2_gcn_whitened/train_labels.npy")
print(y[0])
#y = list(y)
print(y[0][0])

"""
x = np.load("cifar10/pylearn2_gcn_whitened/train.npy")
x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
print(x.shape)

output_dir = "cifar10_npy/train"
serial.mkdir(output_dir)
#for i in range(x.shape[0]):
i = 0
np.save(output_dir+"/"+str(y[i])+",train"+str(i), x[i])
"""

