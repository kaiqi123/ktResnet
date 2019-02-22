import numpy as np
import tensorflow as tf
from pylearn2.utils import serial

y = np.load("cifar10/pylearn2_gcn_whitened/train_labels.npy")
print(y.shape)

x = np.load("cifar10/pylearn2_gcn_whitened/train.npy")
x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
print(x.shape)

output_dir = "cifar10_npy/train"
serial.mkdir(output_dir)
file_names = []
for i in range(x.shape[0]):
    name = output_dir + "/" + str(y[i][0]) + ",train" + str(i)
    np.save(name, x[i])
    file_names.append(name)

open("cifar10_npy/cifar10-train.txt", "w").writelines(file_names)
print(len(file_names))



