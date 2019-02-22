import numpy as np
import tensorflow as tf
from pylearn2.utils import serial


# convert_whiten npyFile to a txtfile for input and multi Npy Files
def deal_npy_file(whitenFile_label, whitenFile_feature, txtfile, mode):
    y = np.load(whitenFile_label)
    print(y.shape)

    x = np.load(whitenFile_feature)
    x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    print(x.shape)

    output_dir = "cifar10_npy/" + mode
    serial.mkdir(output_dir)
    file_names = []
    for i in range(x.shape[0]):
        name = output_dir + "/" + str(y[i][0]) + "," + mode + str(i)
        np.save(name, x[i])
        file_names.append(name+"\n")

    open(txtfile, "w").writelines(file_names)
    print(len(file_names))


# input file
whitenFile_label = "cifar10/pylearn2_gcn_whitened/train_labels.npy"
whitenFile_feature = "cifar10/pylearn2_gcn_whitened/train.npy"
# ouput file
txtfile = "cifar10_npy/cifar10-train.txt"
mode = "train"
deal_npy_file(whitenFile_label, whitenFile_feature, txtfile, mode)