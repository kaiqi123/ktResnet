import numpy as np
import os
# import matplotlib.pyplot as plt
from pylearn2.utils import serial
from scipy import misc
import cv2

"""
Note:
misc did normalization, min=0, max=255
plt Save an array as in image file, not know whether do normalization
"""


# convert_whiten npyFile to a txtfile for input and multi png images
def deal_npy_file(whitenFile_label, whitenFile_feature, txtfile, mode):
    y = np.load(whitenFile_label)
    print(y.shape)

    x = np.load(whitenFile_feature)
    x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    print(x.shape)

    output_dir = "./cifar10_images_from_npy/" + mode + "_misc"
    serial.mkdir(output_dir)
    file_names = []
    # for i in range(x.shape[0]):
    for i in range(1):
        name = output_dir + "/" + mode + str(i) + "_misc.png"
        # plt.imsave(name, x[i])
        misc.imsave(name, x[i])
        file_names.append(str(y[i][0]) + "," + name + "\n")

    open(txtfile, "w").writelines(file_names)
    print(len(file_names))


os.chdir(r'/home/users/kaiqi/ktResnet/')
print(os.getcwd())

# input file
whitenFile_label = "./cifar10/pylearn2_gcn_whitened/train_labels.npy"
whitenFile_feature = "./cifar10/pylearn2_gcn_whitened/train.npy"
# ouput file
txtfile = "./cifar10_input/cifar10-train-misc.txt"
mode = "train"
deal_npy_file(whitenFile_label, whitenFile_feature, txtfile, mode)

# input file
whitenFile_label = "./cifar10/pylearn2_gcn_whitened/test_labels.npy"
whitenFile_feature = "./cifar10/pylearn2_gcn_whitened/test.npy"
# ouput file
txtfile = "./cifar10_input/cifar10-test-misc.txt"
mode = "test"
deal_npy_file(whitenFile_label, whitenFile_feature, txtfile, mode)

