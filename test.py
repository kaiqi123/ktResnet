import numpy as np
import os
#import matplotlib.pyplot as plt
from scipy import misc
import cv2


# convert_whiten npyFile to a txtfile for input and multi Npy Files
def deal_npy_file(whitenFile_label, whitenFile_feature, txtfile, mode):
    y = np.load(whitenFile_label)
    print(y.shape)

    x = np.load(whitenFile_feature)
    x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    print(x.shape)

    i = 0
    print(x[i].shape)
    #plt.imshow(x[i])

    output_dir = "./cifar10_images_from_npy/" + mode
    name = output_dir + "/" + mode + str(i) + "cv2.png"
    cv2.imwrite(name, x[i])
    #misc.imsave(name, x[i])
    # plt.imsave(name, x[i])

    #plt.show()



# input file
whitenFile_label = "./cifar10/pylearn2_gcn_whitened/train_labels.npy"
whitenFile_feature = "./cifar10/pylearn2_gcn_whitened/train.npy"
# ouput file
txtfile = "./cifar10_input/cifar10-train.txt"
mode = "train"
deal_npy_file(whitenFile_label, whitenFile_feature, txtfile, mode)