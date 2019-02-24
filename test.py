import numpy as np
from pylearn2.utils import serial
import os
from PIL import Image


# convert_whiten npyFile to a txtfile for input and multi Npy Files
def deal_npy_file(whitenFile_label, whitenFile_feature, txtfile, mode):
    y = np.load(whitenFile_label)
    print(y.shape)

    x = np.load(whitenFile_feature)
    x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    # print(x.shape[0])
    # x = x.reshape(x.shape[0], 3, 32, 32)
    print(x.shape)

    output_dir = "./cifar10_images_from_npy/" + mode
    serial.mkdir(output_dir)
    i = 0
    print(x[i].shape)
    #file_names = []
    #for i in range(x.shape[0]):

    img = x[i]
    img = Image.fromarray(img.astype(np.uint8))
    #i0 = Image.fromarray(img[0]).convert('L')
    #i1 = Image.fromarray(img[1]).convert('L')
    #i2 = Image.fromarray(img[2]).convert('L')
    #img = Image.merge("RGB", (i0, i1, i2))
    name = output_dir + "/" + mode + str(i) + ".png"
    img.save(name, "png")

    #file_names.append(str(y[i][0]) + "," + output_dir + "/" + mode + str(i) + ".npy" + "\n")

    #open(txtfile, "w").writelines(file_names)
    #print(len(file_names))


os.chdir(r'/home/users/kaiqi/ktResnet/')
print(os.getcwd())


# input file
whitenFile_label = "./cifar10/pylearn2_gcn_whitened/train_labels.npy"
whitenFile_feature = "./cifar10/pylearn2_gcn_whitened/train.npy"
# ouput file
txtfile = "./cifar10_input/cifar10-train.txt"
mode = "train"
deal_npy_file(whitenFile_label, whitenFile_feature, txtfile, mode)