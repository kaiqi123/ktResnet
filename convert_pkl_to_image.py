import cPickle
# from pylearn2.datasets.cifar10 import CIFAR10
import numpy as np

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


#train = CIFAR10(which_set='train', gcn=55.)
#print(train.X.shape)
#print(train.X)
#print(train.y.shape)
#print(train.y)

file = "cifar10/pylearn2_gcn_whitened/train.pkl"
o = unpickle(file)
x = o.X
y = o.y

features = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
np.save("temp/temp.txt", features[0])
print(features.shape)