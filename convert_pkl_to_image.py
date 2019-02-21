import cPickle
from pylearn2.datasets.cifar10 import CIFAR10
import numpy as np

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


train = CIFAR10(which_set='train', gcn=55.)
x = train.X
y_o = train.y
features_origin = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
np.save("temp/temp_origin", features_origin[0])
print(features_origin.shape)

file = "cifar10/pylearn2_gcn_whitened/train.pkl"
o = unpickle(file)
x = o.X
y = o.y

features = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
np.save("temp/temp_whiten", features[0])
print(features.shape)