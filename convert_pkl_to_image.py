import cPickle
from pylearn2.datasets.cifar10 import CIFAR10
import numpy as np

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


train = CIFAR10(which_set='train', gcn=55.)
features_origin = train.X.reshape((train.X.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
np.save("temp/temp_origin", features_origin[7000])
print(features_origin.shape)

np.save("cifar10/pylearn2_gcn_whitened/train_labels", train.y)
print(train.y.shape)

test = CIFAR10(which_set='test', gcn=55.)
np.save("cifar10/pylearn2_gcn_whitened/test_labels", test.y)
print(test.y.shape)

"""
file = "cifar10/pylearn2_gcn_whitened/train.pkl"
o = unpickle(file)
x = o.X
y = o.y

features = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
np.save("temp/temp_whiten", features[7000])
print(features.shape)
"""