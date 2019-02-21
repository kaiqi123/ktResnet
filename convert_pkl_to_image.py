import cPickle
from pylearn2.datasets.cifar10 import CIFAR10

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

file = "cifar10/pylearn2_gcn_whitened/test.pkl"
dict = unpickle(file)
print(dict)

train = CIFAR10(which_set='train', gcn=55.)
print(train.X.shape)
print(type(train.X))
print(train.Y.shape)
print(type(train.Y))