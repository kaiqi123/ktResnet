import cPickle
from pylearn2.datasets.cifar10 import CIFAR10

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


train = CIFAR10(which_set='train', gcn=55.)
print(train.X.shape)
print(train.X)
print(train.y.shape)
print(train.y)

file = "cifar10/pylearn2_gcn_whitened/train.pkl"
o = unpickle(file)
print(o)
print(o.X.shape)
print(o.X)
print(o.y.shape)
print(o.y)
