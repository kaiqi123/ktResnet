import cPickle


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

file = "cifar10/pylearn2_gcn_whitened/test.pkl"
dict = unpickle(file)
print(dict)

