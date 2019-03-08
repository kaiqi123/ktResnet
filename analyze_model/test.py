import tensorflow as tf
import numpy as np
import re
import torch
from torch.autograd import Variable


def g(inputs, params):


    def tr(v):
        if v.ndim == 4:
            return v.transpose(2, 3, 1, 0)
        elif v.ndim == 2:
            return v.transpose()
        return v

    params = {k: tf.constant(tr(v)) for k, v in params.items()}

    def conv2d(x, params, name, stride=1, padding=0):
        x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        z = tf.nn.conv2d(x, params['%s.weight' % name], [1, stride, stride, 1], padding='VALID')
        if '%s.bias' % name in params:
            return tf.nn.bias_add(z, params['%s.bias' % name])
        else:
            return z

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o1 = tf.nn.relu(x)
            o = conv2d(o1, params, b_base + '0', stride=stride, padding=1)
            o = tf.nn.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, padding=1)
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
        return o

    # determine network size by parameters
    #blocks = [sum([re.match('group%d.block\d+.conv0.weight' % j, k) is not None
    #               for k in params.keys()]) for j in range(4)]
    depth = 28
    width = 10
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    o = conv2d(inputs, params, 'conv0', padding=1)
    o_g0 = group(o, params, 'group0', 1, n)
    o_g1 = group(o_g0, params, 'group1', 2, n)
    o_g2 = group(o_g1, params, 'group2', 2, n)

    o = tf.nn.relu(o_g2)
    o = tf.nn.avg_pool(o, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
    o = tf.reshape(o, [-1, int(np.prod(o.get_shape()[1:]))])
    o = tf.matmul(o, params['fc.weight']) + params['fc.bias']
    return o


inputs = torch.randn(1,3,224,224)

#y = f(Variable(inputs), params)
#print(y)

params = {k: v.detach().cpu().numpy() for k, v in torch.load('model.pt7')['params'].items()}
for k, v in sorted(params.items()):
    print(k, tuple(v.shape))

inputs_tf = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

out = g(inputs_tf, params)

sess = tf.Session()
y_tf = sess.run(out, feed_dict={inputs_tf: inputs.permute(0, 2, 3, 1).numpy()})
print(y_tf)
# check that difference between PyTorch and Tensorflow is small
#assert np.abs(y_tf - y.data.numpy()).max() < 1e-5